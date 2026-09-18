/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/internal/MetatomicEvaluator.h"
#include "openmmmetatomic/internal/HarmonicModel.h"
#include "openmm/OpenMMException.h"

#ifdef DIM
#undef DIM
#endif

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <unordered_set>

#ifdef OPENMM_METATOMIC_USE_VESIN
#include <vesin.h>
#endif

#ifdef OPENMM_METATOMIC_TORCH
#include <torch/script.h>
#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>
#endif

using namespace OpenMMMetatomic;
using namespace OpenMM;
using namespace std;

namespace {

string fileExtension(const string& path) {
    const auto dot = path.rfind('.');
    if (dot == string::npos || dot == path.size() - 1)
        return "";
    return path.substr(dot);
}

string resolveBackend(const string& requested, const string& path) {
    if (requested.empty() || requested == "auto") {
        const auto ext = fileExtension(path);
        if (ext == ".pt" || ext == ".pth")
            return "torch";
        return "core";
    }
    return requested;
}

void loadCorePlugins(const string& extensionsDirectory) {
    if (extensionsDirectory.empty())
        return;
    namespace fs = std::filesystem;
    const fs::path root(extensionsDirectory);
    auto loadOne = [](const fs::path& path) {
        metatomic::load_plugin(path.string());
    };
    if (fs::is_regular_file(root)) {
        loadOne(root);
        return;
    }
    if (!fs::is_directory(root)) {
        throw OpenMMException(
            "MetatomicForce: extensions directory '" + extensionsDirectory + "' does not exist"
        );
    }
    for (const auto& entry : fs::directory_iterator(root)) {
        const auto ext = entry.path().extension();
        if (ext == ".so" || ext == ".dylib" || ext == ".dll")
            loadOne(entry.path());
    }
}

// Backend-agnostic pair list: 5 int32 columns per pair (first_atom,
// second_atom, cell_shift_a/b/c) and 3 double columns (the pair vector),
// used to build a metatensor(_torch) TensorBlock for either backend.
struct RawNeighborPairs {
    vector<int32_t> samples;
    vector<double> vectors;

    size_t size() const {
        return samples.size() / 5;
    }
};

bool useVesinNeighbors() {
#ifdef OPENMM_METATOMIC_USE_VESIN
    const char* nl = std::getenv("OPENMM_METATOMIC_NEIGHBOR_LIST");
    if (nl == nullptr || nl[0] == '\0')
        return true;
    return std::strcmp(nl, "naive") != 0 && std::strcmp(nl, "fallback") != 0;
#else
    return false;
#endif
}

RawNeighborPairs computeNeighborPairsNaive(
    const vector<Vec3>& positions, const Vec3 box[3], const array<bool, 3>& pbc,
    double cutoff, bool full
) {
    const int n = static_cast<int>(positions.size());
    const double cutoff2 = cutoff * cutoff;
    RawNeighborPairs out;
    out.samples.reserve(static_cast<size_t>(n) * 10);
    out.vectors.reserve(static_cast<size_t>(n) * 6);
    auto length = [](const Vec3& v) {
        return std::sqrt(v.dot(v));
    };
    auto images = [&](const Vec3& v, bool axisPeriodic) {
        if (!axisPeriodic)
            return 0;
        const double len = length(v);
        if (len < 1e-12)
            throw OpenMMException("MetatomicForce: periodic box vector has zero length");
        return max(1, static_cast<int>(std::ceil(cutoff / len)));
    };
    const int na = images(box[0], pbc[0]);
    const int nb = images(box[1], pbc[1]);
    const int nc = images(box[2], pbc[2]);
    auto consider = [&](int i, int j, int sa, int sb, int sc) {
        if (i == j && sa == 0 && sb == 0 && sc == 0)
            return;
        if (!full && (i > j || (i == j && (sa < 0 || (sa == 0 && sb < 0) || (sa == 0 && sb == 0 && sc <= 0)))))
            return;
        const Vec3 shift = sa * box[0] + sb * box[1] + sc * box[2];
        const Vec3 delta = positions[j] - positions[i] + shift;
        if (delta.dot(delta) > cutoff2)
            return;
        out.samples.push_back(i);
        out.samples.push_back(j);
        out.samples.push_back(sa);
        out.samples.push_back(sb);
        out.samples.push_back(sc);
        out.vectors.push_back(delta[0]);
        out.vectors.push_back(delta[1]);
        out.vectors.push_back(delta[2]);
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int sa = -na; sa <= na; sa++)
                for (int sb = -nb; sb <= nb; sb++)
                    for (int sc = -nc; sc <= nc; sc++)
                        consider(i, j, sa, sb, sc);
        }
    }
    return out;
}

/**
 * Verlet skin, in nm, for vesin's cached topology; 0 disables caching.
 *
 * vesin builds its candidate list with cutoff + skin and reuses it until an
 * atom moves more than skin / 2 (it tracks the cell as well, so a barostat is
 * safe). The returned pairs are always inside the cutoff either way.
 *
 * The default is measured, not guessed: on a 0.75 nm full list (what PET-MAD-XS
 * asks for) a cached call costs 0.42 ms at 288 atoms and 5.5 ms at 3,000
 * against 0.71 and 8.2 ms for a fresh build, and a rebuild costs 2.1 and
 * 24 ms. Larger skins make every step pay for more candidates and make the
 * rebuilds much more expensive, so they lose on both counts.
 */
double neighborSkinNm() {
    // Read every call, like OPENMM_METATOMIC_NEIGHBOR_LIST above: it costs a
    // getenv against a pair-list build, and it keeps the knob testable in a
    // single process.
    const char* value = std::getenv("OPENMM_METATOMIC_NEIGHBOR_SKIN");
    if (value == nullptr || value[0] == '\0')
        return 0.05;
    try {
        return max(0.0, std::stod(value));
    }
    catch (const exception&) {
        throw OpenMMException(
            "MetatomicForce: OPENMM_METATOMIC_NEIGHBOR_SKIN must be a length in nm"
        );
    }
}

/**
 * One of the model's pair-list requests, kept alive across compute() calls.
 *
 * Holding on to vesin's list is what makes the skin cache work at all: the
 * cached topology lives inside the list object, so freeing it after every
 * evaluation (as this code used to) threw the cache away every step. The output
 * buffers are reused for the same reason.
 */
class NeighborList {
public:
    NeighborList() = default;
    NeighborList(const NeighborList&) = delete;
    NeighborList& operator=(const NeighborList&) = delete;

    ~NeighborList() {
#ifdef OPENMM_METATOMIC_USE_VESIN
        vesin_free(&cached);
#endif
    }

    const RawNeighborPairs& compute(
        const vector<Vec3>& positions, const Vec3 box[3], const array<bool, 3>& pbc,
        double cutoff, bool full
    ) {
#ifdef OPENMM_METATOMIC_USE_VESIN
        if (useVesinNeighbors()) {
            buildWithVesin(positions, box, pbc, cutoff, full);
            return pairs;
        }
#endif
        pairs = computeNeighborPairsNaive(positions, box, pbc, cutoff, full);
        return pairs;
    }

private:
#ifdef OPENMM_METATOMIC_USE_VESIN
    void buildWithVesin(
        const vector<Vec3>& positions, const Vec3 box[3], const array<bool, 3>& pbc,
        double cutoff, bool full
    ) {
        const size_t n = positions.size();
        points.resize(n);
        for (size_t i = 0; i < n; i++) {
            points[i][0] = positions[i][0];
            points[i][1] = positions[i][1];
            points[i][2] = positions[i][2];
        }
        double cell[3][3] = {
            {box[0][0], box[0][1], box[0][2]},
            {box[1][0], box[1][1], box[1][2]},
            {box[2][0], box[2][1], box[2][2]},
        };
        bool vesinPbc[3] = {pbc[0], pbc[1], pbc[2]};
        VesinOptions options{};
        options.cutoff = cutoff;
        options.full = full;
        options.sorted = false;
        options.algorithm = VesinAutoAlgorithm;
        options.skin = neighborSkinNm();
        options.n_threads = 0;
        options.return_shifts = true;
        options.return_distances = false;
        options.return_vectors = true;
        const char* error = nullptr;
        const int status = vesin_neighbors(
            reinterpret_cast<const double(*)[3]>(points.data()),
            n,
            cell,
            vesinPbc,
            VesinDevice{VesinCPU, 0},
            options,
            &cached,
            &error
        );
        if (status != 0) {
            string message = error ? error : "vesin neighbor list failed";
            vesin_free(&cached);
            cached = VesinNeighborList{};
            throw OpenMMException("MetatomicForce: " + message);
        }
        pairs.samples.resize(cached.length * 5);
        pairs.vectors.resize(cached.length * 3);
        for (size_t k = 0; k < cached.length; k++) {
            pairs.samples[5 * k + 0] = static_cast<int32_t>(cached.pairs[k][0]);
            pairs.samples[5 * k + 1] = static_cast<int32_t>(cached.pairs[k][1]);
            pairs.samples[5 * k + 2] = cached.shifts ? cached.shifts[k][0] : 0;
            pairs.samples[5 * k + 3] = cached.shifts ? cached.shifts[k][1] : 0;
            pairs.samples[5 * k + 4] = cached.shifts ? cached.shifts[k][2] : 0;
            pairs.vectors[3 * k + 0] = cached.vectors[k][0];
            pairs.vectors[3 * k + 1] = cached.vectors[k][1];
            pairs.vectors[3 * k + 2] = cached.vectors[k][2];
        }
    }

    VesinNeighborList cached{};
    vector<array<double, 3>> points;
#endif
    RawNeighborPairs pairs;
};

// Only periodic directions contribute a cell row; a non-periodic direction is
// zeroed, matching what OpenMM-ML sends for a partially periodic box.
vector<double> cellRows(const Vec3 box[3], const array<bool, 3>& pbc) {
    vector<double> cell(9, 0.0);
    for (int i = 0; i < 3; i++) {
        if (!pbc[i])
            continue;
        cell[3 * i + 0] = box[i][0];
        cell[3 * i + 1] = box[i][1];
        cell[3 * i + 2] = box[i][2];
    }
    return cell;
}

// Match an available output name against a requested base name and an optional
// variant: "energy" or "energy/<variant>".
string pickOutputName(
    const vector<string>& available, const string& base, const string& variant
) {
    const string wanted = variant.empty() ? base : base + "/" + variant;
    for (const auto& name : available)
        if (name == wanted)
            return name;
    if (!variant.empty())
        throw OpenMMException(
            "MetatomicForce: model does not provide output '" + wanted + "'"
        );
    return "";
}

void warnUncertainty(const vector<double>& uncertainty, double threshold) {
    string atoms;
    for (size_t i = 0; i < uncertainty.size(); i++) {
        if (uncertainty[i] <= threshold)
            continue;
        if (!atoms.empty())
            atoms += ", ";
        atoms += to_string(i);
    }
    if (atoms.empty())
        return;
    fprintf(
        stderr,
        "MetatomicForce warning: per-atom energy uncertainty is above the threshold "
        "of %g eV for atoms %s\n",
        threshold, atoms.c_str()
    );
}

// A per-system scalar input ("charge", "spin_multiplicity"): a single block
// with one sample named "system" and one property named after the input.
metatensor::TensorMap systemScalar(const string& name, double value) {
    auto values = make_unique<metatensor::SimpleDataArray<double>>(
        vector<uintptr_t>{1, 1}, vector<double>{value}
    );
    metatensor::TensorBlock block(
        std::move(values),
        metatensor::Labels({"system"}, {{0}}),
        {},
        metatensor::Labels({name}, {{0}})
    );
    vector<metatensor::TensorBlock> blocks;
    blocks.push_back(std::move(block));
    metatensor::TensorMap tensor(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
    tensor.set_info("unit", name == "charge" ? "e" : "");
    return tensor;
}

// Core backend: attach a pair list to a metatomic::System (C++ API) using
// the same raw pair computation as the torch backend below.
void addPairsCore(
    metatomic::System& system,
    const metatomic::PairListOptions& options,
    NeighborList& neighbors,
    const vector<Vec3>& positions,
    const Vec3 box[3],
    const array<bool, 3>& pbc,
    double cutoffNm,
    bool checkConsistency
) {
    const auto& raw = neighbors.compute(positions, box, pbc, cutoffNm, options.full_list());
    const auto nPairs = static_cast<uintptr_t>(raw.size());

    const vector<string> sampleNames = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };
    metatensor::Labels samples = checkConsistency
        ? metatensor::Labels(sampleNames, raw.samples.data(), nPairs)
        : metatensor::Labels(sampleNames, raw.samples.data(), nPairs, metatensor::assume_unique{});
    auto xyz = metatensor::Labels({"xyz"}, {{0}, {1}, {2}});
    auto properties = metatensor::Labels({"distance"}, {{0}});

    auto values = make_unique<metatensor::SimpleDataArray<double>>(
        vector<uintptr_t>{nPairs, 3, 1}, raw.vectors
    );
    metatensor::TensorBlock block(std::move(values), samples, {xyz}, properties);
    system.add_pairs(options, std::move(block));
}

#ifdef OPENMM_METATOMIC_TORCH
torch::Device selectTorchDevice(const vector<string>& supported, const string& desired) {
    torch::optional<string> requested = torch::nullopt;
    if (!desired.empty())
        requested = desired;
    const auto type = metatomic_torch::pick_device(supported, requested);
    if (requested.has_value())
        return torch::Device(*requested);
    return torch::Device(type);
}

torch::Dtype parseDtype(const string& name) {
    if (name == "float64")
        return torch::kFloat64;
    if (name == "float32")
        return torch::kFloat32;
    throw OpenMMException("MetatomicForce: unsupported model dtype '" + name + "'");
}

// Torch mirror of systemScalar(): a single-block TensorMap holding one
// per-system value, tagged with the unit metatomic expects for that input.
metatensor_torch::TensorMap systemScalarTorch(
    const string& name, double value, torch::Dtype dtype, torch::Device device
) {
    auto intOptions = torch::TensorOptions().dtype(torch::kInt32);
    auto samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        vector<string>{"system"}, torch::zeros({1, 1}, intOptions)
    );
    auto properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        vector<string>{name}, torch::zeros({1, 1}, intOptions)
    );
    auto values = torch::full({1, 1}, value, torch::TensorOptions().dtype(dtype));
    auto block = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
        values, samples, vector<metatensor_torch::Labels>{}, properties
    );
    auto keys = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        vector<string>{"_"}, torch::zeros({1, 1}, intOptions)
    );
    auto tensor = torch::make_intrusive<metatensor_torch::TensorMapHolder>(
        keys, vector<metatensor_torch::TensorBlock>{block}
    );
    tensor->set_info("unit", name == "charge" ? "e" : "");
    return tensor->to(dtype, device);
}

void addNeighborList(
    metatomic_torch::System& system,
    const metatomic_torch::NeighborListOptions& request,
    NeighborList& neighborList,
    const vector<Vec3>& positions,
    const Vec3 box[3],
    const array<bool, 3>& pbc,
    bool checkConsistency,
    torch::Device device,
    torch::Dtype dtype
) {
    const double cutoff = request->engine_cutoff("nm");
    const bool full = request->full_list();
    const auto& raw = neighborList.compute(positions, box, pbc, cutoff, full);
    const int64_t nPairs = static_cast<int64_t>(raw.size());

    // from_blob does not own these buffers, and the list reuses them on the next
    // call, so both tensors have to be cloned before they outlive this scope.
    auto sampleTensor = torch::from_blob(
        const_cast<int32_t*>(raw.samples.data()), {nPairs, 5},
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone().to(device);
    auto vectorTensor = torch::from_blob(
        const_cast<double*>(raw.vectors.data()), {nPairs, 3, 1},
        torch::TensorOptions().dtype(torch::kFloat64)
    ).clone().to(device, dtype);

    const vector<string> sampleNames = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };
    metatensor_torch::Labels neighborSamples;
    if (checkConsistency) {
        neighborSamples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            sampleNames, sampleTensor
        );
    }
    else {
        neighborSamples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            sampleNames, sampleTensor, metatensor::assume_unique{}
        );
    }
    auto xyz = torch::tensor({0, 1, 2}, torch::TensorOptions().dtype(torch::kInt32).device(device)).reshape({3, 1});
    auto component = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        vector<string>{"xyz"}, xyz
    );
    auto properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        vector<string>{"distance"}, torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt32).device(device))
    );
    auto neighbors = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
        vectorTensor, neighborSamples, vector<metatensor_torch::Labels>{component}, properties
    );
    metatomic_torch::register_autograd_neighbors(system, neighbors, checkConsistency);
    system->add_neighbor_list(request, neighbors);
}
#endif

} // namespace

class OpenMMMetatomic::MetatomicEvaluatorImpl {
public:
    virtual ~MetatomicEvaluatorImpl() = default;
    virtual const MetatomicEvaluator::ModelInfo& info() const = 0;
    virtual MetatomicEvaluator::Result compute(const vector<Vec3>& positions, const Vec3 box[3]) const = 0;
};

namespace {

class CoreEvaluatorImpl final : public MetatomicEvaluatorImpl {
public:
    explicit CoreEvaluatorImpl(const MetatomicEvaluator::Config& config) {
        loadCorePlugins(config.extensionsDirectory);
        if (config.modelPath == "harmonic") {
            model = make_unique<HarmonicModel>(
                1.0, vector<double>(3 * config.atomicTypes.size(), 0.0)
            );
        }
        else if (config.modelPath == "harmonic-nl") {
            // Same built-in model as "harmonic", but requests a pair list --
            // for testing the core backend's pair-list support itself.
            model = make_unique<NeighborHarmonicModel>(
                1.0, vector<double>(3 * config.atomicTypes.size(), 0.0), 0.3
            );
        }
        else {
            try {
                model = make_unique<metatomic::ExternalModel>(
                    metatomic::load_model(config.modelPath)
                );
            }
            catch (const exception& e) {
                throw OpenMMException(
                    "MetatomicForce: failed to load model '" + config.modelPath + "': " + e.what()
                );
            }
        }

        const auto caps = model->capabilities();
        info_.backend = "core";
        info_.dtype = caps.dtype() == metatomic::ModelCapabilities::DType::Float64 ? "float64" : "float32";
        info_.lengthUnit = caps.length_unit();
        info_.device = "cpu";
        info_.atomicTypes = caps.atomic_types();
        for (auto device : caps.supported_devices()) {
            if (device == metatomic::ModelCapabilities::Device::CPU)
                info_.supportedDevices.push_back("cpu");
            else if (device == metatomic::ModelCapabilities::Device::CUDA)
                info_.supportedDevices.push_back("cuda");
            else if (device == metatomic::ModelCapabilities::Device::ROCM)
                info_.supportedDevices.push_back("rocm");
            else if (device == metatomic::ModelCapabilities::Device::Metal)
                info_.supportedDevices.push_back("metal");
        }

        unordered_set<int64_t> allowed(info_.atomicTypes.begin(), info_.atomicTypes.end());
        typesHost.resize(config.atomicTypes.size());
        for (size_t i = 0; i < config.atomicTypes.size(); i++) {
            const int type = config.atomicTypes[i];
            if (!allowed.count(type)) {
                throw OpenMMException(
                    "MetatomicForce: this model does not support atomic type " + to_string(type)
                );
            }
            typesHost[i] = static_cast<int32_t>(type);
        }

        pairLists = model->requested_pair_lists();
        info_.neighborListRequests = static_cast<int>(pairLists.size());
        // PairListOptions::cutoff() is in "the length unit of the model";
        // compute() always builds its System in "nm" (see makeSystem below),
        // so convert once here rather than on every compute() call.
        const double lengthToNm = metatomic::unit_conversion_factor(info_.lengthUnit, "nm");
        pairListCutoffsNm.reserve(pairLists.size());
        for (const auto& request : pairLists)
            pairListCutoffsNm.push_back(request.cutoff() * lengthToNm);
        neighbors.reserve(pairLists.size());
        for (size_t i = 0; i < pairLists.size(); i++)
            neighbors.push_back(make_unique<NeighborList>());
        for (const auto& input : model->requested_inputs()) {
            const string& name = input.name();
            info_.requestedInputs.push_back(name);
            if (input.sample_kind() != metatomic::SampleKind::System)
                throw OpenMMException(
                    "MetatomicForce: this model requests per-atom input '" + name +
                    "', which MetatomicForce does not provide"
                );
            if (name == "charge")
                extraInputs.emplace_back(name, config.charge);
            else if (name == "spin_multiplicity")
                extraInputs.emplace_back(name, config.spinMultiplicity);
            else
                throw OpenMMException(
                    "MetatomicForce: this model requests extra input '" + name +
                    "', which MetatomicForce does not provide (only charge and "
                    "spin_multiplicity are supported)"
                );
        }

        vector<string> available;
        for (const auto& output : caps.outputs())
            available.push_back(output.name());
        auto variantFor = [&](const string& base) {
            const auto found = config.variants.find(base);
            return found == config.variants.end() ? string() : found->second;
        };
        info_.energyKey = pickOutputName(available, "energy", variantFor("energy"));
        if (info_.energyKey.empty()) {
            throw OpenMMException(
                "MetatomicForce: model '" + config.modelPath + "' does not provide an energy output"
            );
        }
        if (config.nonConservativeForces) {
            info_.nonConservativeForceKey = pickOutputName(
                available, "non_conservative_force", variantFor("non_conservative_force")
            );
            if (info_.nonConservativeForceKey.empty())
                throw OpenMMException(
                    "MetatomicForce: nonConservative=\"forces\" needs a "
                    "non_conservative_force output, which model '" + config.modelPath +
                    "' does not provide"
                );
        }
        if (config.nonConservativeStress) {
            // execute_model would happily return one, but nothing downstream can
            // use it, and unlike the torch path there is no warning worth
            // printing for a knob that does nothing here.
            throw OpenMMException(
                "MetatomicForce: a non-conservative stress is not supported on the "
                "core backend; use backend=\"torch\" (and note OpenMM cannot "
                "consume a stress either way)"
            );
        }
        if (config.uncertaintyThreshold >= 0.0) {
            info_.uncertaintyKey = pickOutputName(
                available, "energy_uncertainty", variantFor("energy_uncertainty")
            );
        }
        uncertaintyThreshold = config.uncertaintyThreshold;
        checkConsistency = config.checkConsistency;
        pbc = config.pbc;
    }

    const MetatomicEvaluator::ModelInfo& info() const override {
        return info_;
    }

    MetatomicEvaluator::Result compute(const vector<Vec3>& positions, const Vec3 box[3]) const override {
        if (positions.size() != typesHost.size()) {
            throw OpenMMException(
                "MetatomicForce: expected " + to_string(typesHost.size()) +
                " positions, got " + to_string(positions.size())
            );
        }
        const size_t n = positions.size();
        // Vec3 is exactly {double data[3]}, so a vector<Vec3> is already a
        // contiguous block of 3*n doubles: build `pos` with one memcpy-able
        // range-construction instead of indexing through Vec3::operator[]
        // per component, per atom.
        static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 layout changed");
        const double* flatPositions = reinterpret_cast<const double*>(positions.data());
        vector<double> pos(flatPositions, flatPositions + 3 * n);
        auto cell = cellRows(box, pbc);
        try {
            auto system = makeSystem("nm", typesHost, pos, pbc, cell);
            for (size_t i = 0; i < pairLists.size(); i++)
                addPairsCore(
                    system, pairLists[i], *neighbors[i], positions, box, pbc,
                    pairListCutoffsNm[i], checkConsistency
                );
            for (const auto& input : extraInputs)
                system.add_custom_data(input.first, systemScalar(input.first, input.second));
            vector<metatomic::System> systems;
            systems.push_back(std::move(system));

            const bool autograd = info_.nonConservativeForceKey.empty();
            vector<metatomic::Quantity> requests;
            auto energy = metatomic::Quantity::builder()
                .name(info_.energyKey)
                .unit("kJ/mol")
                .sample_kind(metatomic::SampleKind::System);
            if (autograd)
                energy.add_gradient(metatomic::Gradients::Positions);
            requests.push_back(energy.build());
            if (!autograd)
                requests.push_back(
                    metatomic::Quantity::builder()
                        .name(info_.nonConservativeForceKey)
                        .unit("kJ/mol/nm")
                        .sample_kind(metatomic::SampleKind::Atom)
                        .build()
                );
            if (!info_.uncertaintyKey.empty())
                requests.push_back(
                    metatomic::Quantity::builder()
                        .name(info_.uncertaintyKey)
                        .unit("eV")
                        .sample_kind(metatomic::SampleKind::Atom)
                        .build()
                );

            auto results = metatomic::execute_model(
                *model, systems, std::nullopt, requests, checkConsistency
            );
            if (results.size() != requests.size())
                throw OpenMMException("MetatomicForce: model returned the wrong number of outputs");

            MetatomicEvaluator::Result result;
            auto energyBlock = results[0].block_by_id(0);
            auto energyValues = energyBlock.values<double>();
            result.energy = energyValues(0, 0);
            result.forces.resize(n);
            if (autograd) {
                auto positionGradient = energyBlock.gradient("positions");
                auto gradient = positionGradient.values<double>();
                for (size_t i = 0; i < n; i++)
                    result.forces[i] = Vec3(
                        -gradient(i, 0, 0), -gradient(i, 1, 0), -gradient(i, 2, 0)
                    );
            }
            else {
                auto forceBlock = results[1].block_by_id(0);
                auto values = forceBlock.values<double>();
                Vec3 mean;
                for (size_t i = 0; i < n; i++) {
                    result.forces[i] = Vec3(values(i, 0, 0), values(i, 1, 0), values(i, 2, 0));
                    mean += result.forces[i];
                }
                // A direct force head can predict a non-zero total force; the
                // metatomic docs ask engines to remove it to avoid drift.
                mean *= 1.0 / static_cast<double>(n);
                for (size_t i = 0; i < n; i++)
                    result.forces[i] -= mean;
            }
            if (!info_.uncertaintyKey.empty()) {
                auto uncertaintyBlock = results.back().block_by_id(0);
                auto values = uncertaintyBlock.values<double>();
                vector<double> uncertainty(n);
                for (size_t i = 0; i < n; i++)
                    uncertainty[i] = values(i, 0);
                result.maxUncertainty = *max_element(uncertainty.begin(), uncertainty.end());
                warnUncertainty(uncertainty, uncertaintyThreshold);
            }
            return result;
        }
        catch (const exception& e) {
            throw OpenMMException(string("MetatomicForce: model evaluation failed: ") + e.what());
        }
    }

    mutable unique_ptr<metatomic::BaseModel> model;
    vector<int32_t> typesHost;
    vector<metatomic::PairListOptions> pairLists;
    vector<double> pairListCutoffsNm;
    /// One per request, kept between compute() calls for the skin cache.
    mutable vector<unique_ptr<NeighborList>> neighbors;
    vector<pair<string, double>> extraInputs;
    bool checkConsistency = false;
    array<bool, 3> pbc = {false, false, false};
    double uncertaintyThreshold = -1.0;
    MetatomicEvaluator::ModelInfo info_;
};

#ifdef OPENMM_METATOMIC_TORCH
class TorchEvaluatorImpl final : public MetatomicEvaluatorImpl {
public:
    explicit TorchEvaluatorImpl(const MetatomicEvaluator::Config& config) {
        torch::optional<string> extensions = torch::nullopt;
        if (!config.extensionsDirectory.empty())
            extensions = config.extensionsDirectory;
        try {
            model = metatomic_torch::load_atomistic_model(config.modelPath, extensions);
        }
        catch (const exception& e) {
            throw OpenMMException(
                "MetatomicForce: failed to load TorchScript model '" + config.modelPath + "': " + e.what()
            );
        }

        capabilities = model.run_method("capabilities")
                          .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
        info_.backend = "torch";
        info_.dtype = capabilities->dtype();
        info_.lengthUnit = capabilities->length_unit();
        info_.supportedDevices = capabilities->supported_devices;
        info_.atomicTypes = capabilities->atomic_types;
        dtype = parseDtype(info_.dtype);
        device = selectTorchDevice(info_.supportedDevices, config.device);
        info_.device = device.str();
        model.to(device);

        unordered_set<int64_t> allowed(info_.atomicTypes.begin(), info_.atomicTypes.end());
        typesHost.resize(config.atomicTypes.size());
        for (size_t i = 0; i < config.atomicTypes.size(); i++) {
            const int type = config.atomicTypes[i];
            if (!allowed.count(type)) {
                throw OpenMMException(
                    "MetatomicForce: this model does not support atomic type " + to_string(type)
                );
            }
            typesHost[i] = type;
        }
        types = torch::tensor(typesHost, torch::TensorOptions().dtype(torch::kInt32)).to(device);

        auto requests = model.run_method("requested_neighbor_lists").toList();
        for (const auto& request : requests) {
            neighborRequests.push_back(
                request.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>()
            );
        }
        info_.neighborListRequests = static_cast<int>(neighborRequests.size());
        neighbors.reserve(neighborRequests.size());
        for (size_t i = 0; i < neighborRequests.size(); i++)
            neighbors.push_back(make_unique<NeighborList>());

        auto requestedInputs = model.run_method("requested_inputs", /*use_new_names=*/true).toGenericDict();
        for (const auto& entry : requestedInputs) {
            const string name(entry.key().toStringRef());
            info_.requestedInputs.push_back(name);
            auto option = entry.value().toCustomClass<metatomic_torch::ModelOutputHolder>();
            if (option->sample_kind() != "system")
                throw OpenMMException(
                    "MetatomicForce: this model requests per-atom input '" + name +
                    "', which MetatomicForce does not provide"
                );
            if (name == "charge")
                extraInputs.emplace_back(name, systemScalarTorch(name, config.charge, dtype, device));
            else if (name == "spin_multiplicity")
                extraInputs.emplace_back(
                    name, systemScalarTorch(name, config.spinMultiplicity, dtype, device)
                );
            else
                throw OpenMMException(
                    "MetatomicForce: this model requests extra input '" + name +
                    "', which MetatomicForce does not provide (only charge and "
                    "spin_multiplicity are supported)"
                );
        }

        auto outputs = capabilities->outputs();
        auto variantFor = [&](const string& base) -> torch::optional<string> {
            const auto found = config.variants.find(base);
            if (found == config.variants.end())
                return torch::nullopt;
            return found->second;
        };
        options = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
        options->set_length_unit("nm");

        info_.energyKey = metatomic_torch::pick_output("energy", outputs, variantFor("energy"));
        if (!outputs.contains(info_.energyKey)) {
            throw OpenMMException(
                "MetatomicForce: model '" + config.modelPath +
                "' does not provide an energy output"
            );
        }
        auto energyOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        energyOut->set_sample_kind(outputs.at(info_.energyKey)->sample_kind());
        energyOut->set_unit("kJ/mol");
        options->outputs.insert(info_.energyKey, energyOut);

        if (config.nonConservativeForces) {
            bool hasForce = false;
            for (const auto& entry : outputs)
                hasForce = hasForce || entry.key().find("non_conservative_force") != string::npos;
            if (!hasForce)
                throw OpenMMException(
                    "MetatomicForce: nonConservative=\"forces\" needs a "
                    "non_conservative_force output, which model '" + config.modelPath +
                    "' does not provide"
                );
            info_.nonConservativeForceKey = metatomic_torch::pick_output(
                "non_conservative_force", outputs, variantFor("non_conservative_force")
            );
            auto forceOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
            forceOut->set_sample_kind("atom");
            forceOut->set_unit("kJ/mol/nm");
            options->outputs.insert(info_.nonConservativeForceKey, forceOut);
        }
        if (config.nonConservativeStress) {
            // Requested for completeness and to surface a missing output early;
            // OpenMM has no virial path, so nothing consumes the stress. NPT goes
            // through a MonteCarlo barostat, which only needs the energy.
            bool hasStress = false;
            for (const auto& entry : outputs)
                hasStress = hasStress || entry.key().find("non_conservative_stress") != string::npos;
            if (!hasStress)
                throw OpenMMException(
                    "MetatomicForce: nonConservative asked for a stress, which model '" +
                    config.modelPath + "' does not provide"
                );
            info_.nonConservativeStressKey = metatomic_torch::pick_output(
                "non_conservative_stress", outputs, variantFor("non_conservative_stress")
            );
            fprintf(
                stderr,
                "MetatomicForce warning: OpenMM cannot use a non-conservative stress; "
                "use a MonteCarloBarostat for NPT\n"
            );
            auto stressOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
            stressOut->set_sample_kind("system");
            stressOut->set_unit("kJ/mol/nm^3");
            options->outputs.insert(info_.nonConservativeStressKey, stressOut);
        }
        // A model without an uncertainty head simply does not get checked, the
        // same choice OpenMM-ML makes.
        bool hasUncertainty = false;
        for (const auto& entry : outputs)
            hasUncertainty = hasUncertainty || entry.key().find("energy_uncertainty") != string::npos;
        if (config.uncertaintyThreshold >= 0.0 && hasUncertainty) {
            info_.uncertaintyKey = metatomic_torch::pick_output(
                "energy_uncertainty", outputs, variantFor("energy_uncertainty")
            );
            auto uncertaintyOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
            uncertaintyOut->set_sample_kind("atom");
            uncertaintyOut->set_unit("eV");
            options->outputs.insert(info_.uncertaintyKey, uncertaintyOut);
        }
        uncertaintyThreshold = config.uncertaintyThreshold;

        checkConsistency = config.checkConsistency;
        this->pbcFlags = config.pbc;
        pbc = torch::tensor(
            {config.pbc[0], config.pbc[1], config.pbc[2]},
            torch::TensorOptions().dtype(torch::kBool)
        ).to(device);
    }

    const MetatomicEvaluator::ModelInfo& info() const override {
        return info_;
    }

    MetatomicEvaluator::Result compute(const vector<Vec3>& positions, const Vec3 box[3]) const override {
        if (positions.size() != typesHost.size()) {
            throw OpenMMException(
                "MetatomicForce: expected " + to_string(typesHost.size()) +
                " positions, got " + to_string(positions.size())
            );
        }
        const int64_t n = static_cast<int64_t>(positions.size());
        const bool autograd = info_.nonConservativeForceKey.empty();
        auto posOptions = torch::TensorOptions().dtype(torch::kFloat64);
        auto posCpu = torch::from_blob(
            const_cast<Vec3*>(positions.data()), {n, 3}, posOptions
        ).clone();
        auto pos = posCpu.to(device, dtype);
        if (autograd)
            pos.set_requires_grad(true);

        const auto cellHost = cellRows(box, pbcFlags);
        auto cell = torch::from_blob(
            const_cast<double*>(cellHost.data()), {3, 3}, posOptions
        ).clone().to(device, dtype);

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(types, pos, cell, pbc);
        for (size_t i = 0; i < neighborRequests.size(); i++)
            addNeighborList(
                system, neighborRequests[i], *neighbors[i], positions, box, pbcFlags,
                checkConsistency, device, dtype
            );
        for (const auto& input : extraInputs)
            system->add_data(input.first, input.second, /*override=*/false);

        c10::IValue output;
        try {
            output = model.forward({vector<metatomic_torch::System>{system}, options, checkConsistency});
        }
        catch (const exception& e) {
            throw OpenMMException(string("MetatomicForce: model evaluation failed: ") + e.what());
        }
        auto dict = output.toGenericDict();
        auto energyMap = dict.at(info_.energyKey).toCustomClass<metatensor_torch::TensorMapHolder>();
        auto energyBlock = metatensor_torch::TensorMapHolder::block_by_id(energyMap, 0);
        auto energyTensor = energyBlock->values().sum();

        MetatomicEvaluator::Result result;
        result.energy = energyTensor.item<double>();
        result.forces.resize(static_cast<size_t>(n));
        static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 layout changed");

        torch::Tensor forceCpu;
        if (autograd) {
            energyTensor.backward();
            auto grad = system->positions().grad();
            if (!grad.defined()) {
                throw OpenMMException(
                    "MetatomicForce: model energy does not depend on positions; cannot compute forces"
                );
            }
            forceCpu = (-grad).to(torch::kCPU).to(torch::kFloat64).contiguous();
        }
        else {
            auto forceMap = dict.at(info_.nonConservativeForceKey)
                                .toCustomClass<metatensor_torch::TensorMapHolder>();
            auto forceBlock = metatensor_torch::TensorMapHolder::block_by_id(forceMap, 0);
            auto forces = forceBlock->values().reshape({n, 3});
            // A direct force head can predict a non-zero total force; the
            // metatomic docs ask engines to remove it to avoid drift.
            forces = forces - forces.mean(0, /*keepdim=*/true);
            forceCpu = forces.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
        }
        std::memcpy(result.forces.data(), forceCpu.data_ptr<double>(), 3 * static_cast<size_t>(n) * sizeof(double));

        if (!info_.uncertaintyKey.empty()) {
            auto map = dict.at(info_.uncertaintyKey).toCustomClass<metatensor_torch::TensorMapHolder>();
            auto block = metatensor_torch::TensorMapHolder::block_by_id(map, 0);
            auto values = block->values().detach().reshape({n}).to(torch::kCPU).to(torch::kFloat64).contiguous();
            const double* data = values.data_ptr<double>();
            vector<double> uncertainty(data, data + n);
            result.maxUncertainty = *max_element(uncertainty.begin(), uncertainty.end());
            warnUncertainty(uncertainty, uncertaintyThreshold);
        }
        return result;
    }

    mutable metatensor_torch::Module model = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities capabilities;
    metatomic_torch::ModelEvaluationOptions options;
    vector<metatomic_torch::NeighborListOptions> neighborRequests;
    /// One per request, kept between compute() calls for the skin cache.
    mutable vector<unique_ptr<NeighborList>> neighbors;
    torch::Tensor types;
    vector<int> typesHost;
    torch::Tensor pbc;
    array<bool, 3> pbcFlags = {false, false, false};
    vector<pair<string, metatensor_torch::TensorMap>> extraInputs;
    torch::Device device = torch::kCPU;
    torch::Dtype dtype = torch::kFloat32;
    bool checkConsistency = false;
    double uncertaintyThreshold = -1.0;
    MetatomicEvaluator::ModelInfo info_;
};
#endif

unique_ptr<MetatomicEvaluatorImpl> makeImpl(const MetatomicEvaluator::Config& config) {
    const auto backend = resolveBackend(config.backend, config.modelPath);
    if (backend == "torch") {
#ifdef OPENMM_METATOMIC_TORCH
        return make_unique<TorchEvaluatorImpl>(config);
#else
        throw OpenMMException(
            "MetatomicForce: TorchScript backend was not compiled. "
            "Rebuild with OPENMM_METATOMIC_TORCH=ON or use setBackend(\"core\")."
        );
#endif
    }
    if (backend == "core")
        return make_unique<CoreEvaluatorImpl>(config);
    throw OpenMMException("MetatomicForce: unknown backend '" + backend + "'");
}

} // namespace

MetatomicEvaluator::MetatomicEvaluator(const Config& config) :
    impl(makeImpl(config)) {
}

MetatomicEvaluator::~MetatomicEvaluator() = default;

const MetatomicEvaluator::ModelInfo& MetatomicEvaluator::info() const {
    return impl->info();
}

MetatomicEvaluator::Result MetatomicEvaluator::compute(const vector<Vec3>& positions,
                                                       const Vec3 boxVectors[3]) const {
    return impl->compute(positions, boxVectors);
}
