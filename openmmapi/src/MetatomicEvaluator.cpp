/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/internal/MetatomicEvaluator.h"
#include "openmmmetatomic/internal/HarmonicModel.h"
#include "openmm/OpenMMException.h"

#ifdef DIM
#undef DIM
#endif

#include <cmath>
#include <filesystem>
#include <sstream>
#include <unordered_set>

#ifdef OPENMM_METATOMIC_TORCH
#include <torch/script.h>
#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>
#ifdef OPENMM_METATOMIC_USE_VESIN
#include <vesin.h>
#endif
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

void addNeighborList(
    metatomic_torch::System& system,
    const metatomic_torch::NeighborListOptions& request,
    const vector<Vec3>& positions,
    const Vec3 box[3],
    bool periodic,
    bool checkConsistency,
    torch::Device device,
    torch::Dtype dtype
) {
    const int n = static_cast<int>(positions.size());
    const double cutoff = request->engine_cutoff("nm");
    const bool full = request->full_list();
    const double cutoff2 = cutoff * cutoff;

    vector<int32_t> samples;
    vector<double> vectors;
    samples.reserve(static_cast<size_t>(n) * 10);
    vectors.reserve(static_cast<size_t>(n) * 6);

#ifdef OPENMM_METATOMIC_USE_VESIN
    {
        vector<array<double, 3>> points(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            points[i][0] = positions[i][0];
            points[i][1] = positions[i][1];
            points[i][2] = positions[i][2];
        }
        double cell[3][3] = {
            {box[0][0], box[0][1], box[0][2]},
            {box[1][0], box[1][1], box[1][2]},
            {box[2][0], box[2][1], box[2][2]},
        };
        bool pbc[3] = {periodic, periodic, periodic};
        VesinOptions options{};
        options.cutoff = cutoff;
        options.full = full;
        options.sorted = false;
        options.algorithm = VesinAutoAlgorithm;
        options.skin = 0.0;
        options.n_threads = 0;
        options.return_shifts = true;
        options.return_distances = false;
        options.return_vectors = true;
        VesinNeighborList neighbors{};
        const char* error = nullptr;
        const int status = vesin_neighbors(
            reinterpret_cast<const double(*)[3]>(points.data()),
            static_cast<size_t>(n),
            cell,
            pbc,
            VesinDevice{VesinCPU, 0},
            options,
            &neighbors,
            &error
        );
        if (status != 0) {
            string message = error ? error : "vesin neighbor list failed";
            vesin_free(&neighbors);
            throw OpenMMException("MetatomicForce: " + message);
        }
        samples.resize(neighbors.length * 5);
        vectors.resize(neighbors.length * 3);
        for (size_t k = 0; k < neighbors.length; k++) {
            samples[5 * k + 0] = static_cast<int32_t>(neighbors.pairs[k][0]);
            samples[5 * k + 1] = static_cast<int32_t>(neighbors.pairs[k][1]);
            samples[5 * k + 2] = neighbors.shifts ? neighbors.shifts[k][0] : 0;
            samples[5 * k + 3] = neighbors.shifts ? neighbors.shifts[k][1] : 0;
            samples[5 * k + 4] = neighbors.shifts ? neighbors.shifts[k][2] : 0;
            vectors[3 * k + 0] = neighbors.vectors[k][0];
            vectors[3 * k + 1] = neighbors.vectors[k][1];
            vectors[3 * k + 2] = neighbors.vectors[k][2];
        }
        vesin_free(&neighbors);
    }
#else
    auto length = [](const Vec3& v) {
        return std::sqrt(v.dot(v));
    };
    int na = 0, nb = 0, nc = 0;
    if (periodic) {
        auto images = [&](const Vec3& v) {
            const double len = length(v);
            if (len < 1e-12)
                throw OpenMMException("MetatomicForce: periodic box vector has zero length");
            return max(1, static_cast<int>(std::ceil(cutoff / len)));
        };
        na = images(box[0]);
        nb = images(box[1]);
        nc = images(box[2]);
    }
    auto consider = [&](int i, int j, int sa, int sb, int sc) {
        if (i == j && sa == 0 && sb == 0 && sc == 0)
            return;
        if (!full && (i > j || (i == j && (sa < 0 || (sa == 0 && sb < 0) || (sa == 0 && sb == 0 && sc <= 0)))))
            return;
        const Vec3 shift = sa * box[0] + sb * box[1] + sc * box[2];
        const Vec3 delta = positions[j] - positions[i] + shift;
        if (delta.dot(delta) > cutoff2)
            return;
        samples.push_back(i);
        samples.push_back(j);
        samples.push_back(sa);
        samples.push_back(sb);
        samples.push_back(sc);
        vectors.push_back(delta[0]);
        vectors.push_back(delta[1]);
        vectors.push_back(delta[2]);
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int sa = -na; sa <= na; sa++)
                for (int sb = -nb; sb <= nb; sb++)
                    for (int sc = -nc; sc <= nc; sc++)
                        consider(i, j, sa, sb, sc);
        }
    }
#endif

    const int64_t nPairs = static_cast<int64_t>(samples.size() / 5);
    auto sampleTensor = torch::from_blob(
        samples.data(), {nPairs, 5}, torch::TensorOptions().dtype(torch::kInt32)
    ).clone().to(device);
    auto vectorTensor = torch::from_blob(
        vectors.data(), {nPairs, 3, 1}, torch::TensorOptions().dtype(torch::kFloat64)
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

        const auto pairLists = model->requested_pair_lists();
        info_.neighborListRequests = static_cast<int>(pairLists.size());
        if (!pairLists.empty()) {
            throw OpenMMException(
                "MetatomicForce: core backend does not yet implement pair lists "
                "(" + to_string(pairLists.size()) + " requested)"
            );
        }
        for (const auto& input : model->requested_inputs()) {
            info_.requestedInputs.push_back(input.name());
            throw OpenMMException(
                "MetatomicForce: this model requests extra input '" + input.name() +
                "', which is not implemented yet."
            );
        }

        bool hasEnergy = false;
        for (const auto& output : caps.outputs()) {
            if (output.name() == "energy") {
                hasEnergy = true;
                info_.energyKey = output.name();
            }
        }
        if (!hasEnergy) {
            throw OpenMMException(
                "MetatomicForce: model '" + config.modelPath + "' does not provide an energy output"
            );
        }
        checkConsistency = config.checkConsistency;
        periodic = config.periodic;
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
        vector<double> pos(3 * n);
        for (size_t i = 0; i < n; i++) {
            pos[3 * i + 0] = positions[i][0];
            pos[3 * i + 1] = positions[i][1];
            pos[3 * i + 2] = positions[i][2];
        }
        vector<double> cell(9, 0.0);
        if (periodic) {
            for (int i = 0; i < 3; i++) {
                cell[3 * i + 0] = box[i][0];
                cell[3 * i + 1] = box[i][1];
                cell[3 * i + 2] = box[i][2];
            }
        }
        try {
            vector<metatomic::System> systems;
            systems.push_back(makeSystem("nm", typesHost, pos, periodic, cell));
            auto evaluated = evaluateCore(*model, systems, checkConsistency);
            MetatomicEvaluator::Result result;
            result.energy = evaluated.energy;
            result.forces.resize(n);
            for (size_t i = 0; i < n; i++) {
                result.forces[i] = Vec3(
                    evaluated.forces[3 * i],
                    evaluated.forces[3 * i + 1],
                    evaluated.forces[3 * i + 2]
                );
            }
            return result;
        }
        catch (const exception& e) {
            throw OpenMMException(string("MetatomicForce: model evaluation failed: ") + e.what());
        }
    }

    mutable unique_ptr<metatomic::BaseModel> model;
    vector<int32_t> typesHost;
    bool checkConsistency = false;
    bool periodic = false;
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

        auto requestedInputs = model.run_method("requested_inputs", /*use_new_names=*/true).toGenericDict();
        for (const auto& entry : requestedInputs) {
            const string name(entry.key().toStringRef());
            info_.requestedInputs.push_back(name);
            throw OpenMMException(
                "MetatomicForce: this model requests extra input '" + name +
                "', which is not implemented yet. Full-system energy and conservative "
                "forces are the current milestone."
            );
        }

        auto outputs = capabilities->outputs();
        info_.energyKey = metatomic_torch::pick_output("energy", outputs, torch::nullopt);
        if (!outputs.contains(info_.energyKey)) {
            throw OpenMMException(
                "MetatomicForce: model '" + config.modelPath +
                "' does not provide an energy output"
            );
        }
        auto energyOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        energyOut->set_sample_kind(outputs.at(info_.energyKey)->sample_kind());
        energyOut->set_unit("kJ/mol");
        options = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
        options->set_length_unit("nm");
        options->outputs.insert(info_.energyKey, energyOut);

        checkConsistency = config.checkConsistency;
        periodic = config.periodic;
        pbc = torch::tensor({periodic, periodic, periodic}, torch::TensorOptions().dtype(torch::kBool)).to(device);
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
        auto posOptions = torch::TensorOptions().dtype(torch::kFloat64);
        auto posCpu = torch::from_blob(
            const_cast<Vec3*>(positions.data()), {n, 3}, posOptions
        ).clone();
        auto pos = posCpu.to(device, dtype).set_requires_grad(true);

        torch::Tensor cell;
        if (periodic) {
            double values[9] = {
                box[0][0], box[0][1], box[0][2],
                box[1][0], box[1][1], box[1][2],
                box[2][0], box[2][1], box[2][2]
            };
            cell = torch::from_blob(values, {3, 3}, posOptions).clone().to(device, dtype);
        }
        else {
            cell = torch::zeros({3, 3}, torch::TensorOptions().dtype(dtype).device(device));
        }

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(types, pos, cell, pbc);
        for (const auto& request : neighborRequests)
            addNeighborList(system, request, positions, box, periodic, checkConsistency, device, dtype);

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
        energyTensor.backward();
        auto grad = system->positions().grad();
        if (!grad.defined()) {
            throw OpenMMException(
                "MetatomicForce: model energy does not depend on positions; cannot compute forces"
            );
        }
        auto forceCpu = (-grad).to(torch::kCPU).to(torch::kFloat64).contiguous();
        MetatomicEvaluator::Result result;
        result.energy = energyTensor.item<double>();
        result.forces.resize(static_cast<size_t>(n));
        const double* ptr = forceCpu.data_ptr<double>();
        for (int64_t i = 0; i < n; i++)
            result.forces[static_cast<size_t>(i)] = Vec3(ptr[3 * i], ptr[3 * i + 1], ptr[3 * i + 2]);
        return result;
    }

    mutable metatensor_torch::Module model = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities capabilities;
    metatomic_torch::ModelEvaluationOptions options;
    vector<metatomic_torch::NeighborListOptions> neighborRequests;
    torch::Tensor types;
    vector<int> typesHost;
    torch::Tensor pbc;
    torch::Device device = torch::kCPU;
    torch::Dtype dtype = torch::kFloat32;
    bool checkConsistency = false;
    bool periodic = false;
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
