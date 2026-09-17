/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * Fair, repeated-trial timing of the native MetatomicForce path (through a
 * real OpenMM::Context) against the same model evaluated directly (no
 * OpenMM), for both the metatomic-core and TorchScript backends, as a
 * function of atom count.
 *
 * Fairness note: the first large allocation/heap-growth for a given atom
 * count is measurably slower than later ones at the same size (page
 * faults, malloc arena growth). Timing case A fully, then case B fully,
 * lets whichever case runs *second* inherit case A's already-warmed
 * memory -- a real effect we hit and had to fix, not a hypothetical one.
 * So every case for a given N is primed (one pass through every case,
 * round-robin, before any of them are timed) before *any* of them starts
 * its timed repeats.
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/HarmonicModel.h"

#include "OpenMM.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef OPENMM_METATOMIC_TORCH
#ifdef DIM
#undef DIM
#endif
#include <torch/script.h>
#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>
#endif

using namespace OpenMMMetatomic;
using namespace OpenMM;

namespace {

struct Stats {
    double meanMs = 0.0;
    double stddevMs = 0.0;
    double medianMs = 0.0;
    double minMs = 0.0;
};

Stats summarize(std::vector<double> samplesMs) {
    Stats out;
    if (samplesMs.empty())
        return out;
    std::sort(samplesMs.begin(), samplesMs.end());
    const double sum = std::accumulate(samplesMs.begin(), samplesMs.end(), 0.0);
    out.meanMs = sum / samplesMs.size();
    double sq = 0.0;
    for (double v : samplesMs)
        sq += (v - out.meanMs) * (v - out.meanMs);
    out.stddevMs = std::sqrt(sq / samplesMs.size());
    out.medianMs = samplesMs[samplesMs.size() / 2];
    out.minMs = samplesMs.front();
    return out;
}

// Prime every case (round-robin, `warmup` passes) before timing any of
// them, then time each case's `repeats` calls back to back. This is what
// keeps memory-warmup effects from favoring whichever case happens to run
// second -- see the fairness note above the includes.
std::vector<Stats> runFairly(const std::vector<std::function<void()>>& cases, int repeats, int warmup) {
    for (int w = 0; w < warmup; w++) {
        for (const auto& fn : cases)
            fn();
    }
    std::vector<std::vector<double>> samples(cases.size());
    for (auto& s : samples)
        s.reserve(repeats);
    for (size_t c = 0; c < cases.size(); c++) {
        for (int i = 0; i < repeats; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            cases[c]();
            const auto t1 = std::chrono::steady_clock::now();
            samples[c].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    std::vector<Stats> out;
    out.reserve(cases.size());
    for (auto& s : samples)
        out.push_back(summarize(std::move(s)));
    return out;
}

std::vector<double> randomPositions(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.3, 0.3); // nm
    std::vector<double> pos(3 * static_cast<size_t>(n));
    for (auto& v : pos)
        v = dist(rng);
    return pos;
}

std::vector<int> typesFor(int n) {
    static const int table[3] = {1, 6, 8};
    std::vector<int> types(n);
    for (int i = 0; i < n; i++)
        types[i] = table[i % 3];
    return types;
}

std::vector<Vec3> toVec3(const std::vector<double>& flat) {
    std::vector<Vec3> out(flat.size() / 3);
    for (size_t i = 0; i < out.size(); i++)
        out[i] = Vec3(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
    return out;
}

// State each case's closure needs to keep alive between priming and timing.
// Held by shared_ptr so main() can build a list of closures without caring
// about each case's concrete type.
struct DirectCoreCase {
    std::vector<int32_t> types32;
    metatomic::ExternalModel model;
    std::vector<double> cell = std::vector<double>(9, 0.0);
    std::vector<double> pos;

    DirectCoreCase(int n, std::vector<double> positions) :
        model(metatomic::BaseModel::to_mta_model(
            std::make_unique<HarmonicModel>(1.0, std::vector<double>(3 * n, 0.0))
        )),
        pos(std::move(positions))
    {
        const auto typesI = typesFor(n);
        types32.assign(typesI.begin(), typesI.end());
    }

    void operator()() {
        std::vector<metatomic::System> systems;
        systems.push_back(makeSystem("nm", types32, pos, false, cell));
        evaluateCore(model, systems, false);
    }
};

// `system` and `integrator` must outlive `context` -- Context takes them by
// reference, not by value, and does not clone them. Declaration order here
// is also initialization order, so `context`'s initializer (which runs
// last) can safely reference the two members declared above it.
struct OpenMMCase {
    System system;
    VerletIntegrator integrator;
    Context context;

    OpenMMCase(MetatomicForce* force, int n, const std::vector<double>& pos, Platform& platform) :
        integrator(0.001),
        context(buildSystem(system, force, n), integrator, platform)
    {
        context.setPositions(toVec3(pos));
    }

    void operator()() {
        context.getState(State::Forces | State::Energy);
    }

private:
    static System& buildSystem(System& system, MetatomicForce* force, int n) {
        for (int i = 0; i < n; i++)
            system.addParticle(1.0);
        system.addForce(force);
        return system;
    }
};

std::function<void()> directCoreCase(int n, const std::vector<double>& pos) {
    auto state = std::make_shared<DirectCoreCase>(n, pos);
    return [state]() { (*state)(); };
}

std::function<void()> openMMCoreCase(int n, const std::vector<double>& pos, Platform& platform) {
    auto* force = new MetatomicForce("harmonic");
    force->setBackend("core");
    force->setAtomicTypes(typesFor(n));
    auto state = std::make_shared<OpenMMCase>(force, n, pos, platform);
    return [state]() { (*state)(); };
}

#ifdef OPENMM_METATOMIC_TORCH
torch::Device selectDevice(const std::vector<std::string>& supported, const std::string& desired) {
    torch::optional<std::string> requested = torch::nullopt;
    if (!desired.empty())
        requested = desired;
    return torch::Device(metatomic_torch::pick_device(supported, requested));
}

struct DirectTorchCase {
    metatensor_torch::Module model;
    metatomic_torch::ModelEvaluationOptions options;
    torch::Tensor types;
    torch::Tensor cellTensor;
    torch::Tensor pbc;
    torch::Device device = torch::kCPU;
    std::string energyKey;
    std::vector<double> pos;
    int n;

    DirectTorchCase(const std::string& path, int n_, std::vector<double> positions) :
        model(metatomic_torch::load_atomistic_model(path)), pos(std::move(positions)), n(n_)
    {
        auto capabilities = model.run_method("capabilities")
                              .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
        auto outputs = capabilities->outputs();
        energyKey = metatomic_torch::pick_output("energy", outputs, torch::nullopt);
        auto energyOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        energyOut->set_sample_kind(outputs.at(energyKey)->sample_kind());
        energyOut->set_unit("kJ/mol");
        options = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
        options->set_length_unit("nm");
        options->outputs.insert(energyKey, energyOut);

        device = selectDevice(capabilities->supported_devices, "cpu");
        model.to(device);
        const auto typesI = typesFor(n);
        types = torch::tensor(typesI, torch::TensorOptions().dtype(torch::kInt32)).to(device);
        cellTensor = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat64)).to(device);
        pbc = torch::tensor({false, false, false}, torch::TensorOptions().dtype(torch::kBool)).to(device);
    }

    void operator()() {
        auto pos64 = torch::from_blob(
            pos.data(), {n, 3}, torch::TensorOptions().dtype(torch::kFloat64)
        ).clone().to(device).set_requires_grad(true);
        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(types, pos64, cellTensor, pbc);
        auto output = model.forward({
            std::vector<metatomic_torch::System>{system}, options, false
        }).toGenericDict();
        auto energyMap = output.at(energyKey).toCustomClass<metatensor_torch::TensorMapHolder>();
        auto energyBlock = metatensor_torch::TensorMapHolder::block_by_id(energyMap, 0);
        auto energyTensor = energyBlock->values().sum();
        energyTensor.backward();
    }
};

std::function<void()> directTorchCase(const std::string& path, int n, const std::vector<double>& pos) {
    auto state = std::make_shared<DirectTorchCase>(path, n, pos);
    return [state]() { (*state)(); };
}

std::function<void()> openMMTorchCase(const std::string& path, int n, const std::vector<double>& pos, Platform& platform) {
    auto* force = new MetatomicForce(path);
    force->setBackend("torch");
    force->setAtomicTypes(typesFor(n));
    auto state = std::make_shared<OpenMMCase>(force, n, pos, platform);
    return [state]() { (*state)(); };
}
#endif

double analyticEnergy(const std::vector<double>& pos) {
    return HarmonicModel::analyticEnergy(1.0, std::vector<double>(pos.size(), 0.0), pos);
}

void verifyEnergy(const char* label, int n, double got, double expected, double tol) {
    if (std::abs(got - expected) > tol * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(
            std::string(label) + " energy mismatch at N=" + std::to_string(n) +
            ": got " + std::to_string(got) + ", expected " + std::to_string(expected)
        );
    }
}

void verifyCore(int n, const std::vector<double>& pos, Platform& platform) {
    auto* force = new MetatomicForce("harmonic");
    force->setBackend("core");
    force->setAtomicTypes(typesFor(n));
    OpenMMCase state(force, n, pos, platform);
    const double got = state.context.getState(State::Energy).getPotentialEnergy();
    verifyEnergy("core MetatomicForce/Context", n, got, analyticEnergy(pos), 1e-6);
}

#ifdef OPENMM_METATOMIC_TORCH
void verifyTorch(const std::string& path, int n, const std::vector<double>& pos, Platform& platform) {
    auto* force = new MetatomicForce(path);
    force->setBackend("torch");
    force->setAtomicTypes(typesFor(n));
    OpenMMCase state(force, n, pos, platform);
    const double got = state.context.getState(State::Energy).getPotentialEnergy();
    verifyEnergy("torch MetatomicForce/Context", n, got, analyticEnergy(pos), 1e-4);
}
#endif

void printRow(const std::string& label, int n, const Stats& s) {
    std::cout << std::left << std::setw(26) << label
              << std::right << std::setw(8) << n
              << std::setw(14) << std::fixed << std::setprecision(5) << s.meanMs
              << std::setw(12) << s.stddevMs
              << std::setw(12) << s.medianMs
              << std::setw(12) << s.minMs
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<int> sizes = {3, 30, 300, 3000, 15000, 60000};
        std::string torchDir;
        std::string pluginsDir;
        int repeats = 50;
        int warmup = 10;
        int torchThreads = 0; // 0 = leave at libtorch's default

        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--torch-dir" && i + 1 < argc) {
                torchDir = argv[++i];
            }
            else if (arg == "--torch-threads" && i + 1 < argc) {
                torchThreads = std::stoi(argv[++i]);
            }
            else if (arg == "--plugins-dir" && i + 1 < argc) {
                pluginsDir = argv[++i];
            }
            else if (arg == "--repeats" && i + 1 < argc) {
                repeats = std::stoi(argv[++i]);
            }
            else if (arg == "--warmup" && i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
            }
            else if (arg == "--sizes" && i + 1 < argc) {
                sizes.clear();
                std::stringstream ss(argv[++i]);
                std::string tok;
                while (std::getline(ss, tok, ','))
                    sizes.push_back(std::stoi(tok));
            }
        }

        if (!pluginsDir.empty())
            Platform::loadPluginsFromDirectory(pluginsDir);

#ifdef OPENMM_METATOMIC_TORCH
        if (torchThreads > 0) {
            torch::set_num_threads(torchThreads);
            torch::set_num_interop_threads(torchThreads);
        }
        std::cout << "torch intra-op threads: " << torch::get_num_threads() << "\n";
#endif

        Platform& platform = Platform::getPlatformByName("Reference");

        std::cout << "OpenMM platform: " << platform.getName() << "\n";
        std::cout << repeats << " timed evals after " << warmup
                  << " round-robin warmup passes over all cases for this N\n\n";
        std::cout << std::left << std::setw(26) << "case"
                  << std::right << std::setw(8) << "atoms"
                  << std::setw(14) << "mean/ms"
                  << std::setw(12) << "stddev"
                  << std::setw(12) << "median"
                  << std::setw(12) << "min"
                  << "\n";

        for (int n : sizes) {
            const auto pos = randomPositions(n, /*seed=*/12345u + static_cast<unsigned>(n));

            verifyCore(n, pos, platform);
            std::vector<std::string> labels = {"core, direct execute_model", "core, MetatomicForce/Context"};
            std::vector<std::function<void()>> cases = {
                directCoreCase(n, pos),
                openMMCoreCase(n, pos, platform),
            };

#ifdef OPENMM_METATOMIC_TORCH
            std::string torchPath;
            if (!torchDir.empty()) {
                torchPath = (std::filesystem::path(torchDir) / ("harmonic-" + std::to_string(n) + ".pt")).string();
                if (std::filesystem::is_regular_file(torchPath)) {
                    verifyTorch(torchPath, n, pos, platform);
                    labels.push_back("torch, direct forward+backward");
                    cases.push_back(directTorchCase(torchPath, n, pos));
                    labels.push_back("torch, MetatomicForce/Context");
                    cases.push_back(openMMTorchCase(torchPath, n, pos, platform));
                }
                else {
                    std::cerr << "missing " << torchPath << ", skipping torch rows for N=" << n << "\n";
                }
            }
#endif

            const auto stats = runFairly(cases, repeats, warmup);
            for (size_t i = 0; i < cases.size(); i++)
                printRow(labels[i], n, stats[i]);
            std::cout << "\n";
        }
        std::cout << "all energies matched the analytic harmonic well within tolerance\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "bench_scaling failed: " << e.what() << "\n";
        return 1;
    }
}
