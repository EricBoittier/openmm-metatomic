/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * Fair, repeated-trial timing of the native MetatomicForce path (through a
 * real OpenMM::Context) against the same model evaluated directly (no
 * OpenMM), for both the metatomic-core and TorchScript backends, as a
 * function of atom count.
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/HarmonicModel.h"

#include "OpenMM.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
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

template <typename Fn>
Stats timeRepeats(Fn&& fn, int repeats, int warmup) {
    for (int i = 0; i < warmup; i++)
        fn();
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return summarize(samples);
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

// ---- direct metatomic-core execute_model, no OpenMM at all ----------------
Stats benchDirectCore(int n, const std::vector<double>& pos, int repeats, int warmup) {
    const auto typesI = typesFor(n);
    std::vector<int32_t> types32(typesI.begin(), typesI.end());
    auto raw = metatomic::BaseModel::to_mta_model(
        std::make_unique<HarmonicModel>(1.0, std::vector<double>(3 * n, 0.0))
    );
    metatomic::ExternalModel model(raw);
    const std::vector<double> cell(9, 0.0);
    auto evalOnce = [&]() {
        std::vector<metatomic::System> systems;
        systems.push_back(makeSystem("nm", types32, pos, false, cell));
        evaluateCore(model, systems, false);
    };
    return timeRepeats(evalOnce, repeats, warmup);
}

// ---- native MetatomicForce (backend=core) through a real Context ---------
Stats benchOpenMMCore(int n, const std::vector<double>& pos, Platform& platform, int repeats, int warmup) {
    System system;
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    auto* force = new MetatomicForce("harmonic");
    force->setBackend("core");
    force->setAtomicTypes(typesFor(n));
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(toVec3(pos));
    auto evalOnce = [&]() {
        context.getState(State::Forces | State::Energy);
    };
    return timeRepeats(evalOnce, repeats, warmup);
}

#ifdef OPENMM_METATOMIC_TORCH
torch::Device selectDevice(const std::vector<std::string>& supported, const std::string& desired) {
    torch::optional<std::string> requested = torch::nullopt;
    if (!desired.empty())
        requested = desired;
    return torch::Device(metatomic_torch::pick_device(supported, requested));
}

// ---- direct TorchScript forward + backward, no OpenMM ---------------------
Stats benchDirectTorch(const std::string& path, int n, const std::vector<double>& pos, int repeats, int warmup) {
    auto model = metatomic_torch::load_atomistic_model(path);
    auto capabilities = model.run_method("capabilities")
                          .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
    auto outputs = capabilities->outputs();
    const auto energyKey = metatomic_torch::pick_output("energy", outputs, torch::nullopt);
    auto energyOut = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    energyOut->set_sample_kind(outputs.at(energyKey)->sample_kind());
    energyOut->set_unit("kJ/mol");
    auto options = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
    options->set_length_unit("nm");
    options->outputs.insert(energyKey, energyOut);

    const auto device = selectDevice(capabilities->supported_devices, "cpu");
    model.to(device);
    auto typesI = typesFor(n);
    auto types = torch::tensor(typesI, torch::TensorOptions().dtype(torch::kInt32)).to(device);
    auto cellTensor = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat64)).to(device);
    auto pbc = torch::tensor({false, false, false}, torch::TensorOptions().dtype(torch::kBool)).to(device);

    auto evalOnce = [&]() {
        auto pos64 = torch::from_blob(
            const_cast<double*>(pos.data()), {n, 3}, torch::TensorOptions().dtype(torch::kFloat64)
        ).clone().to(device).set_requires_grad(true);
        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(types, pos64, cellTensor, pbc);
        auto output = model.forward({
            std::vector<metatomic_torch::System>{system}, options, false
        }).toGenericDict();
        auto energyMap = output.at(energyKey).toCustomClass<metatensor_torch::TensorMapHolder>();
        auto energyBlock = metatensor_torch::TensorMapHolder::block_by_id(energyMap, 0);
        auto energyTensor = energyBlock->values().sum();
        energyTensor.backward();
    };
    return timeRepeats(evalOnce, repeats, warmup);
}

// ---- native MetatomicForce (backend=torch) through a real Context --------
Stats benchOpenMMTorch(const std::string& path, int n, const std::vector<double>& pos, Platform& platform, int repeats, int warmup) {
    System system;
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    auto* force = new MetatomicForce(path);
    force->setBackend("torch");
    force->setAtomicTypes(typesFor(n));
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(toVec3(pos));
    auto evalOnce = [&]() {
        context.getState(State::Forces | State::Energy);
    };
    return timeRepeats(evalOnce, repeats, warmup);
}
#endif

double analyticEnergy(const std::vector<double>& pos) {
    return HarmonicModel::analyticEnergy(1.0, std::vector<double>(pos.size(), 0.0), pos);
}

void verifyCore(int n, const std::vector<double>& pos, Platform& platform) {
    const double expected = analyticEnergy(pos);
    System system;
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    auto* force = new MetatomicForce("harmonic");
    force->setBackend("core");
    force->setAtomicTypes(typesFor(n));
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(toVec3(pos));
    const double got = context.getState(State::Energy).getPotentialEnergy();
    if (std::abs(got - expected) > 1e-6 * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(
            "core MetatomicForce/Context energy mismatch at N=" + std::to_string(n) +
            ": got " + std::to_string(got) + ", expected " + std::to_string(expected)
        );
    }
}

#ifdef OPENMM_METATOMIC_TORCH
void verifyTorch(const std::string& path, int n, const std::vector<double>& pos, Platform& platform) {
    const double expected = analyticEnergy(pos);
    System system;
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    auto* force = new MetatomicForce(path);
    force->setBackend("torch");
    force->setAtomicTypes(typesFor(n));
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(toVec3(pos));
    const double got = context.getState(State::Energy).getPotentialEnergy();
    if (std::abs(got - expected) > 1e-4 * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(
            "torch MetatomicForce/Context energy mismatch at N=" + std::to_string(n) +
            ": got " + std::to_string(got) + ", expected " + std::to_string(expected)
        );
    }
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
        std::cout << repeats << " timed evals after " << warmup << " warmup calls per row\n\n";
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
            printRow("core, direct execute_model", n, benchDirectCore(n, pos, repeats, warmup));
            printRow("core, MetatomicForce/Context", n, benchOpenMMCore(n, pos, platform, repeats, warmup));

#ifdef OPENMM_METATOMIC_TORCH
            if (!torchDir.empty()) {
                const auto path = (std::filesystem::path(torchDir) / ("harmonic-" + std::to_string(n) + ".pt")).string();
                if (std::filesystem::is_regular_file(path)) {
                    verifyTorch(path, n, pos, platform);
                    printRow("torch, direct forward+backward", n, benchDirectTorch(path, n, pos, repeats, warmup));
                    printRow("torch, MetatomicForce/Context", n, benchOpenMMTorch(path, n, pos, platform, repeats, warmup));
                }
                else {
                    std::cerr << "missing " << path << ", skipping torch rows for N=" << n << "\n";
                }
            }
#endif
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
