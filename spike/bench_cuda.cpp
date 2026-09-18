/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * M3: what a CUDA-resident model costs through the CustomCPPForceImpl host
 * path. Three layers per case, so the overheads separate:
 *
 *   evaluator      MetatomicEvaluator::compute() on host positions: neighbor
 *                  list, host->device positions, forward, backward,
 *                  device->host forces.
 *   Context        the same through OpenMM. On the CUDA platform OpenMM also
 *                  copies positions device->host and forces host->device
 *                  around our call, which is the host-path overhead M3 is
 *                  about; on Reference and CPU it hands over host buffers.
 *   round trip     a bare N x 3 tensor host->device->host, the floor under our
 *                  two copies.
 *
 * Timing recipe as everywhere else here: round-robin warmup, then medians.
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/MetatomicEvaluator.h"

#include "../spike/SimHelpers.h"

#ifdef OPENMM_METATOMIC_TORCH
#include <torch/torch.h>
#endif

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace OpenMMMetatomic;
using namespace OpenMMMetatomic::sim;
using namespace OpenMM;

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Row {
    std::string label;
    int atoms = 0;
    double ms = 0.0;
    double energy = 0.0;
    /// Per-step cost with no state download, or 0 for rows that are not MD.
    double stepMs = 0.0;
};

std::vector<std::string> split(const std::string& value) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
        if (!item.empty())
            parts.push_back(item);
    return parts;
}

std::vector<int> splitInts(const std::string& value) {
    std::vector<int> parts;
    for (const auto& item : split(value))
        parts.push_back(std::stoi(item));
    return parts;
}

double evaluatorMedian(
    const std::string& modelPath, const std::string& device, const Geometry& geom,
    int warmup, int repeats, double& energy
) {
    MetatomicEvaluator::Config config;
    config.modelPath = modelPath;
    config.backend = "torch";
    config.device = device;
    config.atomicTypes = geom.types;
    config.pbc = {geom.periodic, geom.periodic, geom.periodic};
    MetatomicEvaluator evaluator(config);
    const Vec3 box[3] = {geom.a, geom.b, geom.c};

    for (int i = 0; i < warmup; i++)
        evaluator.compute(geom.positions, box);
    std::vector<double> samples;
    for (int i = 0; i < repeats; i++) {
        const auto start = Clock::now();
        const auto result = evaluator.compute(geom.positions, box);
        samples.push_back(msSince(start));
        energy = result.energy;
    }
    return medianMs(samples);
}

double contextMedian(
    const std::string& modelPath, const std::string& device, const std::string& platform,
    const Geometry& geom, int warmup, int repeats, int mdSteps,
    double& energy, double& stepMs
) {
    ForceConfig config;
    config.modelPath = modelPath;
    config.backend = "torch";
    config.device = device;
    config.periodic = geom.periodic;
    std::unique_ptr<System> system(makeSystem(geom, config));
    VerletIntegrator integrator(0.0005);
    Context context(*system, integrator, platformByName(platform));
    bindGeometry(context, geom);
    for (int i = 0; i < warmup; i++)
        context.getState(State::Energy | State::Forces);
    std::vector<double> samples;
    for (int i = 0; i < repeats; i++) {
        const auto start = Clock::now();
        auto state = context.getState(State::Energy | State::Forces);
        samples.push_back(msSince(start));
        energy = state.getPotentialEnergy();
    }
    if (mdSteps > 0) {
        // No per-step query: this is what a production run actually pays, and
        // it excludes the force download getState() asks for.
        integrator.step(std::max(1, warmup));
        stepMs = timeSteps(integrator, mdSteps);
    }
    return medianMs(samples);
}

#ifdef OPENMM_METATOMIC_TORCH
/// Floor under our own two copies: N x 3 out and back, nothing else.
double roundTripMedian(int atoms, int warmup, int repeats) {
    if (!torch::cuda::is_available())
        return 0.0;
    auto host = torch::zeros({atoms, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto device = torch::Device(torch::kCUDA);
    auto once = [&]() {
        auto onDevice = host.to(device);
        auto back = onDevice.to(torch::kCPU);
        torch::cuda::synchronize();
        return back.size(0);
    };
    for (int i = 0; i < warmup; i++)
        once();
    std::vector<double> samples;
    for (int i = 0; i < repeats; i++) {
        const auto start = Clock::now();
        once();
        samples.push_back(msSince(start));
    }
    return medianMs(samples);
}
#endif

void printRows(const std::vector<Row>& rows) {
    std::cout << std::left << std::setw(38) << "case"
              << std::right << std::setw(8) << "atoms"
              << std::setw(12) << "median/ms"
              << std::setw(12) << "ms/step"
              << std::setw(18) << "energy/kJ/mol" << "\n";
    for (const auto& row : rows) {
        std::cout << std::left << std::setw(38) << row.label
                  << std::right << std::setw(8) << row.atoms
                  << std::setw(12) << std::fixed << std::setprecision(3) << row.ms;
        if (row.stepMs > 0.0)
            std::cout << std::setw(12) << row.stepMs;
        else
            std::cout << std::setw(12) << "-";
        std::cout << std::setw(18) << std::setprecision(4) << row.energy << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string modelPath;
        std::vector<int> molecules = {32, 96, 256};
        std::vector<std::string> devices = {"cpu", "cuda"};
        int warmup = 5;
        int repeats = 11;
        int mdSteps = 10;

        auto need = [&](int i) {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + std::string(argv[i]));
            return std::string(argv[i + 1]);
        };
        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--model") { modelPath = need(i); i++; }
            else if (arg == "--n-mol") { molecules = splitInts(need(i)); i++; }
            else if (arg == "--devices") { devices = split(need(i)); i++; }
            else if (arg == "--warmup") { warmup = std::stoi(need(i)); i++; }
            else if (arg == "--repeats") { repeats = std::stoi(need(i)); i++; }
            else if (arg == "--md") { mdSteps = std::stoi(need(i)); i++; }
            else if (arg == "--help" || arg == "-h") {
                std::cout <<
                    "openmm-metatomic-bench-cuda --model model.pt\n"
                    "  --n-mol 32,96,256  --devices cpu,cuda\n"
                    "  --warmup 5 --repeats 11 --md 10  (--md 0 skips MD)\n";
                return 0;
            }
            else if (modelPath.empty()) { modelPath = arg; }
            else throw std::runtime_error("unknown argument " + arg);
        }
        if (modelPath.empty() || !std::filesystem::is_regular_file(modelPath)) {
            std::cout << "usage: openmm-metatomic-bench-cuda model.pt\n";
            return 0;
        }

        loadPlatformPlugins();
        std::vector<std::string> platforms;
        for (int i = 0; i < Platform::getNumPlatforms(); i++)
            platforms.push_back(Platform::getPlatform(i).getName());
        std::cout << "model " << modelPath << "  platforms " << availablePlatforms() << "\n";

        std::vector<Row> rows;
        for (int nMol : molecules) {
            const auto geom = waterBox(nMol);
            const int atoms = static_cast<int>(geom.types.size());
            for (const auto& device : devices) {
                double energy = 0.0;
                double ms = 0.0;
                try {
                    ms = evaluatorMedian(modelPath, device, geom, warmup, repeats, energy);
                }
                catch (const std::exception& e) {
                    std::cout << "  evaluator/" << device << " unavailable: " << e.what() << "\n";
                    continue;
                }
                rows.push_back({"evaluator/" + device, atoms, ms, energy, 0.0});
                for (const auto& platform : platforms) {
                    const auto label = "Context/" + platform + "/" + device;
                    double stepMs = 0.0;
                    try {
                        ms = contextMedian(
                            modelPath, device, platform, geom, warmup, repeats, mdSteps,
                            energy, stepMs
                        );
                    }
                    catch (const std::exception& e) {
                        std::cout << "  " << label << " unavailable: " << e.what() << "\n";
                        continue;
                    }
                    rows.push_back({label, atoms, ms, energy, stepMs});
                }
            }
#ifdef OPENMM_METATOMIC_TORCH
            const double trip = roundTripMedian(atoms, warmup, repeats);
            if (trip > 0.0)
                rows.push_back({"round trip host->device->host", atoms, trip, 0.0, 0.0});
#endif
        }
        printRows(rows);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "bench_cuda failed: " << e.what() << "\n";
        return 1;
    }
}
