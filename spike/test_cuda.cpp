/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * M3: the model on a CUDA device, through the CustomCPPForceImpl host path.
 * OpenMM hands us host positions and takes host forces back whatever platform
 * it runs on, so the only thing that can differ between devices and platforms
 * is a transfer bug. Each check therefore compares CUDA against CPU on the same
 * geometry, and skips cleanly when there is no CUDA to test.
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/MetatomicEvaluator.h"

#include "../spike/SimHelpers.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace OpenMMMetatomic;
using namespace OpenMM;

namespace {

std::vector<int> typesFor(int n) {
    static const int table[3] = {1, 6, 8};
    std::vector<int> types(n);
    for (int i = 0; i < n; i++)
        types[i] = table[i % 3];
    return types;
}

void require(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}

std::vector<Vec3> randomPositions(int n, unsigned seed, double spread) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-spread, spread);
    std::vector<Vec3> positions(n);
    for (auto& v : positions)
        v = Vec3(dist(rng), dist(rng), dist(rng));
    return positions;
}

double maxForceDiff(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const Vec3 d = a[i] - b[i];
        worst = std::max(worst, std::sqrt(d.dot(d)));
    }
    return worst;
}

/// Does the model load on CUDA at all? A model that only advertises "cpu", a
/// torch without CUDA, or a box without a device all land here.
bool cudaUsable(const std::string& modelPath, int n) {
    MetatomicEvaluator::Config config;
    config.modelPath = modelPath;
    config.backend = "torch";
    config.device = "cuda";
    config.atomicTypes = typesFor(n);
    try {
        MetatomicEvaluator evaluator(config);
        std::cout << "model on " << evaluator.info().device << "\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "no usable CUDA device (" << e.what() << "), skipping\n";
        return false;
    }
}

// The evaluator alone: same energy and forces on either device, up to the
// float64 model's own reduction-order noise.
void checkEvaluator(const std::string& modelPath) {
    std::cout << "evaluator, cpu vs cuda\n";
    const int n = 24;
    const auto positions = randomPositions(n, 13, 0.4);
    const Vec3 box[3] = {Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)};

    auto run = [&](const std::string& device, bool nonConservative) {
        MetatomicEvaluator::Config config;
        config.modelPath = modelPath;
        config.backend = "torch";
        config.device = device;
        config.atomicTypes = typesFor(n);
        config.pbc = {true, true, true};
        config.nonConservativeForces = nonConservative;
        MetatomicEvaluator evaluator(config);
        return evaluator.compute(positions, box);
    };

    for (bool nonConservative : {false, true}) {
        const auto cpu = run("cpu", nonConservative);
        const auto cuda = run("cuda", nonConservative);
        const double dE = std::abs(cuda.energy - cpu.energy);
        const double dF = maxForceDiff(cuda.forces, cpu.forces);
        std::cout << "  " << (nonConservative ? "non-conservative" : "autograd")
                  << ": |dE| " << dE << " kJ/mol, max|dF| " << dF << " kJ/mol/nm\n";
        require(dE < 1e-6, "energy differs between cpu and cuda");
        require(dF < 1e-6, "forces differ between cpu and cuda");
    }
}

/// Energy and forces from a full Context on a named platform.
std::pair<double, std::vector<Vec3>> contextRun(
    const std::string& modelPath, const std::string& platform, const std::string& device,
    const std::vector<Vec3>& positions, const std::vector<int>& particles
) {
    const int n = static_cast<int>(positions.size());
    System system;
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    auto* force = new MetatomicForce(modelPath);
    force->setBackend("torch");
    force->setDevice(device);
    if (particles.empty())
        force->setAtomicTypes(typesFor(n));
    else {
        std::vector<int> types;
        for (int index : particles)
            types.push_back(typesFor(n)[index]);
        force->setAtomicTypes(types);
        force->setParticles(particles);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, Platform::getPlatformByName(platform));
    context.setPositions(positions);
    auto state = context.getState(State::Energy | State::Forces);
    return {state.getPotentialEnergy(), state.getForces()};
}

// Every platform gets the same answer from a CUDA-resident model. The CUDA
// platform matters most for the subset path: forces come back through a device
// buffer there, so a stale value outside the subset would show up here and not
// on Reference.
void checkPlatforms(const std::string& modelPath) {
    std::cout << "platforms, cuda-resident model\n";
    const int n = 12;
    const auto positions = randomPositions(n, 17, 0.4);
    const std::vector<int> particles = {1, 3, 4, 7, 9};

    for (const auto& particleList : {std::vector<int>{}, particles}) {
        const bool subset = !particleList.empty();
        const auto reference = contextRun(modelPath, "Reference", "cpu", positions, particleList);
        for (const auto& platform : {"Reference", "CPU", "CUDA"}) {
            bool available = false;
            for (int i = 0; i < Platform::getNumPlatforms(); i++)
                available |= Platform::getPlatform(i).getName() == platform;
            if (!available) {
                std::cout << "  " << platform << ": not registered, skipping\n";
                continue;
            }
            const auto got = contextRun(modelPath, platform, "cuda", positions, particleList);
            const double dE = std::abs(got.first - reference.first);
            const double dF = maxForceDiff(got.second, reference.second);
            std::cout << "  " << platform << (subset ? ", subset" : ", full")
                      << ": |dE| " << dE << " kJ/mol, max|dF| " << dF << " kJ/mol/nm\n";
            // Reference and CPU keep forces in double; the CUDA platform's
            // single-precision force buffer is the loose one here.
            require(dE < 1e-6, std::string(platform) + ": energy differs from Reference/cpu");
            require(dF < 1e-3, std::string(platform) + ": forces differ from Reference/cpu");
            if (subset) {
                for (int i = 0; i < n; i++) {
                    if (std::find(particleList.begin(), particleList.end(), i) != particleList.end())
                        continue;
                    const double magnitude = std::sqrt(got.second[i].dot(got.second[i]));
                    require(
                        magnitude < 1e-3,
                        std::string(platform) + ": force leaked outside the subset"
                    );
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        sim::loadPlatformPlugins();
        std::cout << "platforms: " << sim::availablePlatforms() << "\n";
        if (argc < 2 || !std::filesystem::is_regular_file(argv[1])) {
            std::cout << "usage: openmm-metatomic-test-cuda features.pt\n";
            return 0;
        }
        const std::string modelPath = argv[1];
        if (!cudaUsable(modelPath, 24))
            return 0;

        checkEvaluator(modelPath);
        checkPlatforms(modelPath);
        std::cout << "CUDA host path: OK\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "test_cuda failed: " << e.what() << "\n";
        return 1;
    }
}
