/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * M2: validate pair-list plumbing (System::add_pairs / System::pairs for the
 * core backend, the vesin-backed path for both backends) through a real
 * OpenMM::Context, on both non-periodic and periodic systems. Both test
 * models add a pair-list contribution multiplied by zero to the energy, so
 * they share the plain harmonic well's analytic solution -- this isolates
 * "did the pair list round-trip correctly" from "is the physics right".
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"

#include "OpenMM.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

using namespace OpenMMMetatomic;
using namespace OpenMM;

namespace {

double analyticEnergy(const std::vector<Vec3>& positions) {
    double energy = 0.0;
    for (const auto& v : positions)
        for (int c = 0; c < 3; c++)
            energy += 0.5 * v[c] * v[c];
    return energy;
}

std::vector<int> typesFor(int n) {
    static const int table[3] = {1, 6, 8};
    std::vector<int> types(n);
    for (int i = 0; i < n; i++)
        types[i] = table[i % 3];
    return types;
}

void check(const std::string& label, const std::string& modelPath, const std::string& backend,
          const std::vector<Vec3>& positions, bool periodic) {
    const int n = static_cast<int>(positions.size());
    System system;
    for (int i = 0; i < n; i++)
        system.addParticle(1.0);
    Vec3 a(1.0, 0, 0), b(0, 1.0, 0), c(0, 0, 1.0);
    if (periodic)
        system.setDefaultPeriodicBoxVectors(a, b, c);

    auto* force = new MetatomicForce(modelPath);
    force->setBackend(backend);
    force->setAtomicTypes(typesFor(n));
    force->setUsesPeriodicBoundaryConditions(periodic);
    system.addForce(force);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    if (periodic)
        context.setPeriodicBoxVectors(a, b, c);

    const double got = context.getState(State::Energy).getPotentialEnergy();
    const double expected = analyticEnergy(positions);
    const double diff = std::abs(got - expected);
    std::cout << label << ": got " << got << "  expected " << expected << "  diff " << diff << "\n";
    if (diff > 1e-6) {
        throw std::runtime_error(label + ": pair-list energy mismatch (got " +
                                  std::to_string(got) + ", expected " + std::to_string(expected) + ")");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int n = 30;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        std::vector<Vec3> positions(n);
        for (auto& v : positions)
            v = Vec3(dist(rng), dist(rng), dist(rng));

        check("core, non-periodic", "harmonic-nl", "core", positions, false);
        check("core, periodic", "harmonic-nl", "core", positions, true);

        if (argc > 1) {
            const std::string torchPath = argv[1];
            if (std::filesystem::is_regular_file(torchPath)) {
                check("torch, non-periodic", torchPath, "torch", positions, false);
                check("torch, periodic", torchPath, "torch", positions, true);
            }
            else {
                std::cerr << "missing " << torchPath << ", skipping torch checks\n";
            }
        }

        std::cout << "pair-list plumbing: OK\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "test_pairlist failed: " << e.what() << "\n";
        return 1;
    }
}
