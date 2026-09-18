/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * Parity features beyond "full-system energy and autograd forces": particle
 * subsets, non-conservative forces, charge and spin inputs, and per-axis PBC.
 * The model in spike/export_features_torch.py has a closed form for all four,
 * so each check compares against arithmetic rather than another implementation.
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/MetatomicEvaluator.h"

#include "OpenMM.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

using namespace OpenMMMetatomic;
using namespace OpenMM;

namespace {

const double PAIR_ENERGY = 2.0;
const double CHARGE_COEFF = 10.0;
const double SPIN_COEFF = 100.0;
const double CUTOFF_NM = 0.3;

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

void requireClose(const std::string& label, double got, double expected, double tolerance) {
    const double diff = std::abs(got - expected);
    std::cout << "  " << label << ": got " << got << ", expected " << expected
              << " (diff " << diff << ")\n";
    require(diff <= tolerance, label + ": mismatch");
}

// Half pair list count with the same convention as the plugin's naive path.
int countPairs(const std::vector<Vec3>& positions, const Vec3 box[3],
               const bool pbc[3], double cutoff) {
    auto images = [&](const Vec3& v, bool periodic) {
        if (!periodic)
            return 0;
        return std::max(1, static_cast<int>(std::ceil(cutoff / std::sqrt(v.dot(v)))));
    };
    const int na = images(box[0], pbc[0]);
    const int nb = images(box[1], pbc[1]);
    const int nc = images(box[2], pbc[2]);
    const int n = static_cast<int>(positions.size());
    int pairs = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int sa = -na; sa <= na; sa++)
                for (int sb = -nb; sb <= nb; sb++)
                    for (int sc = -nc; sc <= nc; sc++) {
                        if (i == j && sa == 0 && sb == 0 && sc == 0)
                            continue;
                        if (i > j || (i == j && (sa < 0 || (sa == 0 && sb < 0) ||
                                                 (sa == 0 && sb == 0 && sc <= 0))))
                            continue;
                        const Vec3 shift = sa * box[0] + sb * box[1] + sc * box[2];
                        const Vec3 delta = positions[j] - positions[i] + shift;
                        if (delta.dot(delta) <= cutoff * cutoff)
                            pairs++;
                    }
    return pairs;
}

MetatomicEvaluator::Config baseConfig(const std::string& modelPath, int n) {
    MetatomicEvaluator::Config config;
    config.modelPath = modelPath;
    config.backend = "torch";
    config.atomicTypes = typesFor(n);
    return config;
}

std::vector<Vec3> randomPositions(int n, unsigned seed, double spread) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-spread, spread);
    std::vector<Vec3> positions(n);
    for (auto& v : positions)
        v = Vec3(dist(rng), dist(rng), dist(rng));
    return positions;
}

double harmonic(const std::vector<Vec3>& positions) {
    double energy = 0.0;
    for (const auto& v : positions)
        energy += 0.5 * v.dot(v);
    return energy;
}

// A subset run must see exactly what a standalone run on the same atoms sees,
// and must leave every other particle's force untouched.
void checkSubset(const std::string& modelPath) {
    std::cout << "subset\n";
    const int n = 12;
    const auto positions = randomPositions(n, 11, 0.4);
    const std::vector<int> particles = {1, 3, 4, 7, 9};
    std::vector<Vec3> subsetPositions;
    for (int index : particles)
        subsetPositions.push_back(positions[index]);

    auto run = [&](const std::vector<Vec3>& pos, const std::vector<int>& types,
                   const std::vector<int>& subsetIndices) {
        System system;
        for (size_t i = 0; i < pos.size(); i++)
            system.addParticle(1.0);
        auto* force = new MetatomicForce(modelPath);
        force->setBackend("torch");
        force->setAtomicTypes(types);
        if (!subsetIndices.empty())
            force->setParticles(subsetIndices);
        system.addForce(force);
        VerletIntegrator integrator(0.001);
        Context context(system, integrator, Platform::getPlatformByName("Reference"));
        context.setPositions(pos);
        auto state = context.getState(State::Energy | State::Forces);
        return std::make_pair(state.getPotentialEnergy(), state.getForces());
    };

    std::vector<int> subsetTypes;
    for (int index : particles)
        subsetTypes.push_back(typesFor(n)[index]);

    const auto full = run(positions, subsetTypes, particles);
    const auto standalone = run(subsetPositions, subsetTypes, {});
    requireClose("subset energy", full.first, standalone.first, 1e-8);
    for (size_t i = 0; i < particles.size(); i++) {
        const Vec3 diff = full.second[particles[i]] - standalone.second[i];
        require(std::sqrt(diff.dot(diff)) < 1e-8, "subset force mismatch on a selected atom");
    }
    double outside = 0.0;
    for (int i = 0; i < n; i++) {
        if (std::find(particles.begin(), particles.end(), i) != particles.end())
            continue;
        outside += std::sqrt(full.second[i].dot(full.second[i]));
    }
    requireClose("force outside the subset", outside, 0.0, 1e-12);
}

void checkNonConservative(const std::string& modelPath) {
    std::cout << "non-conservative forces\n";
    const int n = 8;
    const auto positions = randomPositions(n, 5, 0.4);
    const Vec3 box[3] = {Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)};

    auto config = baseConfig(modelPath, n);
    config.nonConservativeForces = true;
    MetatomicEvaluator evaluator(config);
    const auto result = evaluator.compute(positions, box);

    // The model returns (i + 1, 2, 3); only the mean-free part survives.
    double meanX = 0.0;
    for (int i = 0; i < n; i++)
        meanX += i + 1;
    meanX /= n;
    for (int i = 0; i < n; i++) {
        const Vec3 expected(i + 1 - meanX, 0.0, 0.0);
        const Vec3 diff = result.forces[i] - expected;
        require(std::sqrt(diff.dot(diff)) < 1e-9, "non-conservative force mismatch");
    }
    std::cout << "  net force removed, per-atom field matches\n";

    auto autogradConfig = baseConfig(modelPath, n);
    MetatomicEvaluator autograd(autogradConfig);
    const auto conservative = autograd.compute(positions, box);
    requireClose("energy is the same either way", result.energy, conservative.energy, 1e-9);
    double rms = 0.0;
    for (int i = 0; i < n; i++) {
        const Vec3 diff = result.forces[i] - conservative.forces[i];
        rms += diff.dot(diff);
    }
    rms = std::sqrt(rms / n);
    std::cout << "  RMS difference from autograd forces: " << rms << "\n";
    require(rms > 1e-3, "the two force paths returned the same thing; one of them is not wired up");
}

void checkChargeAndSpin(const std::string& modelPath) {
    std::cout << "charge and spin\n";
    const int n = 6;
    const auto positions = randomPositions(n, 3, 0.4);
    const Vec3 box[3] = {Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)};

    MetatomicEvaluator neutral(baseConfig(modelPath, n));
    const double base = neutral.compute(positions, box).energy;

    auto charged = baseConfig(modelPath, n);
    charged.charge = -2.0;
    charged.spinMultiplicity = 3.0;
    MetatomicEvaluator evaluator(charged);
    const double got = evaluator.compute(positions, box).energy;
    requireClose(
        "charge and spin shift the energy",
        got - base,
        CHARGE_COEFF * -2.0 + SPIN_COEFF * 2.0,
        1e-8
    );
}

void checkPerAxisPbc(const std::string& modelPath) {
    std::cout << "per-axis PBC\n";
    const int n = 6;
    // A thin box in x: only the x direction can produce extra pairs.
    const Vec3 box[3] = {Vec3(0.35, 0, 0), Vec3(0, 3.0, 0), Vec3(0, 0, 3.0)};
    std::vector<Vec3> positions(n);
    for (int i = 0; i < n; i++)
        positions[i] = Vec3(0.05 * i, 0.02 * i, 0.0);

    const std::array<std::array<bool, 3>, 4> cases = {{
        {false, false, false},
        {true, false, false},
        {false, true, false},
        {true, true, true},
    }};
    for (const auto& pbc : cases) {
        auto config = baseConfig(modelPath, n);
        config.pbc = pbc;
        MetatomicEvaluator evaluator(config);
        const double got = evaluator.compute(positions, box).energy;
        const bool flags[3] = {pbc[0], pbc[1], pbc[2]};
        const double expected =
            harmonic(positions) + PAIR_ENERGY * countPairs(positions, box, flags, CUTOFF_NM);
        const std::string label = std::string("pbc ") + (pbc[0] ? "1" : "0") +
                                  (pbc[1] ? "1" : "0") + (pbc[2] ? "1" : "0");
        requireClose(label, got, expected, 1e-8);
    }
}

void checkUncertainty(const std::string& modelPath) {
    std::cout << "energy uncertainty\n";
    const int n = 6;
    const auto positions = randomPositions(n, 17, 0.4);
    const Vec3 box[3] = {Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)};

    MetatomicEvaluator off(baseConfig(modelPath, n));
    requireClose("not requested", off.compute(positions, box).maxUncertainty, -1.0, 0.0);

    auto config = baseConfig(modelPath, n);
    config.uncertaintyThreshold = 0.03;
    MetatomicEvaluator evaluator(config);
    // The model returns 0.01 * (i + 1) eV, so the largest is 0.01 * n and the
    // atoms above the threshold (indices 3 onwards) are reported on stderr.
    requireClose(
        "largest per-atom uncertainty",
        evaluator.compute(positions, box).maxUncertainty,
        0.01 * n,
        1e-9
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: test_features <features.pt>\n";
            return 1;
        }
        const std::string modelPath = argv[1];
        if (!std::filesystem::is_regular_file(modelPath)) {
            std::cerr << "missing " << modelPath << ", skipping\n";
            return 0;
        }
        checkSubset(modelPath);
        checkNonConservative(modelPath);
        checkChargeAndSpin(modelPath);
        checkPerAxisPbc(modelPath);
        checkUncertainty(modelPath);
        std::cout << "parity features: OK\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "test_features failed: " << e.what() << "\n";
        return 1;
    }
}
