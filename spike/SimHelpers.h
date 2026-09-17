#ifndef OPENMM_METATOMIC_SIM_HELPERS_H_
#define OPENMM_METATOMIC_SIM_HELPERS_H_

#include "openmmmetatomic/MetatomicForce.h"
#include "OpenMM.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace OpenMMMetatomic {
namespace sim {

inline double massFor(int atomicNumber) {
    switch (atomicNumber) {
        case 1: return 1.008;
        case 6: return 12.011;
        case 7: return 14.007;
        case 8: return 15.999;
        case 9: return 18.998;
        case 16: return 32.06;
        default: return 12.0;
    }
}

constexpr double kR = 8.314462618e-3; // kJ/mol/K

inline double temperatureK(double kineticKJ, int nAtoms, bool removeCM) {
    int dof = 3 * nAtoms - (removeCM ? 3 : 0);
    if (dof <= 0)
        return 0.0;
    return 2.0 * kineticKJ / (dof * kR);
}

struct Geometry {
    std::vector<int> types;
    std::vector<OpenMM::Vec3> positions;
    OpenMM::Vec3 a = OpenMM::Vec3(0, 0, 0);
    OpenMM::Vec3 b = OpenMM::Vec3(0, 0, 0);
    OpenMM::Vec3 c = OpenMM::Vec3(0, 0, 0);
    bool periodic = false;
};

// TIP3P-like water in nm (same numbers as examples/_petmad.py).
inline Geometry waterVacuum() {
    Geometry g;
    g.types = {8, 1, 1};
    g.positions = {
        OpenMM::Vec3(0.0, 0.0, 0.0),
        OpenMM::Vec3(0.0, 0.0, 0.096),
        OpenMM::Vec3(0.0, 0.093, -0.024),
    };
    return g;
}

// Independent-atom well at arbitrary N (same cloud as spike/bench_scaling.cpp).
inline Geometry harmonicCloud(int nAtoms, unsigned seed = 1) {
    Geometry g;
    if (nAtoms <= 0)
        throw std::runtime_error("harmonicCloud needs nAtoms > 0");
    static const int table[3] = {1, 6, 8};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.3, 0.3);
    g.types.resize(static_cast<size_t>(nAtoms));
    g.positions.resize(static_cast<size_t>(nAtoms));
    for (int i = 0; i < nAtoms; i++) {
        g.types[i] = table[i % 3];
        g.positions[i] = OpenMM::Vec3(dist(rng), dist(rng), dist(rng));
    }
    return g;
}

inline Geometry waterBox(int nMolecules, double spacingNm = 0.35) {
    Geometry g;
    const int nSide = std::max(1, static_cast<int>(std::ceil(std::cbrt(static_cast<double>(nMolecules)))));
    const auto mono = waterVacuum();
    int count = 0;
    for (int ix = 0; ix < nSide && count < nMolecules; ix++) {
        for (int iy = 0; iy < nSide && count < nMolecules; iy++) {
            for (int iz = 0; iz < nSide && count < nMolecules; iz++) {
                const OpenMM::Vec3 off(ix * spacingNm, iy * spacingNm, iz * spacingNm);
                for (int a = 0; a < 3; a++) {
                    g.types.push_back(mono.types[a]);
                    g.positions.push_back(mono.positions[a] + off);
                }
                count++;
            }
        }
    }
    const double box = nSide * spacingNm;
    g.a = OpenMM::Vec3(box, 0, 0);
    g.b = OpenMM::Vec3(0, box, 0);
    g.c = OpenMM::Vec3(0, 0, box);
    g.periodic = true;
    return g;
}

inline int elementFromPdb(const std::string& line) {
    if (line.size() >= 78) {
        std::string el = line.substr(76, 2);
        el.erase(std::remove_if(el.begin(), el.end(), ::isspace), el.end());
        if (el == "H") return 1;
        if (el == "C") return 6;
        if (el == "N") return 7;
        if (el == "O") return 8;
        if (el == "F") return 9;
        if (el == "S") return 16;
    }
    return 6;
}

inline Geometry loadPdb(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    Geometry g;
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 6, "HETATM") != 0 && line.compare(0, 4, "ATOM") != 0)
            continue;
        const double x = std::stod(line.substr(30, 8)) * 0.1;
        const double y = std::stod(line.substr(38, 8)) * 0.1;
        const double z = std::stod(line.substr(46, 8)) * 0.1;
        g.types.push_back(elementFromPdb(line));
        g.positions.emplace_back(x, y, z);
    }
    if (g.types.empty())
        throw std::runtime_error("no atoms in " + path);
    return g;
}

struct ForceConfig {
    std::string modelPath;
    std::string backend = "auto";
    std::string device;
    bool checkConsistency = false;
    bool periodic = false;
};

inline OpenMM::System* makeSystem(const Geometry& geom, const ForceConfig& config) {
    auto* system = new OpenMM::System();
    for (int type : geom.types)
        system->addParticle(massFor(type));
    if (geom.periodic)
        system->setDefaultPeriodicBoxVectors(geom.a, geom.b, geom.c);
    auto* force = new MetatomicForce(config.modelPath);
    force->setBackend(config.backend);
    if (!config.device.empty())
        force->setDevice(config.device);
    force->setCheckConsistency(config.checkConsistency);
    force->setAtomicTypes(geom.types);
    force->setUsesPeriodicBoundaryConditions(config.periodic || geom.periodic);
    system->addForce(force);
    return system;
}

struct StepRecord {
    double potential = 0;
    double kinetic = 0;
    double temperature = 0;
};

inline double totalEnergy(const StepRecord& r) {
    return r.potential + r.kinetic;
}

struct MdResult {
    std::vector<StepRecord> nve;
    std::vector<StepRecord> nvt;
    double evalMs = 0;
    double nveMsPerStep = 0;
    double nvtMsPerStep = 0;
    double nveDrift = 0;
};

inline StepRecord readState(OpenMM::Context& context, int nAtoms, bool removeCM) {
    auto state = context.getState(OpenMM::State::Energy);
    StepRecord r;
    r.potential = state.getPotentialEnergy();
    r.kinetic = state.getKineticEnergy();
    r.temperature = temperatureK(r.kinetic, nAtoms, removeCM);
    return r;
}

inline OpenMM::Platform& platformByName(const std::string& name) {
    if (name.empty()) {
        try {
            return OpenMM::Platform::getPlatformByName("CPU");
        } catch (...) {
            return OpenMM::Platform::getPlatformByName("Reference");
        }
    }
    return OpenMM::Platform::getPlatformByName(name);
}

inline double medianMs(std::vector<double> samples) {
    if (samples.empty())
        return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

inline void bindGeometry(OpenMM::Context& context, const Geometry& geom) {
    context.setPositions(geom.positions);
    if (geom.periodic)
        context.setPeriodicBoxVectors(geom.a, geom.b, geom.c);
}

// Warm the same Context that will be timed: JIT/autograd/NL buffers live on
// the Context, so a fresh one always looks like a load, not a step.
inline void warmupContext(OpenMM::Context& context, OpenMM::Integrator& integrator,
                          int warmupEvals, int warmupSteps) {
    for (int i = 0; i < std::max(0, warmupEvals); i++)
        context.getState(OpenMM::State::Energy | OpenMM::State::Forces);
    if (warmupSteps > 0)
        integrator.step(warmupSteps);
}

inline double timeEvalMedian(OpenMM::Context& context, int repeats) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(std::max(1, repeats)));
    for (int i = 0; i < std::max(1, repeats); i++) {
        const auto t0 = std::chrono::steady_clock::now();
        context.getState(OpenMM::State::Energy | OpenMM::State::Forces);
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return medianMs(std::move(samples));
}

inline double timeSteps(OpenMM::Integrator& integrator, int nSteps) {
    if (nSteps <= 0)
        return 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    integrator.step(nSteps);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count()
        / static_cast<double>(nSteps);
}

// Honest per-step cost: one Context, warmup, median getState, then
// Integrator::step(N) with no getState in the timed loop.
inline MdResult runMd(
    const Geometry& geom,
    const ForceConfig& config,
    const std::string& platformName,
    int nveSteps,
    int nvtSteps,
    double dtPs,
    double temperature,
    double friction,
    int evalRepeats,
    int warmupEvals = 8,
    int warmupSteps = 10,
    int seed = 1
) {
    MdResult out;
    const int n = static_cast<int>(geom.types.size());
    OpenMM::Platform& platform = platformByName(platformName);

    {
        std::unique_ptr<OpenMM::System> system(makeSystem(geom, config));
        OpenMM::VerletIntegrator integrator(dtPs);
        OpenMM::Context context(*system, integrator, platform);
        bindGeometry(context, geom);
        context.setVelocitiesToTemperature(temperature, seed);
        warmupContext(context, integrator, warmupEvals, warmupSteps);
        out.evalMs = timeEvalMedian(context, evalRepeats);
        if (nveSteps > 0) {
            const auto e0 = readState(context, n, false);
            out.nveMsPerStep = timeSteps(integrator, nveSteps);
            const auto e1 = readState(context, n, false);
            out.nve = {e0, e1};
            out.nveDrift = totalEnergy(e1) - totalEnergy(e0);
        }
    }

    if (nvtSteps > 0) {
        std::unique_ptr<OpenMM::System> nvtSystem(makeSystem(geom, config));
        nvtSystem->addForce(new OpenMM::CMMotionRemover());
        OpenMM::LangevinMiddleIntegrator integrator(temperature, friction, dtPs);
        OpenMM::Context context(*nvtSystem, integrator, platform);
        bindGeometry(context, geom);
        context.setVelocitiesToTemperature(temperature, seed + 1);
        warmupContext(context, integrator, warmupEvals, warmupSteps);
        const auto e0 = readState(context, n, true);
        out.nvtMsPerStep = timeSteps(integrator, nvtSteps);
        const auto e1 = readState(context, n, true);
        out.nvt = {e0, e1};
    }
    return out;
}

inline void printTrace(const std::string& label, const std::vector<StepRecord>& trace, double dtPs) {
    if (trace.empty())
        return;
    const double e0 = totalEnergy(trace.front());
    const double e1 = totalEnergy(trace.back());
    double tMean = 0.0;
    for (const auto& r : trace)
        tMean += r.temperature;
    tMean /= static_cast<double>(trace.size());
    std::cout << label
              << "  samples=" << (trace.size() - 1)
              << "  dt=" << dtPs << " ps"
              << "  E0=" << e0
              << "  E1=" << e1
              << "  drift=" << (e1 - e0)
              << "  Tmean=" << tMean << " K\n";
}

} // namespace sim
} // namespace OpenMMMetatomic

#endif
