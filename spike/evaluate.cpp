/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/internal/HarmonicModel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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

using OpenMMMetatomic::CoreEvaluation;
using OpenMMMetatomic::HarmonicModel;
using OpenMMMetatomic::evaluateCore;
using OpenMMMetatomic::makeSystem;

namespace {

constexpr double kSpring = 1.0;

struct DemoSystem {
    const char* name;
    std::vector<int32_t> types;
    std::vector<double> rest;
    std::vector<double> positions;
    bool periodic;
    std::vector<double> cell;
};

std::vector<double> xyz(std::initializer_list<std::initializer_list<double>> rows) {
    std::vector<double> out;
    out.reserve(rows.size() * 3);
    for (const auto& row : rows) {
        for (double value : row)
            out.push_back(value);
    }
    return out;
}

std::vector<DemoSystem> demoSystems() {
    const std::vector<double> vacuum(9, 0.0);
    const std::vector<double> box = {1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5};
    return {
        DemoSystem{
            "water",
            {1, 1, 8},
            xyz({{0.0757, 0.0586, 0.0}, {-0.0757, 0.0586, 0.0}, {0.0, 0.0, 0.0}}),
            xyz({{0.0857, 0.0586, 0.0}, {-0.0757, 0.0486, 0.01}, {0.0, 0.01, -0.005}}),
            false,
            vacuum,
        },
        DemoSystem{
            "methane",
            {6, 1, 1, 1, 1},
            xyz({
                {0.0, 0.0, 0.0},
                {0.06293, 0.06293, 0.06293},
                {0.06293, -0.06293, -0.06293},
                {-0.06293, 0.06293, -0.06293},
                {-0.06293, -0.06293, 0.06293},
            }),
            xyz({
                {0.005, 0.0, -0.004},
                {0.07293, 0.06293, 0.06293},
                {0.06293, -0.05293, -0.06293},
                {-0.06293, 0.06293, -0.05293},
                {-0.07293, -0.06293, 0.06293},
            }),
            false,
            vacuum,
        },
        DemoSystem{
            "co2",
            {6, 8, 8},
            xyz({{0.0, 0.0, 0.0}, {0.116, 0.0, 0.0}, {-0.116, 0.0, 0.0}}),
            xyz({{0.0, 0.008, 0.0}, {0.126, 0.0, 0.004}, {-0.106, -0.006, 0.0}}),
            false,
            vacuum,
        },
        DemoSystem{
            "carbon8",
            {6, 6, 6, 6, 6, 6, 6, 6},
            xyz({
                {-0.07, -0.07, -0.07}, {-0.07, -0.07, 0.07},
                {-0.07, 0.07, -0.07}, {-0.07, 0.07, 0.07},
                {0.07, -0.07, -0.07}, {0.07, -0.07, 0.07},
                {0.07, 0.07, -0.07}, {0.07, 0.07, 0.07},
            }),
            xyz({
                {-0.062, -0.075, -0.067}, {-0.062, -0.075, 0.073},
                {-0.062, 0.065, -0.067}, {-0.062, 0.065, 0.073},
                {0.078, -0.075, -0.067}, {0.078, -0.075, 0.073},
                {0.078, 0.065, -0.067}, {0.078, 0.065, 0.073},
            }),
            false,
            vacuum,
        },
        DemoSystem{
            "water_pbc",
            {1, 1, 8},
            xyz({{0.0757, 0.0586, 0.0}, {-0.0757, 0.0586, 0.0}, {0.0, 0.0, 0.0}}),
            xyz({{0.0857, 0.0586, 0.0}, {-0.0757, 0.0486, 0.01}, {0.0, 0.01, -0.005}}),
            true,
            box,
        },
    };
}

bool close(double a, double b, double atol = 1e-8, double rtol = 1e-6) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

void requireClose(const char* label, double got, double expected,
                  double atol = 1e-8, double rtol = 1e-6) {
    if (!close(got, expected, atol, rtol)) {
        throw std::runtime_error(
            std::string(label) + ": got " + std::to_string(got) +
            ", expected " + std::to_string(expected)
        );
    }
}

double energyAt(metatomic::BaseModel& model, const DemoSystem& spec, const std::vector<double>& pos) {
    std::vector<metatomic::System> systems;
    systems.push_back(makeSystem("nm", spec.types, pos, spec.periodic, spec.cell));
    auto request = metatomic::Quantity::builder()
        .name("energy")
        .unit("kJ/mol")
        .sample_kind(metatomic::SampleKind::System)
        .build();
    auto results = metatomic::execute_model(model, systems, std::nullopt, {request}, true);
    auto block = results[0].block_by_id(0);
    auto values = block.values<double>();
    return values(0, 0);
}

CoreEvaluation forcesAt(metatomic::BaseModel& model, const DemoSystem& spec) {
    std::vector<metatomic::System> systems;
    systems.push_back(makeSystem("nm", spec.types, spec.positions, spec.periodic, spec.cell));
    return evaluateCore(model, systems, true);
}

std::vector<double> finiteDifferenceForces(metatomic::BaseModel& model, const DemoSystem& spec) {
    const double h = 1e-5;
    auto displaced = spec.positions;
    std::vector<double> forces(spec.positions.size());
    for (size_t i = 0; i < spec.positions.size(); i++) {
        displaced[i] = spec.positions[i] + h;
        const double plus = energyAt(model, spec, displaced);
        displaced[i] = spec.positions[i] - h;
        const double minus = energyAt(model, spec, displaced);
        displaced[i] = spec.positions[i];
        forces[i] = -(plus - minus) / (2.0 * h);
    }
    return forces;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double peak = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        peak = std::max(peak, std::abs(a[i] - b[i]));
    return peak;
}

int runCore(const std::vector<DemoSystem>& systems) {
    std::cout << "== metatomic-core backend ==\n";
    std::cout << std::left << std::setw(12) << "system"
              << std::right << std::setw(8) << "atoms"
              << std::setw(16) << "E_model"
              << std::setw(16) << "E_analytic"
              << std::setw(14) << "max|dF|"
              << std::setw(14) << "max|dF_FD|"
              << "  pbc\n";
    for (const auto& spec : systems) {
        auto raw = metatomic::BaseModel::to_mta_model(
            std::make_unique<HarmonicModel>(kSpring, spec.rest)
        );
        metatomic::ExternalModel model(raw);
        const auto analyticE = HarmonicModel::analyticEnergy(kSpring, spec.rest, spec.positions);
        const auto analyticF = HarmonicModel::analyticForces(kSpring, spec.rest, spec.positions);
        const auto result = forcesAt(model, spec);
        const auto fd = finiteDifferenceForces(model, spec);
        requireClose((std::string(spec.name) + " energy").c_str(), result.energy, analyticE);
        for (size_t i = 0; i < analyticF.size(); i++) {
            requireClose((std::string(spec.name) + " force").c_str(), result.forces[i], analyticF[i]);
            requireClose((std::string(spec.name) + " FD").c_str(), result.forces[i], fd[i], 1e-6, 1e-5);
        }
        std::cout << std::left << std::setw(12) << spec.name
                  << std::right << std::setw(8) << spec.types.size()
                  << std::setw(16) << std::setprecision(8) << result.energy
                  << std::setw(16) << analyticE
                  << std::setw(14) << maxAbsDiff(result.forces, analyticF)
                  << std::setw(14) << maxAbsDiff(result.forces, fd)
                  << "  " << (spec.periodic ? "yes" : "no") << "\n";
    }
    std::cout << "core backend: all systems match analytic + finite differences\n";
    return 0;
}

int runCoreBench(const std::vector<DemoSystem>& systems, int steps) {
    std::cout << "\n== metatomic-core bench (" << steps << " evals) ==\n";
    for (const auto& spec : systems) {
        auto raw = metatomic::BaseModel::to_mta_model(
            std::make_unique<HarmonicModel>(kSpring, spec.rest)
        );
        metatomic::ExternalModel model(raw);
        forcesAt(model, spec);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; i++)
            forcesAt(model, spec);
        const auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0
        ).count();
        std::cout << "bench-core  " << spec.name
                  << "  atoms=" << spec.types.size()
                  << "  " << std::setprecision(4) << (ms / steps)
                  << " ms/eval\n";
    }
    return 0;
}

#ifdef OPENMM_METATOMIC_TORCH
torch::Device selectDevice(const std::vector<std::string>& supported, const std::string& desired) {
    torch::optional<std::string> requested = torch::nullopt;
    if (!desired.empty())
        requested = desired;
    return torch::Device(metatomic_torch::pick_device(supported, requested));
}

int runTorch(const DemoSystem& spec, const std::string& path) {
    std::cout << "\n== TorchScript backend (" << spec.name << ", " << path << ") ==\n";
    auto model = metatomic_torch::load_atomistic_model(path);
    auto capabilities = model.run_method("capabilities")
                          .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
    auto neighborRequests = model.run_method("requested_neighbor_lists").toList();
    auto requestedInputs = model.run_method("requested_inputs", true).toGenericDict();
    if (requestedInputs.size() != 0)
        throw std::runtime_error("TorchScript harmonic should not request extra inputs");
    std::cout << "dtype " << capabilities->dtype()
              << "  neighbors " << neighborRequests.size()
              << "  extra inputs " << requestedInputs.size() << "\n";

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
    auto types = torch::tensor(spec.types, torch::TensorOptions().dtype(torch::kInt32)).to(device);
    auto pos = torch::from_blob(
        const_cast<double*>(spec.positions.data()),
        {static_cast<int64_t>(spec.types.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat64)
    ).clone().to(device).set_requires_grad(true);
    auto cellTensor = torch::from_blob(
        const_cast<double*>(spec.cell.data()),
        {3, 3},
        torch::TensorOptions().dtype(torch::kFloat64)
    ).clone().to(device);
    auto pbc = torch::tensor(
        {spec.periodic, spec.periodic, spec.periodic},
        torch::TensorOptions().dtype(torch::kBool)
    ).to(device);
    auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(types, pos, cellTensor, pbc);

    auto output = model.forward({
        std::vector<metatomic_torch::System>{system}, options, true
    }).toGenericDict();
    auto energyMap = output.at(energyKey).toCustomClass<metatensor_torch::TensorMapHolder>();
    auto energyBlock = metatensor_torch::TensorMapHolder::block_by_id(energyMap, 0);
    auto energyTensor = energyBlock->values().sum();
    energyTensor.backward();
    auto grad = system->positions().grad();
    auto forceCpu = (-grad).to(torch::kCPU).to(torch::kFloat64).contiguous();
    const double energy = energyTensor.item<double>();
    const double* ptr = forceCpu.data_ptr<double>();
    const auto analyticE = HarmonicModel::analyticEnergy(kSpring, spec.rest, spec.positions);
    const auto analyticF = HarmonicModel::analyticForces(kSpring, spec.rest, spec.positions);
    requireClose("torch energy vs analytic", energy, analyticE, 1e-8, 1e-6);
    for (size_t i = 0; i < analyticF.size(); i++)
        requireClose("torch force vs analytic", ptr[i], analyticF[i], 1e-8, 1e-6);
    std::cout << "energy model " << std::setprecision(12) << energy
              << "  analytic " << analyticE << "\n";
    std::cout << "torch backend: energy and conservative forces match analytic\n";
    return 0;
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        int benchSteps = 0;
        std::string torchPath;
        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--bench") {
                benchSteps = (i + 1 < argc) ? std::stoi(argv[++i]) : 100;
                continue;
            }
            torchPath = arg;
        }

        const auto systems = demoSystems();
        runCore(systems);
        if (benchSteps > 0)
            runCoreBench(systems, benchSteps);

        if (!torchPath.empty()) {
#ifdef OPENMM_METATOMIC_TORCH
            if (std::filesystem::is_directory(torchPath)) {
                for (const auto& spec : systems) {
                    const auto model = std::filesystem::path(torchPath)
                        / (std::string(spec.name) + ".pt");
                    runTorch(spec, model.string());
                }
            } else {
                runTorch(systems.front(), torchPath);
            }
#else
            std::cerr << "TorchScript backend was not compiled (OPENMM_METATOMIC_TORCH=OFF)\n";
            return 1;
#endif
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "spike failed: " << e.what() << "\n";
        return 1;
    }
}
