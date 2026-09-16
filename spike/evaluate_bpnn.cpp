/* -------------------------------------------------------------------------- *
 *                     OpenMM-Metatomic SOAP-BPNN spike                       *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/internal/BpnnModel.h"

#include <cmath>
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

using OpenMMMetatomic::BpnnModel;
using OpenMMMetatomic::evaluateCore;
using OpenMMMetatomic::makeSystem;

namespace {

struct DemoSystem {
    const char* name;
    std::vector<int32_t> types;
    std::vector<double> positions;
    bool periodic;
    std::vector<double> cell;
};

std::vector<double> xyz(std::initializer_list<std::initializer_list<double>> rows) {
    std::vector<double> out;
    for (const auto& row : rows)
        for (double value : row)
            out.push_back(value);
    return out;
}

std::vector<DemoSystem> demoSystems() {
    const std::vector<double> vacuum(9, 0.0);
    const std::vector<double> box = {1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5};
    return {
        DemoSystem{"water", {1, 1, 8}, xyz({
            {0.0857, 0.0586, 0.0}, {-0.0757, 0.0486, 0.01}, {0.0, 0.01, -0.005}
        }), false, vacuum},
        DemoSystem{"methane", {6, 1, 1, 1, 1}, xyz({
            {0.005, 0.0, -0.004},
            {0.07293, 0.06293, 0.06293},
            {0.06293, -0.05293, -0.06293},
            {-0.06293, 0.06293, -0.05293},
            {-0.07293, -0.06293, 0.06293},
        }), false, vacuum},
        DemoSystem{"co2", {6, 8, 8}, xyz({
            {0.0, 0.008, 0.0}, {0.126, 0.0, 0.004}, {-0.106, -0.006, 0.0}
        }), false, vacuum},
        DemoSystem{"carbon8", {6, 6, 6, 6, 6, 6, 6, 6}, xyz({
            {-0.062, -0.075, -0.067}, {-0.062, -0.075, 0.073},
            {-0.062, 0.065, -0.067}, {-0.062, 0.065, 0.073},
            {0.078, -0.075, -0.067}, {0.078, -0.075, 0.073},
            {0.078, 0.065, -0.067}, {0.078, 0.065, 0.073},
        }), false, vacuum},
        DemoSystem{"water_pbc", {1, 1, 8}, xyz({
            {0.0857, 0.0586, 0.0}, {-0.0757, 0.0486, 0.01}, {0.0, 0.01, -0.005}
        }), true, box},
    };
}

double maxAbs(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

OpenMMMetatomic::CoreEvaluation evalCore(const DemoSystem& spec) {
    auto raw = metatomic::BaseModel::to_mta_model(std::make_unique<BpnnModel>());
    metatomic::ExternalModel model(raw);
    std::vector<metatomic::System> systems;
    systems.push_back(makeSystem("nm", spec.types, spec.positions, spec.periodic, spec.cell));
    return evaluateCore(model, systems, true);
}

#ifdef OPENMM_METATOMIC_TORCH
struct TorchResult {
    double energy = 0.0;
    std::vector<double> forces;
};

TorchResult evalTorch(const DemoSystem& spec, const std::string& path) {
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

    auto device = torch::Device(torch::kCPU);
    model.to(device);
    auto types = torch::tensor(spec.types, torch::TensorOptions().dtype(torch::kInt32));
    auto pos = torch::from_blob(
        const_cast<double*>(spec.positions.data()),
        {static_cast<int64_t>(spec.types.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat64)
    ).clone().set_requires_grad(true);
    auto cell = torch::from_blob(
        const_cast<double*>(spec.cell.data()),
        {3, 3},
        torch::TensorOptions().dtype(torch::kFloat64)
    ).clone();
    auto pbc = torch::tensor(
        {spec.periodic, spec.periodic, spec.periodic},
        torch::TensorOptions().dtype(torch::kBool)
    );
    auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(types, pos, cell, pbc);
    auto output = model.forward({
        std::vector<metatomic_torch::System>{system}, options, true
    }).toGenericDict();
    auto energyMap = output.at(energyKey).toCustomClass<metatensor_torch::TensorMapHolder>();
    auto energyBlock = metatensor_torch::TensorMapHolder::block_by_id(energyMap, 0);
    auto energyTensor = energyBlock->values().sum();
    energyTensor.backward();
    auto forceCpu = (-system->positions().grad()).to(torch::kCPU).contiguous();
    TorchResult result;
    result.energy = energyTensor.item<double>();
    const double* ptr = forceCpu.data_ptr<double>();
    result.forces.assign(ptr, ptr + spec.positions.size());
    return result;
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        std::string torchPath;
        for (int i = 1; i < argc; i++)
            torchPath = argv[i];

        std::cout << std::scientific << std::setprecision(8);
        std::cout << "== SOAP-BPNN core (metatomic::execute_model) ==\n";
        std::cout << std::left << std::setw(12) << "system" << std::right
                  << std::setw(8) << "atoms" << std::setw(16) << "E_core"
                  << std::setw(14) << "max|dF_FD|" << "  pbc\n";

        double maxForceErr = 0.0;
        for (const auto& spec : demoSystems()) {
            const auto core = evalCore(spec);
            const double direct = BpnnModel::energyOf(
                spec.types, spec.positions, spec.cell, spec.periodic
            );
            if (std::abs(core.energy - direct) > 1e-12)
                throw std::runtime_error(std::string(spec.name) + " execute_model energy mismatch");

            std::vector<double> fd(spec.positions.size());
            auto displaced = spec.positions;
            for (size_t i = 0; i < spec.positions.size(); i++) {
                displaced[i] = spec.positions[i] + BpnnModel::fdStep;
                const double plus = BpnnModel::energyOf(spec.types, displaced, spec.cell, spec.periodic);
                displaced[i] = spec.positions[i] - BpnnModel::fdStep;
                const double minus = BpnnModel::energyOf(spec.types, displaced, spec.cell, spec.periodic);
                displaced[i] = spec.positions[i];
                fd[i] = -(plus - minus) / (2.0 * BpnnModel::fdStep);
            }
            const double dF = maxAbs(core.forces, fd);
            maxForceErr = std::max(maxForceErr, dF);
            if (dF > 1e-6)
                throw std::runtime_error(std::string(spec.name) + " core forces vs FD");
            std::cout << std::left << std::setw(12) << spec.name << std::right
                      << std::setw(8) << spec.types.size() << " "
                      << std::setw(16) << core.energy << " "
                      << std::setw(14) << dF
                      << "  " << (spec.periodic ? "yes" : "no") << "\n";
        }
        std::cout << "core SOAP-BPNN: energy consistent, forces match finite differences\n";

        if (!torchPath.empty()) {
#ifdef OPENMM_METATOMIC_TORCH
            std::cout << "\n== TorchScript SOAP-BPNN (" << torchPath << ") ==\n";
            std::cout << std::left << std::setw(12) << "system" << std::right
                      << std::setw(16) << "E_torch" << std::setw(12) << "ΔE"
                      << std::setw(12) << "ΔF" << "\n";
            for (const auto& spec : demoSystems()) {
                const auto core = evalCore(spec);
                const auto torch = evalTorch(spec, torchPath);
                const double dE = std::abs(torch.energy - core.energy);
                const double dF = maxAbs(torch.forces, core.forces);
                if (dE > 1e-8)
                    throw std::runtime_error(std::string(spec.name) + " torch vs core energy");
                if (dF > 1e-5)
                    throw std::runtime_error(std::string(spec.name) + " torch vs core forces");
                std::cout << std::left << std::setw(12) << spec.name << std::right
                          << std::setw(16) << torch.energy
                          << std::setw(12) << dE
                          << std::setw(12) << dF << "\n";
            }
            std::cout << "torch SOAP-BPNN matches core energy and forces\n";
#else
            std::cerr << "TorchScript backend was not compiled\n";
            return 1;
#endif
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "SOAP-BPNN spike failed: " << e.what() << "\n";
        return 1;
    }
}
