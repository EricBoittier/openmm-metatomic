/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/internal/HarmonicModel.h"

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

using OpenMMMetatomic::CoreEvaluation;
using OpenMMMetatomic::HarmonicModel;
using OpenMMMetatomic::evaluateCore;
using OpenMMMetatomic::makeSystem;

namespace {

constexpr double kSpring = 1.0;
const std::vector<int32_t> atomTypes = {1, 1, 8};
const std::vector<double> rest = {
    0.0, 0.0, 0.0,
    0.1, 0.0, 0.0,
    0.0, 0.1, 0.05,
};
const std::vector<double> positions = {
    0.01, 0.02, -0.01,
    0.12, -0.03, 0.04,
    -0.02, 0.11, 0.02,
};
const std::vector<double> cell(9, 0.0);

bool close(double a, double b, double atol = 1e-8, double rtol = 1e-6) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

void requireClose(const char* label, double got, double expected) {
    if (!close(got, expected)) {
        throw std::runtime_error(
            std::string(label) + ": got " + std::to_string(got) +
            ", expected " + std::to_string(expected)
        );
    }
}

std::string deviceName(metatomic::ModelCapabilities::Device device) {
    switch (device) {
        case metatomic::ModelCapabilities::Device::CPU: return "cpu";
        case metatomic::ModelCapabilities::Device::CUDA: return "cuda";
        case metatomic::ModelCapabilities::Device::ROCM: return "rocm";
        case metatomic::ModelCapabilities::Device::Metal: return "metal";
    }
    return "unknown";
}

double energyAt(metatomic::BaseModel& model, const std::vector<double>& pos) {
    std::vector<metatomic::System> systems;
    systems.push_back(makeSystem("nm", atomTypes, pos, false, cell));
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

CoreEvaluation forcesAt(metatomic::BaseModel& model, const std::vector<double>& pos) {
    std::vector<metatomic::System> systems;
    systems.push_back(makeSystem("nm", atomTypes, pos, false, cell));
    return evaluateCore(model, systems, true);
}

std::vector<double> finiteDifferenceForces(metatomic::BaseModel& model,
                                           const std::vector<double>& pos) {
    const double h = 1e-5;
    auto displaced = pos;
    std::vector<double> forces(pos.size());
    for (size_t i = 0; i < pos.size(); i++) {
        displaced[i] = pos[i] + h;
        const double plus = energyAt(model, displaced);
        displaced[i] = pos[i] - h;
        const double minus = energyAt(model, displaced);
        displaced[i] = pos[i];
        forces[i] = -(plus - minus) / (2.0 * h);
    }
    return forces;
}

void printForces(const char* title, const std::vector<double>& forces) {
    std::cout << title << "\n";
    for (size_t i = 0; i < forces.size() / 3; i++) {
        std::cout << "  atom " << i << ": "
                  << std::setprecision(12)
                  << forces[3 * i] << " "
                  << forces[3 * i + 1] << " "
                  << forces[3 * i + 2] << "\n";
    }
}

int runCore() {
    std::cout << "== metatomic-core backend (in-process HarmonicModel) ==\n";
    auto raw = metatomic::BaseModel::to_mta_model(
        std::make_unique<HarmonicModel>(kSpring, rest)
    );
    metatomic::ExternalModel model(raw);

    const auto caps = model.capabilities();
    const auto meta = model.metadata();
    nlohmann::json metaJson = meta;
    std::cout << metatomic::format_metadata(metaJson.dump()) << "\n";
    std::cout << "dtype: " << (caps.dtype() == metatomic::ModelCapabilities::DType::Float64
                                   ? "float64" : "float32") << "\n";
    std::cout << "length unit: " << caps.length_unit() << "\n";
    std::cout << "devices:";
    for (auto device : caps.supported_devices())
        std::cout << " " << deviceName(device);
    std::cout << "\natomic types:";
    for (auto type : caps.atomic_types())
        std::cout << " " << type;
    std::cout << "\npair lists: " << model.requested_pair_lists().size() << "\n";
    std::cout << "requested inputs: " << model.requested_inputs().size() << "\n";
    for (const auto& output : caps.outputs())
        std::cout << "output: " << output.name() << " [" << output.unit() << "]\n";

    const auto analyticE = HarmonicModel::analyticEnergy(kSpring, rest, positions);
    const auto analyticF = HarmonicModel::analyticForces(kSpring, rest, positions);
    const auto result = forcesAt(model, positions);
    const auto fd = finiteDifferenceForces(model, positions);

    std::cout << std::setprecision(12);
    std::cout << "energy model    " << result.energy << "\n";
    std::cout << "energy analytic " << analyticE << "\n";
    printForces("forces model", result.forces);
    printForces("forces analytic", analyticF);
    printForces("forces FD", fd);

    requireClose("energy vs analytic", result.energy, analyticE);
    for (size_t i = 0; i < analyticF.size(); i++) {
        requireClose("force vs analytic", result.forces[i], analyticF[i]);
        requireClose("force vs FD", result.forces[i], fd[i], 1e-6, 1e-5);
    }
    std::cout << "core backend: energy and conservative forces match analytic + FD\n";
    return 0;
}

#ifdef OPENMM_METATOMIC_TORCH
torch::Device selectDevice(const std::vector<std::string>& supported, const std::string& desired) {
    torch::optional<std::string> requested = torch::nullopt;
    if (!desired.empty())
        requested = desired;
    return torch::Device(metatomic_torch::pick_device(supported, requested));
}

int runTorch(const std::string& path) {
    std::cout << "\n== TorchScript backend (load_atomistic_model) ==\n";
    auto model = metatomic_torch::load_atomistic_model(path);
    auto capabilities = model.run_method("capabilities")
                          .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
    std::cout << "dtype: " << capabilities->dtype() << "\n";
    std::cout << "length unit: " << capabilities->length_unit() << "\n";
    std::cout << "devices:";
    for (const auto& device : capabilities->supported_devices)
        std::cout << " " << device;
    std::cout << "\natomic types:";
    for (auto type : capabilities->atomic_types)
        std::cout << " " << type;
    std::cout << "\n";

    auto neighborRequests = model.run_method("requested_neighbor_lists").toList();
    std::cout << "neighbor lists: " << neighborRequests.size() << "\n";
    auto requestedInputs = model.run_method("requested_inputs", true).toGenericDict();
    std::cout << "requested inputs: " << requestedInputs.size() << "\n";
    if (requestedInputs.size() != 0)
        throw std::runtime_error("TorchScript harmonic should not request extra inputs");

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
    auto types = torch::tensor(atomTypes, torch::TensorOptions().dtype(torch::kInt32)).to(device);
    auto pos = torch::from_blob(
        const_cast<double*>(positions.data()),
        {static_cast<int64_t>(atomTypes.size()), 3},
        torch::TensorOptions().dtype(torch::kFloat64)
    ).clone().to(device).set_requires_grad(true);
    auto cellTensor = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    auto pbc = torch::tensor({false, false, false}, torch::TensorOptions().dtype(torch::kBool)).to(device);
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

    const auto analyticE = HarmonicModel::analyticEnergy(kSpring, rest, positions);
    const auto analyticF = HarmonicModel::analyticForces(kSpring, rest, positions);
    std::cout << std::setprecision(12);
    std::cout << "energy model    " << energy << "\n";
    std::cout << "energy analytic " << analyticE << "\n";
    requireClose("torch energy vs analytic", energy, analyticE, 1e-8, 1e-6);
    for (size_t i = 0; i < analyticF.size(); i++)
        requireClose("torch force vs analytic", ptr[i], analyticF[i], 1e-8, 1e-6);
    std::cout << "torch backend: energy and conservative forces match analytic\n";
    return 0;
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        runCore();
        if (argc > 1) {
#ifdef OPENMM_METATOMIC_TORCH
            runTorch(argv[1]);
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
