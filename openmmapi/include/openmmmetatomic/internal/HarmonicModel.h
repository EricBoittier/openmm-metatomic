#ifndef OPENMM_METATOMIC_HARMONIC_MODEL_H_
#define OPENMM_METATOMIC_HARMONIC_MODEL_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <metatomic.hpp>
#include <metatensor.hpp>

namespace OpenMMMetatomic {

/// Independent-atom harmonic well used by the M0 spike.
///
/// E = 0.5 * k * sum_i ||r_i - r0_i||^2
/// dE/dr_i = k * (r_i - r0_i)
/// F_i = -dE/dr_i
///
/// Units are OpenMM's: nm and kJ/mol. No pair list is required.
class HarmonicModel final : public metatomic::BaseModel {
public:
    HarmonicModel(double k, std::vector<double> restPositions) :
        k_(k), restPositions_(std::move(restPositions)) {
        if (restPositions_.size() % 3 != 0)
            throw std::invalid_argument("HarmonicModel rest positions must be n*3");
    }

    metatomic::ModelCapabilities capabilities() const override {
        return metatomic::ModelCapabilities::builder()
            .atomic_types({1, 6, 8})
            .interaction_range(0.0)
            .length_unit("nm")
            .supported_devices({metatomic::ModelCapabilities::Device::CPU})
            .dtype(metatomic::ModelCapabilities::DType::Float64)
            .add_output(metatomic::Quantity::builder()
                .name("energy")
                .unit("kJ/mol")
                .sample_kind(metatomic::SampleKind::System)
                .build())
            .build();
    }

    metatomic::ModelMetadata metadata() const override {
        return metatomic::ModelMetadata::builder()
            .name("openmm-metatomic-harmonic")
            .description("M0 spike: independent-atom harmonic well")
            .build();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const override {
        return {};
    }

    std::vector<metatomic::Quantity> requested_inputs() const override {
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) override {
        if (selected_atoms != nullptr)
            throw metatomic::Error("HarmonicModel does not support selected_atoms");
        if (systems.size() != 1)
            throw metatomic::Error("HarmonicModel expects exactly one system");

        const auto& system = systems[0];
        const size_t n = system.size();
        if (3 * n != restPositions_.size()) {
            throw metatomic::Error(
                "HarmonicModel expected " + std::to_string(restPositions_.size() / 3) +
                " atoms, got " + std::to_string(n)
            );
        }

        auto positions = copyPositions(system);
        double energy = 0.0;
        std::vector<double> dEdr(3 * n, 0.0);
        for (size_t i = 0; i < n; i++) {
            for (int c = 0; c < 3; c++) {
                const double delta = positions[3 * i + c] - restPositions_[3 * i + c];
                energy += 0.5 * k_ * delta * delta;
                dEdr[3 * i + c] = k_ * delta;
            }
        }

        std::vector<metatensor::TensorMap> outputs;
        outputs.reserve(requested_outputs.size());
        for (const auto& request : requested_outputs) {
            if (request.name() != "energy")
                throw metatomic::Error("HarmonicModel cannot compute '" + request.name() + "'");
            outputs.push_back(energyTensor(n, energy, dEdr, request));
        }
        return outputs;
    }

    static double analyticEnergy(double k, const std::vector<double>& r0,
                                 const std::vector<double>& positions) {
        double energy = 0.0;
        for (size_t i = 0; i < positions.size(); i++) {
            const double delta = positions[i] - r0[i];
            energy += 0.5 * k * delta * delta;
        }
        return energy;
    }

    static std::vector<double> analyticForces(double k, const std::vector<double>& r0,
                                              const std::vector<double>& positions) {
        std::vector<double> forces(positions.size());
        for (size_t i = 0; i < positions.size(); i++)
            forces[i] = -k * (positions[i] - r0[i]);
        return forces;
    }

private:
    static std::vector<double> copyPositions(const metatomic::System& system) {
        auto tensor = system.positions();
        const DLTensor& t = tensor.as_dlpack()->dl_tensor;
        if (t.ndim != 2 || t.shape[1] != 3)
            throw metatomic::Error("positions must have shape (n_atoms, 3)");
        const size_t n = static_cast<size_t>(t.shape[0]);
        std::vector<double> out(3 * n);
        const char* base = static_cast<const char*>(t.data) + t.byte_offset;
        const bool f64 = t.dtype.code == kDLFloat && t.dtype.bits == 64;
        const bool f32 = t.dtype.code == kDLFloat && t.dtype.bits == 32;
        if (!f64 && !f32)
            throw metatomic::Error("positions must be float32 or float64");
        auto at = [&](int64_t i, int64_t j) -> double {
            const int64_t si = t.strides ? t.strides[0] : 3;
            const int64_t sj = t.strides ? t.strides[1] : 1;
            if (f64) {
                return reinterpret_cast<const double*>(base)[i * si + j * sj];
            }
            return static_cast<double>(reinterpret_cast<const float*>(base)[i * si + j * sj]);
        };
        for (size_t i = 0; i < n; i++) {
            out[3 * i + 0] = at(static_cast<int64_t>(i), 0);
            out[3 * i + 1] = at(static_cast<int64_t>(i), 1);
            out[3 * i + 2] = at(static_cast<int64_t>(i), 2);
        }
        return out;
    }

    static bool wantsPositions(const metatomic::Quantity& request) {
        for (auto gradient : request.gradients()) {
            if (gradient == metatomic::Gradients::Positions)
                return true;
        }
        return false;
    }

    static metatensor::TensorMap energyTensor(size_t n, double energy,
                                              const std::vector<double>& dEdr,
                                              const metatomic::Quantity& request) {
        auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{1, 1}, std::vector<double>{energy}
        );
        auto samples = metatensor::Labels({"system"}, {{0}});
        auto properties = metatensor::Labels({"energy"}, {{0}});
        auto block = metatensor::TensorBlock(std::move(values), samples, {}, properties);

        if (wantsPositions(request)) {
            std::vector<double> grad(n * 3);
            for (size_t i = 0; i < n * 3; i++)
                grad[i] = dEdr[i];
            auto gvalues = std::make_unique<metatensor::SimpleDataArray<double>>(
                std::vector<uintptr_t>{n, 3, 1}, std::move(grad)
            );
            std::vector<int32_t> rows(n * 3);
            for (size_t i = 0; i < n; i++) {
                rows[3 * i + 0] = 0;
                rows[3 * i + 1] = 0;
                rows[3 * i + 2] = static_cast<int32_t>(i);
            }
            auto gsamples = metatensor::Labels(
                {"sample", "system", "atom"}, rows.data(), n
            );
            auto xyz = metatensor::Labels({"xyz"}, {{0}, {1}, {2}});
            auto gblock = metatensor::TensorBlock(
                std::move(gvalues), gsamples, {xyz}, properties
            );
            block.add_gradient("positions", std::move(gblock));
        }

        std::vector<metatensor::TensorBlock> blocks;
        blocks.push_back(std::move(block));
        return metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
    }

    double k_;
    std::vector<double> restPositions_;
};

inline metatomic::DLPackTensor dlpackFrom(std::vector<uintptr_t> shape, std::vector<double> data) {
    auto array = std::make_unique<metatensor::SimpleDataArray<double>>(std::move(shape), std::move(data));
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return metatomic::DLPackTensor(mts.as_dlpack(cpu, nullptr, version));
}

inline metatomic::DLPackTensor dlpackTypes(const std::vector<int32_t>& types) {
    auto array = std::make_unique<metatensor::SimpleDataArray<int32_t>>(
        std::vector<uintptr_t>{types.size()}, types
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    return metatomic::DLPackTensor(mts.as_dlpack(cpu, nullptr, version));
}

inline metatomic::DLPackTensor dlpackPbc(bool periodic) {
    const uint8_t flag = periodic ? 1 : 0;
    auto array = std::make_unique<metatensor::SimpleDataArray<uint8_t>>(
        std::vector<uintptr_t>{3}, std::vector<uint8_t>{flag, flag, flag}
    );
    auto mts = metatensor::DataArrayBase::to_mts_array(std::move(array));
    DLDevice cpu = {kDLCPU, 0};
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    auto* tensor = mts.as_dlpack(cpu, nullptr, version);
    tensor->dl_tensor.dtype.code = DLDataTypeCode::kDLBool;
    return metatomic::DLPackTensor(tensor);
}

inline metatomic::System makeSystem(const std::string& lengthUnit,
                                    const std::vector<int32_t>& types,
                                    const std::vector<double>& positions,
                                    bool periodic,
                                    const std::vector<double>& cell) {
    return metatomic::System(
        lengthUnit,
        dlpackTypes(types),
        dlpackFrom({types.size(), 3}, positions),
        dlpackFrom({3, 3}, cell),
        dlpackPbc(periodic)
    );
}

struct CoreEvaluation {
    double energy = 0.0;
    std::vector<double> forces;
};

inline CoreEvaluation evaluateCore(metatomic::BaseModel& model,
                                   std::vector<metatomic::System>& systems,
                                   bool checkConsistency) {
    auto energy = metatomic::Quantity::builder()
        .name("energy")
        .unit("kJ/mol")
        .sample_kind(metatomic::SampleKind::System)
        .add_gradient(metatomic::Gradients::Positions)
        .build();
    auto results = metatomic::execute_model(
        model, systems, std::nullopt, {energy}, checkConsistency
    );
    if (results.empty())
        throw metatomic::Error("model returned no energy output");

    auto block = results[0].block_by_id(0);
    auto values = block.values<double>();
    CoreEvaluation out;
    out.energy = values(0, 0);

    auto gradient = block.gradient("positions");
    auto gvalues = gradient.values<double>();
    const size_t n = systems[0].size();
    out.forces.resize(3 * n);
    for (size_t i = 0; i < n; i++) {
        out.forces[3 * i + 0] = -gvalues(i, 0, 0);
        out.forces[3 * i + 1] = -gvalues(i, 1, 0);
        out.forces[3 * i + 2] = -gvalues(i, 2, 0);
    }
    return out;
}

} // namespace OpenMMMetatomic

#endif
