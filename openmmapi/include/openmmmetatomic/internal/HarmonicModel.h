#ifndef OPENMM_METATOMIC_HARMONIC_MODEL_H_
#define OPENMM_METATOMIC_HARMONIC_MODEL_H_

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <metatomic.hpp>
#include <metatensor.hpp>

namespace OpenMMMetatomic {

namespace detail {

inline std::vector<double> copyPositions(const metatomic::System& system) {
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

inline bool wantsPositions(const metatomic::Quantity& request) {
    for (auto gradient : request.gradients()) {
        if (gradient == metatomic::Gradients::Positions)
            return true;
    }
    return false;
}

inline metatensor::TensorMap energyTensor(size_t n, double energy,
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

} // namespace detail

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
                .add_gradient(metatomic::Gradients::Positions)
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

        auto positions = detail::copyPositions(system);
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
            outputs.push_back(detail::energyTensor(n, energy, dEdr, request));
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
    double k_;
    std::vector<double> restPositions_;
};

/// Same independent-atom harmonic well as HarmonicModel, but requests a pair
/// list and reads it (summed into the energy with a zero coefficient) before
/// computing. Exists purely to exercise the core backend's pair-list plumbing
/// (System::add_pairs / System::pairs) end to end: since the pair-list
/// contribution to the energy is always exactly zero, this model has the
/// same analytic solution as HarmonicModel, so the M0 spike's existing
/// analytic + finite-difference checks apply unchanged.
class NeighborHarmonicModel final : public metatomic::BaseModel {
public:
    NeighborHarmonicModel(double k, std::vector<double> restPositions, double cutoff) :
        k_(k), restPositions_(std::move(restPositions)),
        pairOptions_(metatomic::PairListOptions::builder().cutoff(cutoff).full_list(false).build())
    {
        if (restPositions_.size() % 3 != 0)
            throw std::invalid_argument("NeighborHarmonicModel rest positions must be n*3");
    }

    metatomic::ModelCapabilities capabilities() const override {
        return metatomic::ModelCapabilities::builder()
            .atomic_types({1, 6, 8})
            .interaction_range(pairOptions_.cutoff())
            .length_unit("nm")
            .supported_devices({metatomic::ModelCapabilities::Device::CPU})
            .dtype(metatomic::ModelCapabilities::DType::Float64)
            .add_output(metatomic::Quantity::builder()
                .name("energy")
                .unit("kJ/mol")
                .sample_kind(metatomic::SampleKind::System)
                .add_gradient(metatomic::Gradients::Positions)
                .build())
            .build();
    }

    metatomic::ModelMetadata metadata() const override {
        return metatomic::ModelMetadata::builder()
            .name("openmm-metatomic-neighbor-harmonic")
            .description("Core backend pair-list plumbing test: harmonic well that also reads a neighbor list")
            .build();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const override {
        return {pairOptions_};
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
            throw metatomic::Error("NeighborHarmonicModel does not support selected_atoms");
        if (systems.size() != 1)
            throw metatomic::Error("NeighborHarmonicModel expects exactly one system");

        const auto& system = systems[0];
        const size_t n = system.size();
        if (3 * n != restPositions_.size()) {
            throw metatomic::Error(
                "NeighborHarmonicModel expected " + std::to_string(restPositions_.size() / 3) +
                " atoms, got " + std::to_string(n)
            );
        }

        // Touch the pair list (proves add_pairs()/pairs() round-trip through
        // the engine) without perturbing the energy: multiplied by zero.
        auto pairs = system.pairs(pairOptions_);
        const auto nPairs = pairs.samples().count();
        auto pairValues = pairs.values<double>();
        double pairTouch = 0.0;
        for (size_t k = 0; k < nPairs; k++)
            for (int c = 0; c < 3; c++)
                pairTouch += pairValues(k, c, 0);

        auto positions = detail::copyPositions(system);
        double energy = pairTouch * 0.0;
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
                throw metatomic::Error("NeighborHarmonicModel cannot compute '" + request.name() + "'");
            outputs.push_back(detail::energyTensor(n, energy, dEdr, request));
        }
        return outputs;
    }

private:
    double k_;
    std::vector<double> restPositions_;
    metatomic::PairListOptions pairOptions_;
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

inline metatomic::DLPackTensor dlpackPbc(const std::array<bool, 3>& pbc) {
    auto array = std::make_unique<metatensor::SimpleDataArray<uint8_t>>(
        std::vector<uintptr_t>{3},
        std::vector<uint8_t>{
            static_cast<uint8_t>(pbc[0] ? 1 : 0),
            static_cast<uint8_t>(pbc[1] ? 1 : 0),
            static_cast<uint8_t>(pbc[2] ? 1 : 0),
        }
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
                                    const std::array<bool, 3>& pbc,
                                    const std::vector<double>& cell) {
    return metatomic::System(
        lengthUnit,
        dlpackTypes(types),
        dlpackFrom({types.size(), 3}, positions),
        dlpackFrom({3, 3}, cell),
        dlpackPbc(pbc)
    );
}

inline metatomic::System makeSystem(const std::string& lengthUnit,
                                    const std::vector<int32_t>& types,
                                    const std::vector<double>& positions,
                                    bool periodic,
                                    const std::vector<double>& cell) {
    return makeSystem(
        lengthUnit, types, positions,
        std::array<bool, 3>{periodic, periodic, periodic}, cell
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
