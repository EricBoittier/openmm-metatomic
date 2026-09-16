#ifndef OPENMM_METATOMIC_BPNN_MODEL_H_
#define OPENMM_METATOMIC_BPNN_MODEL_H_

#include "openmmmetatomic/internal/BpnnWeights.h"
#include "openmmmetatomic/internal/HarmonicModel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace OpenMMMetatomic {

/// SOAP-BPNN matching metatrain ``soap_bpnn`` (legacy per-species MLP).
///
/// SOAP: shifted-cosine cutoff, Bernstein radial basis, real spherical
/// harmonics through l = 1, power-spectrum contraction over m. BPNN: SiLU MLP
/// with no bias, one hidden layer, last linear onto atomic energy.
class BpnnModel final : public metatomic::BaseModel {
public:
    static constexpr double cutoff = 0.5;
    static constexpr double width = 0.05;
    static constexpr int nRadial = 2;
    static constexpr int nSpecies = 3;
    static constexpr int soapL0 = nSpecies * nRadial;
    static constexpr double fdStep = 1e-6;

    metatomic::ModelCapabilities capabilities() const override {
        return metatomic::ModelCapabilities::builder()
            .atomic_types({1, 6, 8})
            .interaction_range(cutoff)
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
            .name("openmm-metatomic-soap-bpnn")
            .description("SOAP-BPNN twin of metatrain soap_bpnn")
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
            throw metatomic::Error("BpnnModel does not support selected_atoms");
        if (systems.size() != 1)
            throw metatomic::Error("BpnnModel expects exactly one system");

        const auto& system = systems[0];
        auto types = copyTypes(system);
        auto positions = copyPositions(system);
        auto cell = copyCell(system);
        const bool periodic = copyPeriodic(system);
        const size_t n = types.size();
        const double energy = energyOf(types, positions, cell, periodic);
        std::vector<double> dEdr(3 * n, 0.0);
        const bool needGrad = wantsPositions(requested_outputs);
        if (needGrad) {
            auto displaced = positions;
            for (size_t i = 0; i < 3 * n; i++) {
                displaced[i] = positions[i] + fdStep;
                const double plus = energyOf(types, displaced, cell, periodic);
                displaced[i] = positions[i] - fdStep;
                const double minus = energyOf(types, displaced, cell, periodic);
                displaced[i] = positions[i];
                dEdr[i] = (plus - minus) / (2.0 * fdStep);
            }
        }

        std::vector<metatensor::TensorMap> outputs;
        for (const auto& request : requested_outputs) {
            if (request.name() != "energy")
                throw metatomic::Error("BpnnModel cannot compute '" + request.name() + "'");
            outputs.push_back(packEnergy(n, energy, dEdr, request, needGrad));
        }
        return outputs;
    }

    static double energyOf(const std::vector<int32_t>& types,
                           const std::vector<double>& positions,
                           const std::vector<double>& cell,
                           bool periodic) {
        const size_t n = types.size();
        std::vector<double> features(n * static_cast<size_t>(kBpnnSoapSize), 0.0);
        for (size_t i = 0; i < n; i++)
            soapAtom(i, types, positions, cell, periodic, features.data() + i * kBpnnSoapSize);
        double energy = 0.0;
        for (size_t i = 0; i < n; i++) {
            const int s = speciesIndex(types[i]);
            double hidden[kBpnnHidden];
            for (int h = 0; h < kBpnnHidden; h++) {
                double z = 0.0;
                for (int k = 0; k < kBpnnSoapSize; k++)
                    z += kBpnnW1[s][h][k] * features[i * kBpnnSoapSize + k];
                hidden[h] = silu(z);
            }
            for (int h = 0; h < kBpnnHidden; h++)
                energy += kBpnnW2[s][h] * hidden[h];
        }
        return energy;
    }

private:
    static int speciesIndex(int32_t z) {
        if (z == 1) return 0;
        if (z == 6) return 1;
        if (z == 8) return 2;
        throw metatomic::Error("BpnnModel only supports H, C, O");
    }

    static double silu(double x) {
        if (x > 60.0) return x;
        if (x < -60.0) return 0.0;
        return x / (1.0 + std::exp(-x));
    }

    static double shiftedCosine(double r) {
        if (r >= cutoff) return 0.0;
        const double inner = cutoff - width;
        if (r >= inner)
            return 0.5 * (1.0 + std::cos(std::acos(-1.0) * (r - inner) / width));
        return 1.0;
    }

    static void minImage(double dr[3], const std::vector<double>& cell, bool periodic) {
        if (!periodic) return;
        for (int c = 0; c < 3; c++) {
            const double length = cell[3 * c + c];
            if (length <= 0.0) continue;
            dr[c] -= length * std::nearbyint(dr[c] / length);
        }
    }

    static void soapAtom(size_t i,
                         const std::vector<int32_t>& types,
                         const std::vector<double>& positions,
                         const std::vector<double>& cell,
                         bool periodic,
                         double* out) {
        const size_t n = types.size();
        double c0[3][2] = {};
        double c1[3][2][3] = {};
        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            double dr[3] = {
                positions[3 * j + 0] - positions[3 * i + 0],
                positions[3 * j + 1] - positions[3 * i + 1],
                positions[3 * j + 2] - positions[3 * i + 2],
            };
            minImage(dr, cell, periodic);
            const double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            const double r = std::sqrt(r2);
            if (r > cutoff || r < 1e-14) continue;
            const double fc = shiftedCosine(r);
            double x = r / cutoff;
            if (x < 0.0) x = 0.0;
            if (x > 1.0) x = 1.0;
            const double B[2] = {(1.0 - x) * fc, x * fc};
            const int s = speciesIndex(types[j]);
            const double inv = 1.0 / r;
            const double hat[3] = {dr[0] * inv, dr[1] * inv, dr[2] * inv};
            for (int p = 0; p < nRadial; p++) {
                c0[s][p] += B[p];
                for (int m = 0; m < 3; m++)
                    c1[s][p][m] += B[p] * hat[m];
            }
        }
        double c0f[6];
        double c1f[6][3];
        for (int s = 0; s < nSpecies; s++) {
            for (int p = 0; p < nRadial; p++) {
                const int a = s * nRadial + p;
                c0f[a] = c0[s][p];
                for (int m = 0; m < 3; m++)
                    c1f[a][m] = c1[s][p][m];
            }
        }
        for (int a = 0; a < soapL0; a++) {
            for (int b = 0; b < soapL0; b++) {
                out[a * soapL0 + b] = c0f[a] * c0f[b];
                double p1 = 0.0;
                for (int m = 0; m < 3; m++)
                    p1 += c1f[a][m] * c1f[b][m];
                out[soapL0 * soapL0 + a * soapL0 + b] = p1;
            }
        }
    }

    static bool wantsPositions(const std::vector<metatomic::Quantity>& requested) {
        for (const auto& request : requested) {
            for (auto gradient : request.gradients()) {
                if (gradient == metatomic::Gradients::Positions)
                    return true;
            }
        }
        return false;
    }

    static metatensor::TensorMap packEnergy(size_t n, double energy,
                                            const std::vector<double>& dEdr,
                                            const metatomic::Quantity& request,
                                            bool withGrad) {
        auto values = std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{1, 1}, std::vector<double>{energy}
        );
        auto samples = metatensor::Labels({"system"}, {{0}});
        auto properties = metatensor::Labels({"energy"}, {{0}});
        auto block = metatensor::TensorBlock(std::move(values), samples, {}, properties);
        if (withGrad) {
            auto gvalues = std::make_unique<metatensor::SimpleDataArray<double>>(
                std::vector<uintptr_t>{n, 3, 1}, dEdr
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
            block.add_gradient(
                "positions",
                metatensor::TensorBlock(std::move(gvalues), gsamples, {xyz}, properties)
            );
        }
        std::vector<metatensor::TensorBlock> blocks;
        blocks.push_back(std::move(block));
        return metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
    }

    static std::vector<double> copyPositions(const metatomic::System& system) {
        auto tensor = system.positions();
        const DLTensor& t = tensor.as_dlpack()->dl_tensor;
        const size_t n = static_cast<size_t>(t.shape[0]);
        std::vector<double> out(3 * n);
        const char* base = static_cast<const char*>(t.data) + t.byte_offset;
        const int64_t si = t.strides ? t.strides[0] : 3;
        const int64_t sj = t.strides ? t.strides[1] : 1;
        const auto* data = reinterpret_cast<const double*>(base);
        for (size_t i = 0; i < n; i++) {
            out[3 * i + 0] = data[static_cast<int64_t>(i) * si + 0 * sj];
            out[3 * i + 1] = data[static_cast<int64_t>(i) * si + 1 * sj];
            out[3 * i + 2] = data[static_cast<int64_t>(i) * si + 2 * sj];
        }
        return out;
    }

    static std::vector<int32_t> copyTypes(const metatomic::System& system) {
        auto tensor = system.types();
        const DLTensor& t = tensor.as_dlpack()->dl_tensor;
        const size_t n = static_cast<size_t>(t.shape[0]);
        std::vector<int32_t> out(n);
        const char* base = static_cast<const char*>(t.data) + t.byte_offset;
        const int64_t stride = t.strides ? t.strides[0] : 1;
        const auto* data = reinterpret_cast<const int32_t*>(base);
        for (size_t i = 0; i < n; i++)
            out[i] = data[static_cast<int64_t>(i) * stride];
        return out;
    }

    static std::vector<double> copyCell(const metatomic::System& system) {
        auto tensor = system.cell();
        const DLTensor& t = tensor.as_dlpack()->dl_tensor;
        std::vector<double> out(9);
        const char* base = static_cast<const char*>(t.data) + t.byte_offset;
        const auto* data = reinterpret_cast<const double*>(base);
        const int64_t s0 = t.strides ? t.strides[0] : 3;
        const int64_t s1 = t.strides ? t.strides[1] : 1;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                out[3 * i + j] = data[i * s0 + j * s1];
        return out;
    }

    static bool copyPeriodic(const metatomic::System& system) {
        auto tensor = system.pbc();
        const DLTensor& t = tensor.as_dlpack()->dl_tensor;
        const char* base = static_cast<const char*>(t.data) + t.byte_offset;
        if (t.dtype.code == kDLBool || t.dtype.bits == 8) {
            return static_cast<const uint8_t*>(static_cast<const void*>(base))[0] != 0;
        }
        return false;
    }
};

} // namespace OpenMMMetatomic

#endif
