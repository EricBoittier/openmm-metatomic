#ifndef OPENMM_METATOMIC_EVALUATOR_H_
#define OPENMM_METATOMIC_EVALUATOR_H_

#include "openmm/Vec3.h"
#include "openmmmetatomic/internal/windowsExportMetatomic.h"
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace OpenMMMetatomic {

class MetatomicEvaluatorImpl;

/**
 * Loads an exported Metatomic model and evaluates system energy plus
 * conservative forces. Positions are nanometers; energy is kJ/mol; forces are
 * kJ/mol/nm. This is the shared evaluation engine used by the OpenMM force
 * and by the standalone spike.
 */
class OPENMM_EXPORT_METATOMIC MetatomicEvaluator {
public:
    struct Config {
        std::string modelPath;
        std::string device;
        std::string extensionsDirectory;
        bool checkConsistency = false;
        std::vector<int> atomicTypes;
        /// Periodicity per box vector; a non-periodic direction gets a zero cell row.
        std::array<bool, 3> pbc = {false, false, false};
        std::string backend = "auto";
        /// Take forces and/or stress from the model instead of autograd.
        bool nonConservativeForces = false;
        bool nonConservativeStress = false;
        /// Output name -> variant, for models that publish several variants.
        std::map<std::string, std::string> variants;
        /// Per-atom uncertainty threshold in eV; negative disables the check.
        double uncertaintyThreshold = -1.0;
        double charge = 0.0;
        double spinMultiplicity = 1.0;

        bool anyPeriodic() const {
            return pbc[0] || pbc[1] || pbc[2];
        }
        bool allPeriodic() const {
            return pbc[0] && pbc[1] && pbc[2];
        }
    };

    struct Result {
        double energy = 0.0;
        std::vector<OpenMM::Vec3> forces;
        /// Largest per-atom energy uncertainty in eV, or -1 if not requested.
        double maxUncertainty = -1.0;
    };

    struct ModelInfo {
        std::string dtype;
        std::string lengthUnit;
        std::string energyKey;
        std::string nonConservativeForceKey;
        std::string nonConservativeStressKey;
        std::string uncertaintyKey;
        std::vector<std::string> supportedDevices;
        std::vector<int64_t> atomicTypes;
        std::vector<std::string> requestedInputs;
        int neighborListRequests = 0;
        std::string device;
        std::string backend;
    };

    explicit MetatomicEvaluator(const Config& config);
    ~MetatomicEvaluator();

    MetatomicEvaluator(const MetatomicEvaluator&) = delete;
    MetatomicEvaluator& operator=(const MetatomicEvaluator&) = delete;

    const ModelInfo& info() const;
    Result compute(const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3 boxVectors[3]) const;

private:
    std::unique_ptr<MetatomicEvaluatorImpl> impl;
};

} // namespace OpenMMMetatomic

#endif
