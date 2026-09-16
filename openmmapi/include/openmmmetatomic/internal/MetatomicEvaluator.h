#ifndef OPENMM_METATOMIC_EVALUATOR_H_
#define OPENMM_METATOMIC_EVALUATOR_H_

#include "openmm/Vec3.h"
#include "openmmmetatomic/internal/windowsExportMetatomic.h"
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
        bool periodic = false;
    };

    struct Result {
        double energy = 0.0;
        std::vector<OpenMM::Vec3> forces;
    };

    struct ModelInfo {
        std::string dtype;
        std::string lengthUnit;
        std::string energyKey;
        std::vector<std::string> supportedDevices;
        std::vector<int64_t> atomicTypes;
        std::vector<std::string> requestedInputs;
        int neighborListRequests = 0;
        std::string device;
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
