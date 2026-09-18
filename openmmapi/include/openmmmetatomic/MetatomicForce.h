#ifndef OPENMM_METATOMICFORCE_H_
#define OPENMM_METATOMICFORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Eric D. Boittier                                                  *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Force.h"
#include "openmmmetatomic/internal/windowsExportMetatomic.h"
#include <map>
#include <string>
#include <vector>

namespace OpenMMMetatomic {

/**
 * A Force that evaluates an exported Metatomic model, either for a complete
 * OpenMM System or for a subset of its particles. The public object stores
 * serializable configuration, not live Torch objects. The model is loaded when
 * a Context is created.
 *
 * Positions are taken in nanometers and the returned energy is in kJ/mol.
 * Forces are -dE/dx by default, or a model's direct force head when
 * setNonConservative() asks for one. Atomic types must be set explicitly; they
 * are not inferred from particle masses.
 */
class OPENMM_EXPORT_METATOMIC MetatomicForce : public OpenMM::Force {
public:
    explicit MetatomicForce(const std::string& modelPath);

    void setDevice(const std::string& device);
    void setExtensionsDirectory(const std::string& path);
    void setCheckConsistency(bool enabled);
    void setAtomicTypes(const std::vector<int>& types);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /// Backend: "auto" (default), "torch" (TorchScript .pt), or "core" (metatomic-core).
    void setBackend(const std::string& backend);

    /**
     * Restrict the model to a subset of the System's particles, in the order the
     * model should see them. An empty list (the default) means every particle.
     * getAtomicTypes() must then have one entry per listed particle. Forces on
     * particles outside the subset are left untouched, so a conventional force
     * field can cover them, as in OpenMM-ML's mixed ML/MM systems.
     */
    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

    /**
     * Periodicity per box vector. setUsesPeriodicBoundaryConditions() sets all
     * three at once, and usesPeriodicBoundaryConditions() is true if any
     * direction is periodic. Non-periodic directions get a zero cell row.
     */
    void setPeriodicDirections(bool a, bool b, bool c);
    bool getPeriodicDirection(int axis) const;

    /**
     * Where forces come from: "" (default, autograd on the energy), "forces"
     * (the model's non_conservative_force output), "stress", or "both".
     * A non-conservative stress is requested but not used: OpenMM has no virial
     * path, so only a MonteCarlo barostat can drive NPT, off the energy.
     */
    void setNonConservative(const std::string& mode);
    const std::string& getNonConservative() const;

    /// Select a named variant of an output ("energy", "energy_uncertainty",
    /// "non_conservative_force", "non_conservative_stress").
    void setVariant(const std::string& output, const std::string& variant);
    std::string getVariant(const std::string& output) const;
    std::vector<std::string> getVariantOutputs() const;

    /// Per-atom energy uncertainty threshold in eV. Negative (the default)
    /// disables the check; otherwise a model that provides energy_uncertainty
    /// warns whenever an atom exceeds the threshold.
    void setUncertaintyThreshold(double eVPerAtom);
    double getUncertaintyThreshold() const;

    /// Total charge and spin multiplicity, used only if the model requests them.
    void setCharge(double charge);
    double getCharge() const;
    void setSpinMultiplicity(double multiplicity);
    double getSpinMultiplicity() const;

    const std::string& getModelPath() const;
    const std::string& getDevice() const;
    const std::string& getExtensionsDirectory() const;
    bool getCheckConsistency() const;
    const std::vector<int>& getAtomicTypes() const;
    bool usesPeriodicBoundaryConditions() const;
    const std::string& getBackend() const;

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    std::string modelPath;
    std::string device;
    std::string extensionsDirectory;
    std::string backend;
    std::string nonConservative;
    bool checkConsistency;
    bool pbc[3];
    double uncertaintyThreshold;
    double charge;
    double spinMultiplicity;
    std::vector<int> atomicTypes;
    std::vector<int> particles;
    std::map<std::string, std::string> variants;
};

} // namespace OpenMMMetatomic

#endif
