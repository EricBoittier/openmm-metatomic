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
#include <string>
#include <vector>

namespace OpenMMMetatomic {

/**
 * A Force that evaluates an exported Metatomic model for a complete OpenMM
 * System. The public object stores serializable configuration, not live Torch
 * objects. The model is loaded when a Context is created.
 *
 * Positions are taken in nanometers and the returned energy is in kJ/mol.
 * Conservative forces are -dE/dx. Atomic types must be set explicitly; they
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

    const std::string& getModelPath() const;
    const std::string& getDevice() const;
    const std::string& getExtensionsDirectory() const;
    bool getCheckConsistency() const;
    const std::vector<int>& getAtomicTypes() const;
    bool usesPeriodicBoundaryConditions() const;

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    std::string modelPath;
    std::string device;
    std::string extensionsDirectory;
    bool checkConsistency;
    bool usePeriodic;
    std::vector<int> atomicTypes;
};

} // namespace OpenMMMetatomic

#endif
