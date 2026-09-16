/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/MetatomicForceImpl.h"

using namespace OpenMMMetatomic;
using namespace OpenMM;
using namespace std;

MetatomicForce::MetatomicForce(const string& modelPath) :
    modelPath(modelPath), checkConsistency(false), usePeriodic(false) {
    setName("MetatomicForce");
}

void MetatomicForce::setDevice(const string& device) {
    this->device = device;
}

void MetatomicForce::setExtensionsDirectory(const string& path) {
    extensionsDirectory = path;
}

void MetatomicForce::setCheckConsistency(bool enabled) {
    checkConsistency = enabled;
}

void MetatomicForce::setAtomicTypes(const vector<int>& types) {
    atomicTypes = types;
}

void MetatomicForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

const string& MetatomicForce::getModelPath() const {
    return modelPath;
}

const string& MetatomicForce::getDevice() const {
    return device;
}

const string& MetatomicForce::getExtensionsDirectory() const {
    return extensionsDirectory;
}

bool MetatomicForce::getCheckConsistency() const {
    return checkConsistency;
}

const vector<int>& MetatomicForce::getAtomicTypes() const {
    return atomicTypes;
}

bool MetatomicForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

ForceImpl* MetatomicForce::createImpl() const {
    return new MetatomicForceImpl(*this);
}
