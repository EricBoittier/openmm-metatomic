/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/MetatomicForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMMMetatomic;
using namespace OpenMM;
using namespace std;

MetatomicForce::MetatomicForce(const string& modelPath) :
    modelPath(modelPath), backend("auto"), checkConsistency(false),
    pbc{false, false, false}, uncertaintyThreshold(-1.0), charge(0.0),
    spinMultiplicity(1.0) {
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
    setPeriodicDirections(periodic, periodic, periodic);
}

void MetatomicForce::setBackend(const string& backend) {
    if (backend != "auto" && backend != "torch" && backend != "core")
        throw OpenMMException("MetatomicForce: backend must be auto, torch, or core");
    this->backend = backend;
}

void MetatomicForce::setParticles(const vector<int>& particles) {
    for (int index : particles)
        if (index < 0)
            throw OpenMMException("MetatomicForce: particle indices must be non-negative");
    this->particles = particles;
}

const vector<int>& MetatomicForce::getParticles() const {
    return particles;
}

void MetatomicForce::setPeriodicDirections(bool a, bool b, bool c) {
    pbc[0] = a;
    pbc[1] = b;
    pbc[2] = c;
}

bool MetatomicForce::getPeriodicDirection(int axis) const {
    if (axis < 0 || axis > 2)
        throw OpenMMException("MetatomicForce: periodic direction axis must be 0, 1, or 2");
    return pbc[axis];
}

void MetatomicForce::setNonConservative(const string& mode) {
    if (mode != "" && mode != "forces" && mode != "stress" && mode != "both")
        throw OpenMMException(
            "MetatomicForce: nonConservative must be \"\", \"forces\", \"stress\", or \"both\""
        );
    nonConservative = mode;
}

const string& MetatomicForce::getNonConservative() const {
    return nonConservative;
}

void MetatomicForce::setVariant(const string& output, const string& variant) {
    if (variant.empty())
        variants.erase(output);
    else
        variants[output] = variant;
}

string MetatomicForce::getVariant(const string& output) const {
    const auto found = variants.find(output);
    return found == variants.end() ? string() : found->second;
}

vector<string> MetatomicForce::getVariantOutputs() const {
    vector<string> names;
    names.reserve(variants.size());
    for (const auto& entry : variants)
        names.push_back(entry.first);
    return names;
}

void MetatomicForce::setUncertaintyThreshold(double eVPerAtom) {
    uncertaintyThreshold = eVPerAtom;
}

double MetatomicForce::getUncertaintyThreshold() const {
    return uncertaintyThreshold;
}

void MetatomicForce::setCharge(double charge) {
    this->charge = charge;
}

double MetatomicForce::getCharge() const {
    return charge;
}

void MetatomicForce::setSpinMultiplicity(double multiplicity) {
    spinMultiplicity = multiplicity;
}

double MetatomicForce::getSpinMultiplicity() const {
    return spinMultiplicity;
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
    return pbc[0] || pbc[1] || pbc[2];
}

const string& MetatomicForce::getBackend() const {
    return backend;
}

ForceImpl* MetatomicForce::createImpl() const {
    return new MetatomicForceImpl(*this);
}
