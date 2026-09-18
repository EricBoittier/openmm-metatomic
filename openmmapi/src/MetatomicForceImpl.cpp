/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/internal/MetatomicForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/System.h"

using namespace OpenMMMetatomic;
using namespace OpenMM;
using namespace std;

MetatomicForceImpl::MetatomicForceImpl(const MetatomicForce& owner) :
    CustomCPPForceImpl(owner), owner(owner) {
}

MetatomicForceImpl::~MetatomicForceImpl() = default;

void MetatomicForceImpl::initialize(ContextImpl& context) {
    const int n = context.getSystem().getNumParticles();
    particles = owner.getParticles();
    for (int index : particles)
        if (index >= n)
            throw OpenMMException(
                "MetatomicForce: particle index " + to_string(index) +
                " is out of range for a System with " + to_string(n) + " particles"
            );
    const size_t expected = particles.empty() ? static_cast<size_t>(n) : particles.size();
    if (owner.getAtomicTypes().size() != expected) {
        throw OpenMMException(
            "MetatomicForce: atomic types must be set explicitly for every particle "
            "the model sees (got " + to_string(owner.getAtomicTypes().size()) +
            " types for " + to_string(expected) + " particles). Do not infer types from masses."
        );
    }
    MetatomicEvaluator::Config config;
    config.modelPath = owner.getModelPath();
    config.device = owner.getDevice();
    config.extensionsDirectory = owner.getExtensionsDirectory();
    config.checkConsistency = owner.getCheckConsistency();
    config.atomicTypes = owner.getAtomicTypes();
    for (int axis = 0; axis < 3; axis++)
        config.pbc[axis] = owner.getPeriodicDirection(axis);
    config.backend = owner.getBackend();
    const string& nonConservative = owner.getNonConservative();
    config.nonConservativeForces = nonConservative == "forces" || nonConservative == "both";
    config.nonConservativeStress = nonConservative == "stress" || nonConservative == "both";
    for (const auto& output : owner.getVariantOutputs())
        config.variants[output] = owner.getVariant(output);
    config.uncertaintyThreshold = owner.getUncertaintyThreshold();
    config.charge = owner.getCharge();
    config.spinMultiplicity = owner.getSpinMultiplicity();
    evaluator = make_unique<MetatomicEvaluator>(config);
    subset.resize(particles.size());
    CustomCPPForceImpl::initialize(context);
}

double MetatomicForceImpl::computeForce(ContextImpl& context, const vector<Vec3>& positions,
                                        vector<Vec3>& forces) {
    Vec3 a, b, c;
    context.getPeriodicBoxVectors(a, b, c);
    const Vec3 box[3] = {a, b, c};
    if (particles.empty()) {
        auto result = evaluator->compute(positions, box);
        forces = std::move(result.forces);
        return result.energy;
    }
    for (size_t i = 0; i < particles.size(); i++)
        subset[i] = positions[particles[i]];
    auto result = evaluator->compute(subset, box);
    // The platform kernels keep this buffer between calls and add it into the
    // context forces, so everything outside the subset has to be cleared.
    forces.assign(positions.size(), Vec3());
    for (size_t i = 0; i < particles.size(); i++)
        forces[particles[i]] = result.forces[i];
    return result.energy;
}
