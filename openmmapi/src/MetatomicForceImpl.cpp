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
    if (static_cast<int>(owner.getAtomicTypes().size()) != n) {
        throw OpenMMException(
            "MetatomicForce: atomic types must be set explicitly for every particle "
            "(got " + to_string(owner.getAtomicTypes().size()) + " types for " +
            to_string(n) + " particles). Do not infer types from masses."
        );
    }
    MetatomicEvaluator::Config config;
    config.modelPath = owner.getModelPath();
    config.device = owner.getDevice();
    config.extensionsDirectory = owner.getExtensionsDirectory();
    config.checkConsistency = owner.getCheckConsistency();
    config.atomicTypes = owner.getAtomicTypes();
    config.periodic = owner.usesPeriodicBoundaryConditions();
    config.backend = owner.getBackend();
    evaluator = make_unique<MetatomicEvaluator>(config);
    CustomCPPForceImpl::initialize(context);
}

double MetatomicForceImpl::computeForce(ContextImpl& context, const vector<Vec3>& positions,
                                        vector<Vec3>& forces) {
    Vec3 a, b, c;
    context.getPeriodicBoxVectors(a, b, c);
    const Vec3 box[3] = {a, b, c};
    auto result = evaluator->compute(positions, box);
    forces = result.forces;
    return result.energy;
}
