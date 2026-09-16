#ifndef OPENMM_METATOMICFORCEIMPL_H_
#define OPENMM_METATOMICFORCEIMPL_H_

#include "openmmmetatomic/MetatomicForce.h"
#include "openmmmetatomic/internal/MetatomicEvaluator.h"
#include "openmm/internal/CustomCPPForceImpl.h"
#include <memory>

namespace OpenMMMetatomic {

class OPENMM_EXPORT_METATOMIC MetatomicForceImpl : public OpenMM::CustomCPPForceImpl {
public:
    MetatomicForceImpl(const MetatomicForce& owner);
    ~MetatomicForceImpl();

    void initialize(OpenMM::ContextImpl& context);
    double computeForce(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& positions,
                        std::vector<OpenMM::Vec3>& forces);
    const MetatomicForce& getOwner() const {
        return owner;
    }

private:
    const MetatomicForce& owner;
    std::unique_ptr<MetatomicEvaluator> evaluator;
};

} // namespace OpenMMMetatomic

#endif
