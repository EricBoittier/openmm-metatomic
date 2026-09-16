#ifndef OPENMM_METATOMICFORCEPROXY_H_
#define OPENMM_METATOMICFORCEPROXY_H_

#include "openmm/serialization/SerializationProxy.h"
#include "openmmmetatomic/internal/windowsExportMetatomic.h"

namespace OpenMMMetatomic {

class OPENMM_EXPORT_METATOMIC MetatomicForceProxy : public OpenMM::SerializationProxy {
public:
    MetatomicForceProxy();
    void serialize(const void* object, OpenMM::SerializationNode& node) const;
    void* deserialize(const OpenMM::SerializationNode& node) const;
};

} // namespace OpenMMMetatomic

#endif
