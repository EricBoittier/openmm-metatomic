/* -------------------------------------------------------------------------- *
 *                              OpenMM-Metatomic                              *
 * -------------------------------------------------------------------------- */

#include "openmmmetatomic/serialization/MetatomicForceProxy.h"
#include "openmmmetatomic/MetatomicForce.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/OpenMMException.h"

using namespace OpenMMMetatomic;
using namespace OpenMM;
using namespace std;

MetatomicForceProxy::MetatomicForceProxy() : SerializationProxy("MetatomicForce") {
}

void MetatomicForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const MetatomicForce& force = *reinterpret_cast<const MetatomicForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setStringProperty("modelPath", force.getModelPath());
    node.setStringProperty("device", force.getDevice());
    node.setStringProperty("extensionsDirectory", force.getExtensionsDirectory());
    node.setBoolProperty("checkConsistency", force.getCheckConsistency());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    SerializationNode& types = node.createChildNode("AtomicTypes");
    for (int type : force.getAtomicTypes())
        types.createChildNode("Type").setIntProperty("value", type);
}

void* MetatomicForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("MetatomicForceProxy: unsupported version number");
    auto* force = new MetatomicForce(node.getStringProperty("modelPath"));
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setDevice(node.getStringProperty("device", ""));
        force->setExtensionsDirectory(node.getStringProperty("extensionsDirectory", ""));
        force->setCheckConsistency(node.getBoolProperty("checkConsistency", false));
        force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic", false));
        vector<int> types;
        for (const auto& child : node.getChildren()) {
            if (child.getName() == "AtomicTypes") {
                for (const auto& typeNode : child.getChildren())
                    types.push_back(typeNode.getIntProperty("value"));
            }
        }
        force->setAtomicTypes(types);
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
