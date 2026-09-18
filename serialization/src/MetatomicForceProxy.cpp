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
    node.setIntProperty("version", 2);
    const MetatomicForce& force = *reinterpret_cast<const MetatomicForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setStringProperty("modelPath", force.getModelPath());
    node.setStringProperty("device", force.getDevice());
    node.setStringProperty("extensionsDirectory", force.getExtensionsDirectory());
    node.setBoolProperty("checkConsistency", force.getCheckConsistency());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setBoolProperty("periodicA", force.getPeriodicDirection(0));
    node.setBoolProperty("periodicB", force.getPeriodicDirection(1));
    node.setBoolProperty("periodicC", force.getPeriodicDirection(2));
    node.setStringProperty("backend", force.getBackend());
    node.setStringProperty("nonConservative", force.getNonConservative());
    node.setDoubleProperty("uncertaintyThreshold", force.getUncertaintyThreshold());
    node.setDoubleProperty("charge", force.getCharge());
    node.setDoubleProperty("spinMultiplicity", force.getSpinMultiplicity());
    SerializationNode& types = node.createChildNode("AtomicTypes");
    for (int type : force.getAtomicTypes())
        types.createChildNode("Type").setIntProperty("value", type);
    SerializationNode& particles = node.createChildNode("Particles");
    for (int index : force.getParticles())
        particles.createChildNode("Particle").setIntProperty("index", index);
    SerializationNode& variants = node.createChildNode("Variants");
    for (const auto& output : force.getVariantOutputs())
        variants.createChildNode("Variant")
            .setStringProperty("output", output)
            .setStringProperty("variant", force.getVariant(output));
}

void* MetatomicForceProxy::deserialize(const SerializationNode& node) const {
    const int version = node.getIntProperty("version");
    if (version != 1 && version != 2)
        throw OpenMMException("MetatomicForceProxy: unsupported version number");
    auto* force = new MetatomicForce(node.getStringProperty("modelPath"));
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setDevice(node.getStringProperty("device", ""));
        force->setExtensionsDirectory(node.getStringProperty("extensionsDirectory", ""));
        force->setCheckConsistency(node.getBoolProperty("checkConsistency", false));
        force->setBackend(node.getStringProperty("backend", "auto"));
        const bool periodic = node.getBoolProperty("usesPeriodic", false);
        force->setPeriodicDirections(
            node.getBoolProperty("periodicA", periodic),
            node.getBoolProperty("periodicB", periodic),
            node.getBoolProperty("periodicC", periodic)
        );
        force->setNonConservative(node.getStringProperty("nonConservative", ""));
        force->setUncertaintyThreshold(node.getDoubleProperty("uncertaintyThreshold", -1.0));
        force->setCharge(node.getDoubleProperty("charge", 0.0));
        force->setSpinMultiplicity(node.getDoubleProperty("spinMultiplicity", 1.0));
        vector<int> types, particles;
        for (const auto& child : node.getChildren()) {
            if (child.getName() == "AtomicTypes") {
                for (const auto& typeNode : child.getChildren())
                    types.push_back(typeNode.getIntProperty("value"));
            }
            else if (child.getName() == "Particles") {
                for (const auto& particleNode : child.getChildren())
                    particles.push_back(particleNode.getIntProperty("index"));
            }
            else if (child.getName() == "Variants") {
                for (const auto& variantNode : child.getChildren())
                    force->setVariant(
                        variantNode.getStringProperty("output"),
                        variantNode.getStringProperty("variant")
                    );
            }
        }
        force->setAtomicTypes(types);
        force->setParticles(particles);
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
