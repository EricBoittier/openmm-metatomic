"""Round-trip MetatomicForce through the version-2 XML proxy."""

import openmm as mm
import pytest

pytest.importorskip("openmmmetatomic")

from openmmmetatomic import MetatomicForce


def configured():
    force = MetatomicForce("model.pt")
    force.setDevice("cpu")
    force.setExtensionsDirectory("/tmp/extensions")
    force.setCheckConsistency(True)
    force.setBackend("torch")
    force.setAtomicTypes([1, 6, 8])
    force.setParticles([3, 4, 5])
    force.setPeriodicDirections(True, False, True)
    force.setNonConservative("forces")
    force.setVariant("energy", "pbe")
    force.setVariant("energy_uncertainty", "ensemble")
    force.setUncertaintyThreshold(0.25)
    force.setCharge(-1.5)
    force.setSpinMultiplicity(3.0)
    force.setForceGroup(2)
    return force


def round_trip(xml):
    # cast() returns a reference, so the deserialized owner has to outlive it;
    # the SWIG wrapper keeps it alive through the returned proxy.
    return MetatomicForce.cast(mm.XmlSerializer.deserialize(xml))


def test_every_field_survives():
    force = configured()
    copy = round_trip(mm.XmlSerializer.serialize(force))

    assert copy.getModelPath() == "model.pt"
    assert copy.getDevice() == "cpu"
    assert copy.getExtensionsDirectory() == "/tmp/extensions"
    assert copy.getCheckConsistency()
    assert copy.getBackend() == "torch"
    assert list(copy.getAtomicTypes()) == [1, 6, 8]
    assert list(copy.getParticles()) == [3, 4, 5]
    assert [copy.getPeriodicDirection(i) for i in range(3)] == [True, False, True]
    assert copy.usesPeriodicBoundaryConditions()
    assert copy.getNonConservative() == "forces"
    assert sorted(copy.getVariantOutputs()) == ["energy", "energy_uncertainty"]
    assert copy.getVariant("energy") == "pbe"
    assert copy.getVariant("energy_uncertainty") == "ensemble"
    assert copy.getUncertaintyThreshold() == pytest.approx(0.25)
    assert copy.getCharge() == pytest.approx(-1.5)
    assert copy.getSpinMultiplicity() == pytest.approx(3.0)
    assert copy.getForceGroup() == 2


def test_version_1_still_reads():
    force = MetatomicForce("model.pt")
    force.setAtomicTypes([1, 1, 8])
    force.setUsesPeriodicBoundaryConditions(True)
    xml = mm.XmlSerializer.serialize(force)

    # A version-1 document has neither the new attributes nor the new children.
    v1 = xml.replace('version="2"', 'version="1"')
    for tag in ("Particles", "Variants"):
        v1 = v1.replace(f"\t<{tag}/>\n", "").replace(f"\t<{tag}>\n\t</{tag}>\n", "")
    copy = round_trip(v1)

    assert list(copy.getAtomicTypes()) == [1, 1, 8]
    assert all(copy.getPeriodicDirection(i) for i in range(3))
    assert copy.getParticles() == ()
    assert copy.getNonConservative() == ""
    assert copy.getVariantOutputs() == ()
    assert copy.getUncertaintyThreshold() == pytest.approx(-1.0)
    assert copy.getCharge() == pytest.approx(0.0)
    assert copy.getSpinMultiplicity() == pytest.approx(1.0)


def test_unknown_version_is_refused():
    xml = mm.XmlSerializer.serialize(MetatomicForce("model.pt"))
    with pytest.raises(Exception, match="unsupported version"):
        mm.XmlSerializer.deserialize(xml.replace('version="2"', 'version="3"'))


def test_a_force_in_a_system_round_trips():
    system = mm.System()
    for _ in range(6):
        system.addParticle(1.0)
    system.addForce(configured())

    copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    force = MetatomicForce.cast(copy.getForce(0))
    assert list(force.getParticles()) == [3, 4, 5]
    assert force.getNonConservative() == "forces"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
