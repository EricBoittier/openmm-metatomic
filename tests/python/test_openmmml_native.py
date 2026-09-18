"""``MLPotential("metatomic-native")`` must match ``MLPotential("metatomic")``.

The native plugin and the ``openmm.PythonForce`` backend evaluate the same
exported model, so every System OpenMM-ML can build from one of them -- pure ML,
mechanically embedded, ``lambda_interpolate`` -- has to give the same energy and
forces from the other. Mirrors ``openmm-ml/test/TestMetatomicPotential.py``.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

torch = pytest.importorskip("torch", reason="torch is not installed")
mta = pytest.importorskip("metatomic.torch", reason="metatomic-torch is not installed")
pytest.importorskip("openmmml", reason="openmm-ml is not installed")

from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa: E402

import _openmm  # noqa: E402
from openmmml import MLPotential  # noqa: E402

pytest.importorskip("openmmmetatomic")
import openmmmetatomic  # noqa: E402

ML_ATOMS = list(range(15))
DATA = _openmm.openmm_ml_data()


class HarmonicModel(torch.nn.Module):
    """``E = k Σ(r - r0)²`` in the model's own units (Å, kJ/mol)."""

    def __init__(self, force_constant: float, equilibrium_positions: torch.Tensor):
        super().__init__()
        self.force_constant = force_constant
        self.register_buffer("equilibrium_positions", equilibrium_positions)

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            energy[i] += torch.sum(
                self.force_constant
                * (system.positions - self.equilibrium_positions) ** 2
            )
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


@pytest.fixture(scope="module")
def harmonic_toluene(tmp_path_factory):
    if DATA is None:
        pytest.skip("openmm-ml test data not found")
    pdb = app.PDBFile(str(DATA / "toluene" / "toluene.pdb"))
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    path = str(tmp_path_factory.mktemp("model") / "harmonic.pt")
    capabilities = mta.ModelCapabilities(
        outputs={"energy": mta.ModelOutput(unit="kJ/mol", sample_kind="system")},
        atomic_types=sorted(set(numbers)),
        interaction_range=0.0,
        length_unit="Angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    model = HarmonicModel(1.0, torch.tensor(positions, dtype=torch.float64))
    mta.AtomisticModel(model.eval(), mta.ModelMetadata(), capabilities).save(path)
    return pdb, numbers, positions, path


def is_native(force):
    """``System.getForce`` hands back a base ``Force``; the plugin adds a cast."""
    return openmmmetatomic.MetatomicForce.isinstance(force)


def native_force(force):
    return openmmmetatomic.MetatomicForce.cast(force)


def potentials(model_path):
    openmmmetatomic.register()
    _openmm.ensure_openmm_ml_embeddings()
    return (
        MLPotential("metatomic", modelPath=model_path, device="cpu"),
        MLPotential("metatomic-native", modelPath=model_path, device="cpu"),
    )


def energy_forces(system, positions, platform_name="Reference"):
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001),
        mm.Platform.getPlatformByName(platform_name),
    )
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    del context
    return energy, np.asarray(forces, dtype=np.float64)


def test_pure_system_matches_python_force(harmonic_toluene):
    pdb, _, positions, model_path = harmonic_toluene
    displaced = (positions + 0.1) * unit.angstrom
    reference, native = potentials(model_path)
    e_ref, f_ref = energy_forces(reference.createSystem(pdb.topology), displaced)
    e_native, f_native = energy_forces(native.createSystem(pdb.topology), displaced)
    assert np.isclose(e_native, e_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f_native, f_ref, rtol=1e-5, atol=1e-6)


def mixed_systems(potential, topology, mm_system, interpolate=False):
    # The native backend answers getMLLongRange() with False, so mechanical
    # embedding refuses an explicit mlLongRange; the PythonForce backend returns
    # None and needs to be told. Either way the ML part stays short-ranged.
    args = (
        {}
        if potential._impl.getMLLongRange() is not None
        else {"mlLongRange": False}
    )
    return potential.createMixedSystem(
        topology,
        mm_system,
        ML_ATOMS,
        interpolate=interpolate,
        **args,
    )


def test_mixed_system_matches_python_force(harmonic_toluene):
    _, _, _, model_path = harmonic_toluene
    prmtop = app.AmberPrmtopFile(str(DATA / "toluene" / "toluene-explicit.prm7"))
    inpcrd = app.AmberInpcrdFile(str(DATA / "toluene" / "toluene-explicit.rst7"))
    mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
    reference, native = potentials(model_path)

    e_ref, f_ref = energy_forces(
        mixed_systems(reference, prmtop.topology, mm_system), inpcrd.positions
    )
    e_native, f_native = energy_forces(
        mixed_systems(native, prmtop.topology, mm_system), inpcrd.positions
    )
    assert np.isclose(e_native, e_ref, rtol=1e-6)
    np.testing.assert_allclose(f_native, f_ref, rtol=1e-4, atol=1e-4)

    # The native force must respect lambda_interpolate the same way, which it
    # gets for free: mechanical embedding only calls addForces().
    interp = mixed_systems(native, prmtop.topology, mm_system, interpolate=True)
    context = mm.Context(
        interp, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(inpcrd.positions)
    at_one = context.getState(getEnergy=True).getPotentialEnergy()
    context.setParameter("lambda_interpolate", 0)
    at_zero = context.getState(getEnergy=True).getPotentialEnergy()
    del context
    assert np.isclose(
        at_one.value_in_unit(unit.kilojoules_per_mole), e_native, rtol=1e-5
    )
    mm_energy, _ = energy_forces(mm_system, inpcrd.positions)
    assert np.isclose(
        at_zero.value_in_unit(unit.kilojoules_per_mole), mm_energy, rtol=1e-5
    )


def test_native_force_carries_the_subset(harmonic_toluene):
    _, _, _, model_path = harmonic_toluene
    prmtop = app.AmberPrmtopFile(str(DATA / "toluene" / "toluene-explicit.prm7"))
    mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
    _, native = potentials(model_path)
    system = mixed_systems(native, prmtop.topology, mm_system)
    forces = [native_force(f) for f in system.getForces() if is_native(f)]
    assert forces
    assert tuple(forces[0].getParticles()) == tuple(ML_ATOMS)
    assert forces[0].usesPeriodicBoundaryConditions()
    assert native._impl.getMLLongRange() is False


def test_partial_pbc_reaches_the_force(harmonic_toluene):
    pdb, _, _, model_path = harmonic_toluene
    _, native = potentials(model_path)
    system = native.createSystem(pdb.topology, pbc=[True, False, True])
    force = native_force(system.getForce(0))
    assert [force.getPeriodicDirection(i) for i in range(3)] == [True, False, True]


def test_charge_and_spin_reach_the_force(harmonic_toluene):
    pdb, _, _, model_path = harmonic_toluene
    _, native = potentials(model_path)
    system = native.createSystem(pdb.topology, charge=-1.0, multiplicity=3)
    force = native_force(system.getForce(0))
    assert force.getCharge() == -1.0
    assert force.getSpinMultiplicity() == 3.0


def test_non_conservative_needs_an_output(harmonic_toluene):
    pdb, _, _, model_path = harmonic_toluene
    openmmmetatomic.register()
    potential = MLPotential(
        "metatomic-native",
        modelPath=model_path,
        device="cpu",
        non_conservative="forces",
    )
    system = potential.createSystem(pdb.topology)
    with pytest.raises(Exception, match="non_conservative_force"):
        mm.Context(
            system,
            mm.VerletIntegrator(0.001),
            mm.Platform.getPlatformByName("Reference"),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
