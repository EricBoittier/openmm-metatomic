"""Try to break ``MetatomicForce`` on a complex mixed ML/MM System.

Amber19 ACE-ALA-NME in explicit water already carries every bonded term the
plugin will see in a real mixed System (bonds, angles, proper and improper
torsions, CMAP, PME, rigid water, H-mass repartitioning). CHARMM36 adds NBFix
and 1-4 custom bonds. On top of that: NaCl, an injected Ryckaert–Bellemans
torsion, link atoms, ``lambda_interpolate``, a reversed / gapped subset, two
native forces in one System, a 2,000-atom ML water region, and a few MD steps
to catch a stale force buffer.
"""

import os
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

torch = pytest.importorskip("torch", reason="torch is not installed")
pytest.importorskip("metatomic.torch", reason="metatomic-torch is not installed")
pytest.importorskip("openmmml", reason="openmm-ml is not installed")

import _openmm  # noqa: E402
import _complex_mixed as _cm  # noqa: E402
from openmmml import MLPotential  # noqa: E402

import openmmmetatomic  # noqa: E402

if _cm.alanine_path() is None:
    pytest.skip("openmm-ml alanine-dipeptide data not found", allow_module_level=True)


def potentials(model_path):
    openmmmetatomic.register()
    _openmm.ensure_openmm_ml_embeddings()
    return (
        MLPotential("metatomic", modelPath=model_path, device="cpu", uncertainty_threshold=None),
        MLPotential(
            "metatomic-native",
            modelPath=model_path,
            device="cpu",
            uncertainty_threshold=None,
        ),
    )


@pytest.fixture(scope="module")
def amber_box():
    topology, positions, system, _ = _cm.load_alanine_box("amber19", add_salt=True)
    info = _cm.inventory(system, topology)
    _cm.require_terms(
        info,
        [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "CMAPTorsionForce",
            "NonbondedForce",
            "RBTorsionForce",
            "CustomExternalForce",
            "CMMotionRemover",
        ],
        "amber19 alanine box",
    )
    assert info["details"].get("improper_torsions", 0) >= 1
    assert info["details"].get("proper_torsions", 0) >= 1
    assert info["details"].get("cmaps", 0) >= 1
    assert info["details"].get("nonbonded_method") == mm.NonbondedForce.PME
    assert info["constraints"] > 0
    assert _cm.ion_atoms(topology), "addIons did not put Na/Cl in the box"
    numbers = [atom.element.atomic_number for atom in topology.atoms()]
    return topology, positions, system, info, numbers


@pytest.fixture(scope="module")
def probe_model(amber_box, tmp_path_factory):
    _, _, _, _, numbers = amber_box
    path = str(tmp_path_factory.mktemp("complex_mixed") / "probe.pt")
    # Na (11) and Cl (17) come from addIons; keep them even if a case never
    # sends them to the model, so a mis-routed subset fails loudly.
    _cm.export_probe_model(path, sorted(set(numbers) | {11, 17}))
    return path


@pytest.fixture(scope="module")
def charmm_box():
    try:
        topology, positions, system, _ = _cm.load_alanine_box("charmm", add_salt=False)
    except Exception as exc:
        pytest.skip(f"charmm36 force field not available: {exc}")
    info = _cm.inventory(system, topology)
    _cm.require_terms(
        info,
        [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
            "CMAPTorsionForce",
            "NonbondedForce",
            "CustomNonbondedForce",
            "CustomBondForce",
        ],
        "charmm36 alanine box",
    )
    return topology, positions, system, info


def _agree(system_a, system_b, positions, box, rtol=1e-6, atol=1e-6, parameters=None):
    e_a, f_a = _cm.energy_forces(system_a, positions, box, parameters=parameters)
    e_b, f_b = _cm.energy_forces(system_b, positions, box, parameters=parameters)
    assert np.isclose(e_a, e_b, rtol=rtol, atol=atol), f"ΔE={e_b - e_a}"
    n = min(len(f_a), len(f_b))
    np.testing.assert_allclose(f_a[:n], f_b[:n], rtol=rtol, atol=atol)
    return e_a, f_a, e_b, f_b


def test_amber19_inventory_is_actually_complex(amber_box):
    topology, _, _, info, _ = amber_box
    peptide = _cm.peptide_atoms(topology)
    waters = _cm.water_atoms(topology)
    ions = _cm.ion_atoms(topology)
    assert len(peptide) == 22
    assert len(waters) > 2000
    assert len(ions) >= 2
    assert info["particles"] == len(peptide) + len(waters) + len(ions)
    print("amber19", _cm.format_inventory(info), "ions", ions)


def test_peptide_in_water_matches_python_force(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    peptide = _cm.peptide_atoms(topology)
    reference, native = potentials(probe_model)
    mixed_ref = _cm.mixed(reference, topology, mm_system, peptide)
    mixed_nat = _cm.mixed(native, topology, mm_system, peptide)
    force = _cm.native_forces(mixed_nat)[0]
    assert tuple(force.getParticles()) == tuple(peptide)
    assert force.usesPeriodicBoundaryConditions()
    _agree(mixed_ref, mixed_nat, positions, box)


def test_lambda_interpolate_collapses(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    peptide = _cm.peptide_atoms(topology)
    _, native = potentials(probe_model)
    interp = _cm.mixed(native, topology, mm_system, peptide, interpolate=True)
    mixed = _cm.mixed(native, topology, mm_system, peptide, interpolate=False)
    mm_energy, _ = _cm.energy_forces(mm_system, positions, box)
    ml_energy, _ = _cm.energy_forces(mixed, positions, box)

    at_zero, _ = _cm.energy_forces(
        interp, positions, box, parameters={"lambda_interpolate": 0.0}
    )
    at_one, _ = _cm.energy_forces(
        interp, positions, box, parameters={"lambda_interpolate": 1.0}
    )
    at_half, _ = _cm.energy_forces(
        interp, positions, box, parameters={"lambda_interpolate": 0.5}
    )
    assert np.isclose(at_zero, mm_energy, rtol=1e-5, atol=1e-4)
    assert np.isclose(at_one, ml_energy, rtol=1e-5, atol=1e-4)
    assert np.isclose(at_half, 0.5 * (mm_energy + ml_energy), rtol=1e-5, atol=1e-3)


def test_ala_link_atoms_match_python_force(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    ala = _cm.residue_atoms(topology, "ALA")
    reference, native = potentials(probe_model)
    kwargs = dict(interpolate=False, returnInfo=True)
    info_ref = _cm.mixed(reference, topology, mm_system, ala, **kwargs)
    info_nat = _cm.mixed(native, topology, mm_system, ala, **kwargs)
    assert info_nat["system"].getNumParticles() > topology.getNumAtoms()
    assert any(info_nat["system"].isVirtualSite(i) for i in range(info_nat["system"].getNumParticles()))
    links = [
        openmmmetatomic.MetatomicForce.cast(f).getParticles()
        for f in info_nat["system"].getForces()
        if openmmmetatomic.MetatomicForce.isinstance(f)
    ]
    # Link hydrogens are appended and included in the ML subset.
    assert len(links[0]) > len(ala)
    _agree(info_ref["system"], info_nat["system"], positions, box)


def test_gapped_waters_and_reversed_peptide(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    peptide = _cm.peptide_atoms(topology)
    waters = _cm.water_atoms(topology)
    mols = [waters[i : i + 3] for i in range(0, len(waters), 3)]
    gapped = [atom for mol in mols[::2] for atom in mol]
    # Every other water molecule: discontiguous blocks, no link atoms.
    assert len(gapped) > 1000
    reference, native = potentials(probe_model)
    _agree(
        _cm.mixed(reference, topology, mm_system, gapped),
        _cm.mixed(native, topology, mm_system, gapped),
        positions,
        box,
        rtol=1e-5,
        atol=1e-4,
    )

    reversed_atoms = list(reversed(peptide))
    mixed = _cm.mixed(native, topology, mm_system, reversed_atoms)
    force = _cm.native_forces(mixed)[0]
    # Mechanical embedding currently passes atoms in the order given; a silent
    # sort here would pair the wrong atomic types with the wrong positions.
    assert tuple(force.getParticles()) == tuple(reversed_atoms)
    assert list(force.getAtomicTypes()) == [
        list(topology.atoms())[i].element.atomic_number for i in reversed_atoms
    ]
    _agree(
        _cm.mixed(reference, topology, mm_system, reversed_atoms),
        mixed,
        positions,
        box,
    )


def test_double_link_to_one_mm_atom_is_refused(amber_box, probe_model):
    """Two ML hydrogens on the same MM carbon: OpenMM-ML must reject this."""
    topology, _, mm_system, _, _ = amber_box
    peptide = _cm.peptide_atoms(topology)
    # ACE H1 and H2, both bonded to CH3.
    gapped = peptide[::2]
    reference, native = potentials(probe_model)
    with pytest.raises(ValueError, match="Multiple link bonds"):
        _cm.mixed(reference, topology, mm_system, gapped)
    with pytest.raises(ValueError, match="Multiple link bonds"):
        _cm.mixed(native, topology, mm_system, gapped)


def test_waters_as_ml_is_a_large_subset(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    waters = _cm.water_atoms(topology)
    assert len(waters) > 2000
    reference, native = potentials(probe_model)
    mixed_nat = _cm.mixed(native, topology, mm_system, waters)
    force = _cm.native_forces(mixed_nat)[0]
    assert tuple(force.getParticles()) == tuple(waters)
    _agree(
        _cm.mixed(reference, topology, mm_system, waters),
        mixed_nat,
        positions,
        box,
        rtol=1e-5,
        atol=1e-4,
    )


def test_ions_as_ml(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    ions = _cm.ion_atoms(topology)
    reference, native = potentials(probe_model)
    _agree(
        _cm.mixed(reference, topology, mm_system, ions),
        _cm.mixed(native, topology, mm_system, ions),
        positions,
        box,
    )


def test_charge_shift_survives_a_mixed_system(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    box = topology.getPeriodicBoxVectors()
    peptide = _cm.peptide_atoms(topology)
    _, native = potentials(probe_model)
    zero, _ = _cm.energy_forces(
        _cm.mixed(native, topology, mm_system, peptide, charge=0.0),
        positions,
        box,
    )
    charged, _ = _cm.energy_forces(
        _cm.mixed(native, topology, mm_system, peptide, charge=-1.0, multiplicity=3),
        positions,
        box,
    )
    # HarmonicProbe: 10*charge + 100*(spin-1) = -10 + 200 = +190 kJ/mol.
    assert np.isclose(charged - zero, 190.0, atol=1e-6)


def test_stale_force_buffer_after_md_steps(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    peptide = _cm.peptide_atoms(topology)
    _, native = potentials(probe_model)
    mixed = _cm.mixed(native, topology, mm_system, peptide, forceGroup=6)
    integrator = mm.VerletIntegrator(0.0005)
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    context.setVelocitiesToTemperature(300 * unit.kelvin, 7)
    integrator.step(8)
    forces = np.asarray(
        context.getState(getForces=True)
        .getForces(asNumpy=True)
        .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    )
    ml = _cm.native_forces(mixed)[0]
    assert ml.getForceGroup() == 6
    outside = [i for i in range(mixed.getNumParticles()) if i not in set(ml.getParticles())]
    # MM bonded/nonbonded stay on group 0; this is the ML scatter only.
    group = ml.getForceGroup()
    ml_only = np.asarray(
        context.getState(getForces=True, groups={group})
        .getForces(asNumpy=True)
        .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    )
    leaked = np.linalg.norm(ml_only[outside], axis=1).max()
    del context
    assert leaked < 1e-8, f"ML force leaked onto MM atoms: max {leaked}"
    assert np.linalg.norm(forces) > 0.0


def test_two_native_forces_do_not_clobber(probe_model):
    """Two disjoint subsets in one System: a stale buffer would mix them."""
    n = 40
    positions = [mm.Vec3(0.02 * i, 0.01 * (i % 5), 0.0) for i in range(n)] * unit.nanometer
    types = [1, 6, 8, 7] * 10
    left, right = list(range(0, 10)), list(range(30, 40))

    def one_subset(particles):
        system = mm.System()
        for _ in range(n):
            system.addParticle(1.0)
        force = openmmmetatomic.MetatomicForce(probe_model)
        force.setBackend("torch")
        force.setDevice("cpu")
        force.setParticles(particles)
        force.setAtomicTypes([types[i] for i in particles])
        force.setUncertaintyThreshold(-1.0)
        system.addForce(force)
        return system

    both = mm.System()
    for _ in range(n):
        both.addParticle(1.0)
    for particles, group in ((left, 1), (right, 2)):
        force = openmmmetatomic.MetatomicForce(probe_model)
        force.setBackend("torch")
        force.setDevice("cpu")
        force.setParticles(particles)
        force.setAtomicTypes([types[i] for i in particles])
        force.setForceGroup(group)
        force.setUncertaintyThreshold(-1.0)
        both.addForce(force)

    e_left, f_left = _cm.energy_forces(one_subset(left), positions)
    e_right, f_right = _cm.energy_forces(one_subset(right), positions)
    e_both, f_both = _cm.energy_forces(both, positions)
    assert np.isclose(e_both, e_left + e_right, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f_both[left], f_left[left], atol=1e-6)
    np.testing.assert_allclose(f_both[right], f_right[right], atol=1e-6)
    middle = list(range(10, 30))
    assert np.linalg.norm(f_both[middle]) < 1e-8


def test_charmm_nbfix_matches_python_force(charmm_box, probe_model):
    topology, positions, mm_system, info = charmm_box
    box = topology.getPeriodicBoxVectors()
    peptide = _cm.peptide_atoms(topology)
    print("charmm36", _cm.format_inventory(info))
    reference, native = potentials(probe_model)
    _agree(
        _cm.mixed(reference, topology, mm_system, peptide),
        _cm.mixed(native, topology, mm_system, peptide),
        positions,
        box,
        rtol=1e-5,
        atol=1e-4,
    )


def test_ethanol_smirnoff_propers(probe_model):
    paths = _cm.ethanol_paths()
    if paths is None:
        pytest.skip("ethanol test data not found")
    pdb_path, xml_path = paths
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(xml_path))
    mm_system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    info = _cm.inventory(mm_system, pdb.topology)
    _cm.require_terms(
        info,
        ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce"],
        "ethanol",
    )
    heavy = [atom.index for atom in pdb.topology.atoms() if atom.element.symbol != "H"]
    reference, native = potentials(probe_model)
    _agree(
        _cm.mixed(reference, pdb.topology, mm_system, heavy),
        _cm.mixed(native, pdb.topology, mm_system, heavy),
        pdb.positions,
        None,
    )


def test_barostat_finite_differences_the_energy(amber_box, probe_model):
    topology, positions, mm_system, _, _ = amber_box
    peptide = _cm.peptide_atoms(topology)
    _, native = potentials(probe_model)
    mixed = _cm.mixed(native, topology, mm_system, peptide)
    mixed.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 1))
    integrator = mm.VerletIntegrator(0.0005)
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    context.setVelocitiesToTemperature(300 * unit.kelvin, 3)
    e0 = context.getState(getEnergy=True).getPotentialEnergy()
    integrator.step(3)
    e1 = context.getState(getEnergy=True).getPotentialEnergy()
    del context
    # The barostat may or may not accept a move; either way the energy is finite.
    assert np.isfinite(e0.value_in_unit(unit.kilojoules_per_mole))
    assert np.isfinite(e1.value_in_unit(unit.kilojoules_per_mole))


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v", "-s"]))
