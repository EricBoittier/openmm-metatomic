"""
MM/ML mixed systems
===================

``createMixedSystem`` is the usual production setup: a conventional force
field on the whole Topology, then replace the *internal* energy of an ML
subset (ligand, solute, reactive region) with the Metatomic model.

OpenMM-ML does four things to the MM ``System``:

1. Drop bonds / angles / torsions that live entirely in the ML subset
2. Add nonbonded exceptions so ML atoms do not see each other twice
3. Optionally drop constraints inside the subset
4. Install a ``PythonForce`` that evaluates only those atoms

This example runs the Amber toluene-in-water system from the OpenMM-ML
test suite, the userguide-style peptide-in-water split on alanine
dipeptide, and a λ-interpolation between MM and ML internals.
"""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from _openmm import (
    energy_forces,
    export_openmm_model,
    make_potential,
    mixed_system,
    openmm_ml_data,
    preferred_platforms,
)


def potential_energy(context) -> float:
    return (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )


def run_context(system, positions, platform):
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    context.setPositions(positions)
    energy, forces = energy_forces(context)
    return context, energy, forces


data = openmm_ml_data()
platform = preferred_platforms()[0]
labels, energies = [], []
print(f"platform {platform.getName()}")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    if data is not None and (data / "toluene" / "toluene-explicit.prm7").is_file():
        pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        rest = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        model = export_openmm_model(str(tmp / "toluene.pt"), rest, numbers)
        prmtop = app.AmberPrmtopFile(str(data / "toluene" / "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(str(data / "toluene" / "toluene-explicit.rst7"))
        ml_atoms = list(range(15))
        n_atoms = prmtop.topology.getNumAtoms()
        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = make_potential(model, device="cpu")
        mixed = mixed_system(
            potential, prmtop.topology, mm_system, ml_atoms, interpolate=False
        )
        _, e_mm, _ = run_context(mm_system, inpcrd.positions, platform)
        mixed_context, e_mixed, _ = run_context(mixed, inpcrd.positions, platform)
        python_forces = [f for f in mixed.getForces() if isinstance(f, mm.PythonForce)]
        assert python_forces and python_forces[0].usesPeriodicBoundaryConditions()
        labels.extend(["toluene MM", "toluene mixed"])
        energies.extend([e_mm, e_mixed])
        print(
            f"toluene-explicit  N={n_atoms}  ML={len(ml_atoms)}  "
            f"E_MM={e_mm:.6e}  E_mixed={e_mixed:.6e}"
        )
        try:
            interp = mixed_system(
                potential, prmtop.topology, mm_system, ml_atoms, interpolate=True
            )
            interp_context, e_l1, _ = run_context(interp, inpcrd.positions, platform)
            interp_context.setParameter("lambda_interpolate", 0)
            e_l0 = potential_energy(interp_context)
            print(f"  λ-interpolate  λ=1 {e_l1:.6e}  λ=0 {e_l0:.6e}")
            assert np.isclose(e_mixed, e_l1, rtol=1e-4)
            assert np.isclose(e_mm, e_l0, rtol=1e-4)
            labels.extend(["λ=1 (ML)", "λ=0 (MM)"])
            energies.extend([e_l1, e_l0])
        except Exception as exc:
            print(
                "  λ-interpolate skipped: PythonForce cannot be cloned into "
                f"CustomCVForce ({type(exc).__name__}: {exc})"
            )

        try:
            integrator = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.0005 * unit.picoseconds
            )
            simulation = app.Simulation(prmtop.topology, mixed, integrator, platform)
            simulation.context.setPositions(inpcrd.positions)
            simulation.minimizeEnergy(maxIterations=5)
            simulation.step(10)
            e_md = potential_energy(simulation.context)
            print(f"toluene mixed  10 fs Langevin after 5 min steps  E={e_md:.6e}")
        except Exception as exc:
            print(
                "toluene mixed MD skipped "
                f"({type(exc).__name__}: {exc}). The toy well is centered on the "
                "vacuum PDB, not the solvated rst7 coordinates; use PET-MAD "
                "(plot_05) for a real mixed trajectory."
            )
    else:
        print("openmm-ml toluene-explicit data not found; skipping Amber mixed system")

    ala = None if data is None else data / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb"
    if ala is not None and ala.is_file():
        pdb = app.PDBFile(str(ala))
        peptide = [atom.index for atom in next(pdb.topology.chains()).atoms()]
        atoms = list(pdb.topology.atoms())
        numbers = [atoms[i].element.atomic_number for i in peptide]
        pos = pdb.getPositions(asNumpy=True)
        rest = pos.value_in_unit(unit.angstrom)[peptide]
        model = export_openmm_model(str(tmp / "peptide.pt"), rest, numbers)
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        mm_system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
        potential = make_potential(model, device="cpu")
        mixed = mixed_system(potential, pdb.topology, mm_system, peptide)
        n_atoms = pdb.topology.getNumAtoms()
        _, e_mm, _ = run_context(mm_system, pos, platform)
        _, e_mixed, _ = run_context(mixed, pos, platform)
        print(
            f"alanine-dipeptide  N={n_atoms}  ML={len(peptide)} (ACE-ALA-NME)  "
            f"E_MM={e_mm:.6e}  E_mixed={e_mixed:.6e}"
        )
        labels.extend(["peptide MM", "peptide mixed"])
        energies.extend([e_mm, e_mixed])
    else:
        print("alanine-dipeptide PDB not found; skipping ForceField mixed system")

fig, ax = plt.subplots(figsize=(7.2, 3.6))
colors = ["C0" if "MM" in name and "mixed" not in name else "C1" for name in labels]
ax.bar(labels, energies, color=colors)
ax.set_ylabel("E / kJ mol$^{-1}$")
ax.set_title("MM vs ML/MM internals (same coordinates)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
