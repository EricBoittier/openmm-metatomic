"""
Pure ML through OpenMM
======================

The current end-user path is OpenMM-ML, not the native plugin::

    potential = MLPotential("metatomic", modelPath="model.pt", device="cpu")
    system = potential.createSystem(topology)
    simulation = app.Simulation(topology, system, integrator, platform)

This example exports the same independent-atom well used by OpenMM-ML's
metatomic tests (Å, eV) and runs vacuum water, methane, and toluene as
**pure ML** systems: ``createSystem``, energy/forces vs the TorchScript
model, local minimization, and a short Langevin trajectory.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from _harmonic import SYSTEMS
from _openmm import (
    direct_energy_forces,
    energy_forces,
    export_openmm_model,
    make_potential,
    openmm_ml_data,
    preferred_platforms,
    symbols_from_numbers,
    vacuum_topology,
)

NM_TO_ANG = 10.0


def system_spec(name: str):
    spec = SYSTEMS[name]
    rest = np.asarray(spec["rest"], dtype=np.float64) * NM_TO_ANG
    pos = np.asarray(spec["positions"], dtype=np.float64) * NM_TO_ANG
    numbers = spec["types"]
    return spec["title"], numbers, rest, pos, vacuum_topology(symbols_from_numbers(numbers))


def load_toluene():
    data = openmm_ml_data()
    if data is None:
        return None
    pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return "toluene (vacuum PDB)", numbers, pos, pos + 0.1, pdb.topology


jobs = [system_spec("water"), system_spec("methane")]
toluene = load_toluene()
if toluene is not None:
    jobs.append(toluene)

platform = preferred_platforms()[0]
rows = []

print(f"OpenMM {mm.Platform.getPlatformByName(platform.getName()).getName()}  "
      f"platforms={[p.getName() for p in preferred_platforms()]}")

fig, axes = plt.subplots(1, len(jobs), figsize=(4.0 * len(jobs), 3.4), squeeze=False)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for ax, job in zip(axes[0], jobs):
        title, numbers, rest, pos, topology = job
        model = export_openmm_model(str(tmp / f"{title.split()[0]}.pt"), rest, numbers)
        potential = make_potential(model, device="cpu")
        system = potential.createSystem(topology)
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
        )
        simulation = app.Simulation(topology, system, integrator, platform)
        simulation.context.setPositions(pos * unit.angstrom)

        energy, forces = energy_forces(simulation.context)
        energy_ref, forces_ref = direct_energy_forces(model, numbers, pos)
        dE = abs(energy - energy_ref)
        dF = float(np.max(np.abs(forces - forces_ref)))
        assert np.isclose(energy, energy_ref, rtol=1e-5, atol=1e-6)
        assert dF < 1e-4

        simulation.minimizeEnergy(maxIterations=25)
        e_min, _ = energy_forces(simulation.context)
        pe = []
        for _ in range(40):
            simulation.step(1)
            pe.append(
                simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )
        python_forces = [f for f in system.getForces() if isinstance(f, mm.PythonForce)]
        rows.append((title, len(numbers), energy, e_min, dE, dF, len(python_forces)))
        print(
            f"{title:<28} N={len(numbers):3d}  E={energy:12.6e}  "
            f"E_min={e_min:12.6e}  ΔE={dE:.2e}  ΔF={dF:.2e}  PythonForce={len(python_forces)}"
        )
        ax.plot(pe, color="C0")
        ax.set_title(title)
        ax.set_xlabel("Langevin step (1 fs)")
        ax.set_ylabel("E / kJ mol$^{-1}$")

print("\nenergy matches load_atomistic_model; minimization + 40 fs NVT on each vacuum system")
fig.suptitle(f"Pure ML Langevin, {platform.getName()} platform")
fig.tight_layout()
