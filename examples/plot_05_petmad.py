"""
PET-MAD from Hugging Face
=========================

`PET-MAD <https://arxiv.org/abs/2503.14118>`_ is a universal potential trained
on the MAD dataset. Checkpoints live on the Hub
(`lab-cosmo/upet <https://huggingface.co/lab-cosmo/upet>`_); the file OpenMM
loads is an exported TorchScript ``.pt``.

This example does both Hub steps a user actually runs:

1. ``hf download lab-cosmo/upet models/pet-mad-xs-v1.5.0.ckpt``
2. ``mtt export`` that checkpoint to TorchScript
3. ``upet.save_upet`` — the packaged Hugging Face → ``.pt`` path

then evaluates water (and toluene, if the OpenMM-ML test PDB is present)
through ``MLPotential("metatomic")``, the current OpenMM end-user API.
CPU and CUDA are compared when a GPU is visible. A short NVE run checks
the ``PythonForce`` survives repeated ``getState`` calls, not just one
energy.
"""

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch

from _openmm import (
    energy_forces,
    make_potential,
    openmm_ml_data,
    preferred_platforms,
    vacuum_topology,
)
from _petmad import (
    WATER_NM,
    WATER_NUMBERS,
    WATER_SYMBOLS,
    evaluate_exported,
    ensure_models,
)

ckpt, converted, official = ensure_models()
print(f"checkpoint  {ckpt}")
print(f"converted   {converted}  ({converted.stat().st_size / 1e6:.1f} MB)")
print(f"official pt {official}  ({official.stat().st_size / 1e6:.1f} MB)")

pos_ang = WATER_NM * 10.0
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch devices: {devices}")

rows = []
for label, path in (("converted", converted), ("official", official)):
    energy, forces = evaluate_exported(str(path), WATER_NUMBERS, pos_ang, device="cpu")
    rows.append((label, "direct/cpu", energy, float(np.linalg.norm(forces))))
    print(f"{label:10} direct/cpu  E={energy:12.6f} kJ/mol  |F|={np.linalg.norm(forces):.4f}")

dE = abs(rows[0][2] - rows[1][2])
print(f"ΔE converted vs official pt: {dE:.4e} kJ/mol")
assert np.isfinite(rows[0][2]) and np.isfinite(rows[1][2])

topology = vacuum_topology(WATER_SYMBOLS)
platform = preferred_platforms()[0]
print(f"OpenMM platform {platform.getName()}")

openmm_rows = []
for device in devices:
    potential = make_potential(str(converted), device=device)
    system = potential.createSystem(topology)
    integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(WATER_NM * unit.nanometers)
    energy, forces = energy_forces(context)
    openmm_rows.append((device, energy, forces))
    print(f"OpenMM-ML/{device:<4}  E={energy:12.6f} kJ/mol  |F|={np.linalg.norm(forces):.4f}")
    assert np.isclose(energy, rows[0][2], rtol=1e-4, atol=5e-2)

if len(openmm_rows) == 2:
    print(
        f"ΔE cpu vs cuda: {abs(openmm_rows[0][1] - openmm_rows[1][1]):.4e} kJ/mol"
    )

# Short NVE on the CPU OpenMM path — same pattern as production ML MD.
potential = make_potential(str(converted), device="cpu")
system = potential.createSystem(topology)
integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(WATER_NM * unit.nanometers)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 0)
totals = []
for _ in range(40):
    state = simulation.context.getState(getEnergy=True)
    totals.append(
        (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(
            unit.kilojoules_per_mole
        )
    )
    integrator.step(1)
totals = np.asarray(totals)
print(
    f"NVE 40 x 0.5 fs  mean={totals.mean():.4f}  std={totals.std():.4f}  "
    f"drift={totals[-1] - totals[0]:.4f} kJ/mol"
)
assert np.all(np.isfinite(totals))

# Toluene vacuum + ML/MM (ligand internals PET-MAD, solvent Amber) when test data exists.
data = openmm_ml_data()
if data is not None:
    pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    pos = pdb.getPositions(asNumpy=True)
    pot = make_potential(str(converted), device="cpu")
    vacuum = pot.createSystem(pdb.topology)
    ctx = mm.Context(vacuum, mm.VerletIntegrator(0.0005), platform)
    ctx.setPositions(pos)
    e_vac, _ = energy_forces(ctx)
    print(f"toluene vacuum PET-MAD  N={len(numbers)}  E={e_vac:.6f} kJ/mol")

    prm = data / "toluene" / "toluene-explicit.prm7"
    rst = data / "toluene" / "toluene-explicit.rst7"
    if prm.is_file():
        prmtop = app.AmberPrmtopFile(str(prm))
        inpcrd = app.AmberInpcrdFile(str(rst))
        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
        mixed = pot.createMixedSystem(
            prmtop.topology, mm_system, list(range(15)), interpolate=False
        )
        ctx_m = mm.Context(mixed, mm.VerletIntegrator(0.0005), platform)
        ctx_m.setPositions(inpcrd.positions)
        e_mix, _ = energy_forces(ctx_m)
        print(
            f"toluene ML/MM PET-MAD  N={prmtop.topology.getNumAtoms()}  "
            f"ML=15  E={e_mix:.6f} kJ/mol"
        )

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
axes[0].bar(
    [r[0] + "\n" + r[1] for r in rows] + [f"OpenMM\n{d}" for d, _, _ in openmm_rows],
    [r[2] for r in rows] + [e for _, e, _ in openmm_rows],
    color=["C0", "C0"] + ["C1"] * len(openmm_rows),
)
axes[0].set_ylabel("E / kJ mol$^{-1}$")
axes[0].set_title("Water, PET-MAD-XS v1.5.0")
axes[0].tick_params(axis="x", rotation=15)
axes[1].plot(np.arange(len(totals)) * 0.5, totals, color="C2")
axes[1].set_xlabel("t / fs")
axes[1].set_ylabel("E$_\\mathrm{tot}$ / kJ mol$^{-1}$")
axes[1].set_title("40-step NVE")
fig.tight_layout()
