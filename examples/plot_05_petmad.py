"""
PET-MAD from Hugging Face
=========================

`PET-MAD <https://arxiv.org/abs/2503.14118>`_ is a universal potential trained
on the MAD dataset. This repo vendors the extra-small v1.5.0 TorchScript
export (`models/pet-mad-xs-v1.5.0.pt`, ~20 MB) produced from
`lab-cosmo/upet <https://huggingface.co/lab-cosmo/upet>`_. Larger S/M
checkpoints stay on the Hub.

The file OpenMM loads is that ``.pt``. This example runs it through
``MLPotential("metatomic")`` on water (CPU and CUDA), a short NVE trajectory,
and — when the OpenMM-ML test data is present — toluene in vacuum and as the
ML region of toluene in explicit solvent.
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
    ensure_model,
    evaluate_exported,
)

model_path = ensure_model()
print(f"model {model_path}  ({model_path.stat().st_size / 1e6:.1f} MB)")

pos_ang = WATER_NM * 10.0
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print(f"torch devices: {devices}")

energy_ref, forces_ref = evaluate_exported(str(model_path), WATER_NUMBERS, pos_ang, device="cpu")
print(f"direct/cpu  E={energy_ref:12.6f} kJ/mol  |F|={np.linalg.norm(forces_ref):.4f}")
assert np.isfinite(energy_ref)

topology = vacuum_topology(WATER_SYMBOLS)
platform = preferred_platforms()[0]
print(f"OpenMM platform {platform.getName()}")

openmm_rows = []
for device in devices:
    potential = make_potential(str(model_path), device=device)
    system = potential.createSystem(topology)
    integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(WATER_NM * unit.nanometers)
    energy, forces = energy_forces(context)
    openmm_rows.append((device, energy, forces))
    print(f"OpenMM-ML/{device:<4}  E={energy:12.6f} kJ/mol  |F|={np.linalg.norm(forces):.4f}")
    assert np.isclose(energy, energy_ref, rtol=1e-4, atol=5e-2)

if len(openmm_rows) == 2:
    print(f"ΔE cpu vs cuda: {abs(openmm_rows[0][1] - openmm_rows[1][1]):.4e} kJ/mol")

potential = make_potential(str(model_path), device="cpu")
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

data = openmm_ml_data()
if data is not None:
    pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
    numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    pos = pdb.getPositions(asNumpy=True)
    pot = make_potential(str(model_path), device="cpu")
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
labels = ["direct/cpu"] + [f"OpenMM\n{d}" for d, _, _ in openmm_rows]
values = [energy_ref] + [e for _, e, _ in openmm_rows]
axes[0].bar(labels, values, color=["C0"] + ["C1"] * len(openmm_rows))
axes[0].set_ylabel("E / kJ mol$^{-1}$")
axes[0].set_title("Water, PET-MAD-XS v1.5.0")
axes[1].plot(np.arange(len(totals)) * 0.5, totals, color="C2")
axes[1].set_xlabel("t / fs")
axes[1].set_ylabel("E$_\\mathrm{tot}$ / kJ mol$^{-1}$")
axes[1].set_title("40-step NVE")
fig.tight_layout()
