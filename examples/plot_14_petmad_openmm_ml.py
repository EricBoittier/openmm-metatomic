"""
OpenMM-ML with PET-MAD-XS
=========================

``MLPotential("metatomic")`` is the OpenMM-ML entry point. The constructor
knobs that matter for PET-MAD-XS::

    potential = MLPotential(
        "metatomic",
        modelPath="pet-mad-xs-v1.5.0.pt",
        device="cpu",            # or "cuda"
        checkConsistency=False,
        non_conservative=False,  # True / "forces" / "stress"
        uncertainty_threshold=0.1,
    )
    system = potential.createSystem(topology)           # vacuum
    system = potential.createSystem(topology, pbc=(1, 1, 1))  # periodic

This example uses the vendored extra-small checkpoint:

* conservative (autograd) vs ``non_conservative="forces"`` on vacuum water
  — NVE drift and a force scatter
* ``energy_uncertainty`` with the default 0.1 eV threshold
* Langevin NVT, then a MonteCarlo barostat on a 32-water box (the energy
  depends on the cell through a strain tensor, so NPT is finite-difference;
  non-conservative stress cannot drive the barostat)
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch

from _openmm import (
    energy_forces,
    make_potential,
    preferred_platforms,
    save_figure,
    vacuum_topology,
)
from _petmad import (
    WATER_NM,
    WATER_SYMBOLS,
    ensure_model,
    water_box_topology,
)

model_path = ensure_model()
platform = preferred_platforms()[0]
print(f"model {model_path.name}  platform {platform.getName()}")
print(f"cuda available: {torch.cuda.is_available()}")

topology = vacuum_topology(WATER_SYMBOLS)
dt = 0.0005 * unit.picoseconds
nve_steps = 40


def nve_trace(non_conservative):
    potential = make_potential(
        str(model_path), device="cpu", non_conservative=non_conservative
    )
    system = potential.createSystem(topology)
    integrator = mm.VerletIntegrator(dt)
    context = mm.Context(system, integrator, platform)
    context.setPositions(WATER_NM * unit.nanometers)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    integrator.step(8)
    totals, pots = [], []
    for _ in range(nve_steps + 1):
        state = context.getState(getEnergy=True)
        pot = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kin = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        pots.append(pot)
        totals.append(pot + kin)
        integrator.step(1)
    return np.asarray(totals), np.asarray(pots)


cons_e, _ = nve_trace(False)
nc_e, _ = nve_trace("forces")
print(
    f"NVE water  conservative drift={cons_e[-1] - cons_e[0]:.4f}  "
    f"non_conservative drift={nc_e[-1] - nc_e[0]:.4f} kJ/mol"
)

cons_pot = make_potential(str(model_path), device="cpu", non_conservative=False)
nc_pot = make_potential(str(model_path), device="cpu", non_conservative="forces")
cons_ctx = mm.Context(
    cons_pot.createSystem(topology), mm.VerletIntegrator(dt), platform
)
nc_ctx = mm.Context(nc_pot.createSystem(topology), mm.VerletIntegrator(dt), platform)
cons_ctx.setPositions(WATER_NM * unit.nanometers)
nc_ctx.setPositions(WATER_NM * unit.nanometers)
_, f_cons = energy_forces(cons_ctx)
_, f_nc = energy_forces(nc_ctx)
rms = float(np.sqrt(np.mean((f_cons - f_nc) ** 2)))
print(f"force RMS conservative vs non_conservative: {rms:.3f} kJ/mol/nm")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    uq = make_potential(
        str(model_path), device="cpu", uncertainty_threshold=0.1
    )
    ctx = mm.Context(uq.createSystem(topology), mm.VerletIntegrator(dt), platform)
    ctx.setPositions(WATER_NM * unit.nanometers)
    energy_forces(ctx)
uq_warns = [w for w in caught if "uncertaint" in str(w.message).lower()]
print(f"uncertainty_threshold=0.1 eV  uncertainty warnings={len(uq_warns)}")

n_mol, spacing = 32, 0.50
box_top, box_pos, box = water_box_topology(n_mol, spacing)
print(
    f"NPT box  molecules={n_mol}  atoms={box_top.getNumAtoms()}  "
    f"L={box[0, 0]:.3f} nm"
)
npt_potential = make_potential(
    str(model_path), device="cpu", uncertainty_threshold=None
)
npt_system = npt_potential.createSystem(box_top, pbc=(True, True, True))
npt_system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 4))
integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, dt
)
simulation = app.Simulation(box_top, npt_system, integrator, platform)
simulation.context.setPositions(box_pos * unit.nanometers)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 2)
volumes = []
for _ in range(16):
    simulation.step(1)
    state = simulation.context.getState()
    volumes.append(state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3))

print(
    f"NPT 16 x 0.5 fs  V0={volumes[0]:.4f}  V1={volumes[-1]:.4f} nm^3  "
    f"(MonteCarloBarostat, conservative energy)"
)

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
t = np.arange(len(cons_e)) * 0.5
axes[0].plot(t, cons_e - cons_e[0], label="autograd")
axes[0].plot(t, nc_e - nc_e[0], label='non_conservative="forces"')
axes[0].set_xlabel("t / fs")
axes[0].set_ylabel(r"$\Delta E_\mathrm{tot}$ / kJ mol$^{-1}$")
axes[0].set_title("NVE, 1 water")
axes[0].legend(fontsize=8)
axes[1].scatter(f_cons.ravel(), f_nc.ravel(), s=18, c="C1")
lims = np.array(axes[1].get_xlim())
axes[1].plot(lims, lims, color="0.5", lw=0.8)
axes[1].set_xlabel("autograd $F$")
axes[1].set_ylabel("non-conservative $F$")
axes[1].set_title(f"forces, RMS {rms:.2f}")
axes[2].plot(np.arange(len(volumes)) * 0.5, volumes, color="C2")
axes[2].set_xlabel("t / fs")
axes[2].set_ylabel(r"$V$ / nm$^3$")
axes[2].set_title(f"NPT, {n_mol} waters")
fig.tight_layout()
save_figure(fig, __file__)
