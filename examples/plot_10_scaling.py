"""
End-to-end timings vs. system size
===================================

The other timing gallery (:doc:`plot_08_timings`) fixes a handful of small
molecules. This one sweeps the atom count, on the current end-user path
(``MLPotential("metatomic")`` -> ``PythonForce``), to see how ``createSystem``,
``Context`` construction, and per-step cost actually scale.

Two TorchScript models are exported from the same independent-atom harmonic
well: one with no neighbor list (pure elementwise cost, O(N) by
construction) and one that requests a 4 Angstrom ``vesin`` neighbor list, so
the pairlist-construction cost is visible too. Water molecules are tiled on a
cubic lattice wide enough to keep the neighbor list sparse.
"""

import gc
import os
import sys
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
    export_openmm_model,
    format_ms,
    make_potential,
    preferred_platforms,
    timed,
    vacuum_topology,
)

NM_TO_ANG = 10.0
WATER_REST = np.asarray(SYSTEMS["water"]["rest"], dtype=np.float64) * NM_TO_ANG
WATER_TYPES = SYSTEMS["water"]["types"]
SYMBOLS = {1: "H", 8: "O"}
SPACING = 6.0  # angstrom; bigger than the 4 A cutoff keeps the neighbor list sparse
N_MOLECULES = [1, 10, 100, 1000, 5000, 20000]
if "sphinx_gallery" in sys.modules or os.environ.get("SPHINX_GALLERY_RUNNING"):
    N_MOLECULES = [1, 10, 100]


def water_cluster(n_molecules: int, spacing: float = SPACING, seed: int = 0):
    n_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    offsets = [
        (ix, iy, iz)
        for ix in range(n_side)
        for iy in range(n_side)
        for iz in range(n_side)
    ][:n_molecules]
    rest = np.concatenate(
        [WATER_REST + np.array(offset) * spacing for offset in offsets], axis=0
    )
    types = list(WATER_TYPES) * n_molecules
    rng = np.random.default_rng(seed)
    positions = rest + rng.normal(scale=0.01, size=rest.shape)
    symbols = [SYMBOLS[t] for t in types]
    return types, symbols, rest, positions


def time_case(topology, positions, model_path, platform, n_eval, n_md):
    potential = make_potential(model_path, device="cpu")
    _, t_sys = timed(lambda: potential.createSystem(topology), repeat=1)
    system = potential.createSystem(topology)
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds
    )
    ctx, t_ctx = timed(
        lambda: mm.Context(system, mm.VerletIntegrator(0.001), platform), repeat=1
    )
    del ctx
    gc.collect()
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    def energy():
        return simulation.context.getState(getEnergy=True, getForces=True)

    _, t_first = timed(energy, repeat=1)
    _, t_eval = timed(energy, repeat=n_eval, warmup=1)
    if n_md > 0:
        _, t_md = timed(lambda: simulation.step(n_md), repeat=1)
        md_ms = float(np.median(t_md) * 1e3) / n_md
    else:
        md_ms = float("nan")
    return {
        "create_ms": float(np.median(t_sys) * 1e3),
        "context_ms": float(np.median(t_ctx) * 1e3),
        "first_ms": float(np.median(t_first) * 1e3),
        "eval_ms": float(np.median(t_eval) * 1e3),
        "md_ms_per_step": md_ms,
    }


records = []
platform = preferred_platforms()[0]
print(f"platform: {platform.getName()}\n")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for i, n_mol in enumerate(N_MOLECULES):
        types, symbols, rest, positions = water_cluster(n_mol)
        n_atoms = len(types)
        topology = vacuum_topology(symbols)
        n_eval = 10 if n_atoms <= 3000 else 5
        n_md = 10 if n_atoms <= 3000 else 0

        plain_path = export_openmm_model(
            str(tmp / f"plain-{i}.pt"), rest, types, neighbor_list=False
        )
        plain = time_case(
            topology, positions * unit.angstrom, plain_path, platform, n_eval, n_md
        )

        nl_path = export_openmm_model(
            str(tmp / f"nl-{i}.pt"), rest, types, neighbor_list=True
        )
        nl = time_case(
            topology, positions * unit.angstrom, nl_path, platform, n_eval, n_md
        )

        _, t_direct = timed(
            lambda p=plain_path: direct_energy_forces(p, types, positions, device="cpu"),
            repeat=n_eval,
            warmup=1,
        )
        direct_ms = float(np.median(t_direct) * 1e3)

        records.append(
            {"n_atoms": n_atoms, "plain": plain, "nl": nl, "direct_ms": direct_ms}
        )
        print(
            f"N={n_atoms:>6}  createSystem {plain['create_ms']:9.2f} ms  "
            f"Context {plain['context_ms']:7.2f} ms  "
            f"eval (no NL) {plain['eval_ms']:8.3f} ms  "
            f"eval (vesin NL) {nl['eval_ms']:8.3f} ms  "
            f"direct torch {direct_ms:8.3f} ms  "
            f"MD/step {plain['md_ms_per_step']:8.3f} ms"
        )

n_atoms = np.array([r["n_atoms"] for r in records], dtype=np.float64)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].loglog(n_atoms, [r["plain"]["eval_ms"] for r in records], "o-", label="PythonForce, no NL")
axes[0].loglog(n_atoms, [r["nl"]["eval_ms"] for r in records], "s-", label="PythonForce, vesin NL")
axes[0].loglog(n_atoms, [r["direct_ms"] for r in records], "^-", label="direct TorchScript")
axes[0].set_xlabel("atoms")
axes[0].set_ylabel("median energy+force / ms")
axes[0].set_title("Per-call evaluation vs. N")
axes[0].legend(fontsize=8)

axes[1].loglog(n_atoms, [r["plain"]["create_ms"] for r in records], "o-", label="createSystem")
axes[1].loglog(n_atoms, [r["plain"]["context_ms"] for r in records], "s-", label="Context")
axes[1].set_xlabel("atoms")
axes[1].set_ylabel("ms")
axes[1].set_title("Setup cost vs. N")
axes[1].legend(fontsize=8)
fig.tight_layout()
