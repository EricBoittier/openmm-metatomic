"""
Native MetatomicForce vs PythonForce, head to head
==================================================

``MLPotential("metatomic")`` evaluates the model from Python through
``openmm.PythonForce``; ``MLPotential("metatomic-native")`` hands the same model
to the C++ plugin. Both build the same Systems through OpenMM-ML, so this
compares only the evaluation path.

Two systems, both PET-MAD-XS: toluene in vacuum (pure ML, 15 atoms) and toluene
in explicit water through ``createMixedSystem`` (15 ML atoms of 6,495, so the
native force exercises its subset gather/scatter).

Timing follows the same recipe as the other benchmarks here: one hot Context per
case, warmup evaluations and warmup MD steps, the median of repeated
``getState`` calls, and ``Integrator.step(N)`` timed as a block with no per-step
query. Two additions, because a 15-atom PET-MAD evaluation is short enough that
the machine, not the code, decides the number:

* torch runs on a fixed, small thread count
  (``OPENMM_METATOMIC_BENCH_THREADS``, default 4). Left at the default 24, the
  threads spend most of a small evaluation spin-waiting, and a single busy core
  elsewhere on the box moves the median by a factor of five.
* the two backends are measured round-robin on live Contexts rather than one
  after the other, so a machine that gets busier partway through the run
  penalises both equally.

Run the script three times and take the median of the three launches before
quoting a number.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch

from _openmm import (
    energy_forces,
    make_native_potential,
    make_potential,
    mixed_system,
    openmm_ml_data,
    preferred_platforms,
    save_figure,
)
from _petmad import ensure_model

EVAL_ROUNDS = 7
EVALS_PER_ROUND = 3
WARMUP_EVALS = 5
WARMUP_STEPS = 3
MD_STEPS = 10
MD_ROUNDS = 3
ML_ATOMS = list(range(15))
THREADS = int(os.environ.get("OPENMM_METATOMIC_BENCH_THREADS", "4"))

torch.set_num_threads(THREADS)

data = openmm_ml_data()
if data is None:
    raise SystemExit("openmm-ml test data not found; set OPENMM_ML_TEST_DATA")
model = str(ensure_model())
platform = preferred_platforms()[0]
print(f"platform {platform.getName()}  model {model}  torch threads {THREADS}")


def potentials(**kwargs):
    return {
        "PythonForce": make_potential(model, device="cpu", **kwargs),
        "native": make_native_potential(model, device="cpu", **kwargs),
    }


def hot_context(system, positions, box_vectors):
    """A Context with its integrator, warmed up and ready to be timed."""
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.0005 * unit.picoseconds
    )
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    for _ in range(WARMUP_EVALS):
        context.getState(getEnergy=True, getForces=True)
    return context, integrator


def compare(label, systems, positions, box_vectors=None):
    live = {
        name: hot_context(system, positions, box_vectors)
        for name, system in systems.items()
    }
    rows = {}
    for name, (context, _) in live.items():
        energy, forces = energy_forces(context)
        rows[name] = dict(energy=energy, forces=forces, eval=[], step=[])

    for _ in range(EVAL_ROUNDS):
        for name, (context, _) in live.items():
            start = time.perf_counter()
            for _ in range(EVALS_PER_ROUND):
                context.getState(getEnergy=True, getForces=True)
            rows[name]["eval"].append(
                (time.perf_counter() - start) * 1e3 / EVALS_PER_ROUND
            )

    for name, (context, integrator) in live.items():
        context.setVelocitiesToTemperature(300 * unit.kelvin, 7)
        integrator.step(WARMUP_STEPS)
    for _ in range(MD_ROUNDS):
        for name, (context, integrator) in live.items():
            start = time.perf_counter()
            integrator.step(MD_STEPS)
            context.getState(getEnergy=True)
            rows[name]["step"].append((time.perf_counter() - start) * 1e3 / MD_STEPS)

    for name, row in rows.items():
        row["eval_ms"] = float(np.median(row["eval"]))
        row["step_ms"] = float(np.median(row["step"]))
        print(
            f"{label:<22} {name:<16} PE {row['energy']:16.6f}  "
            f"eval {row['eval_ms']:8.3f} ms (min {min(row['eval']):8.3f})  "
            f"{row['step_ms']:8.3f} ms/step"
        )
    live.clear()

    ref, other = rows["PythonForce"], rows["native"]
    d = np.linalg.norm(other["forces"] - ref["forces"], axis=1)
    print(
        f"{label:<22} {'agreement':<16} ΔPE {other['energy'] - ref['energy']:.3e} kJ/mol  "
        f"max|ΔF| {d.max():.3e}  RMS {np.sqrt((d ** 2).mean()):.3e} kJ/mol/nm  "
        f"(speedup {ref['eval_ms'] / other['eval_ms']:.2f}x)"
    )
    return rows


cases = {}

pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
pure = potentials()
cases["toluene, pure ML"] = compare(
    "toluene, pure ML",
    {name: p.createSystem(pdb.topology) for name, p in pure.items()},
    pdb.positions,
)

prmtop = app.AmberPrmtopFile(str(data / "toluene" / "toluene-explicit.prm7"))
inpcrd = app.AmberInpcrdFile(str(data / "toluene" / "toluene-explicit.rst7"))
mixed = potentials()
cases["toluene in water, mixed"] = compare(
    "toluene in water, mixed",
    {
        name: mixed_system(
            p,
            prmtop.topology,
            prmtop.createSystem(nonbondedMethod=app.PME),
            ML_ATOMS,
            interpolate=False,
        )
        for name, p in mixed.items()
    },
    inpcrd.positions,
    inpcrd.boxVectors,
)

# The direct force head skips the backward pass; PET-MAD-XS publishes one, so the
# same comparison shows what that costs on either side.
nc = potentials(non_conservative="forces")
cases["toluene, non-conservative"] = compare(
    "toluene, non-conservative",
    {name: p.createSystem(pdb.topology) for name, p in nc.items()},
    pdb.positions,
)

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
labels = list(cases)
x = np.arange(len(labels))
width = 0.36
for offset, name, color in ((-width / 2, "PythonForce", "C0"), (width / 2, "native", "C1")):
    axes[0].bar(
        x + offset,
        [cases[label][name]["eval_ms"] for label in labels],
        width,
        color=color,
        label=name,
    )
    axes[1].bar(
        x + offset,
        [cases[label][name]["step_ms"] for label in labels],
        width,
        color=color,
        label=name,
    )
for ax, title, ylabel in (
    (axes[0], "energy + forces", "median eval / ms"),
    (axes[1], f"Langevin, {MD_STEPS} steps", "ms / step"),
):
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(", ", "\n") for label in labels], fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

speedup = [
    cases[label]["PythonForce"]["eval_ms"] / cases[label]["native"]["eval_ms"]
    for label in labels
]
axes[2].bar(x, speedup, 0.5, color="C2")
axes[2].axhline(1.0, color="k", lw=0.8, ls="--")
axes[2].set_xticks(x)
axes[2].set_xticklabels([label.replace(", ", "\n") for label in labels], fontsize=8)
axes[2].set_title("native speedup on eval")
axes[2].set_ylabel(r"$t_\mathrm{PythonForce} / t_\mathrm{native}$")
for xi, value in zip(x, speedup):
    axes[2].text(xi, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
save_figure(fig, __file__)
