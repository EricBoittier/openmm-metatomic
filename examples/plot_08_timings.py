"""
End-to-end timings
==================

Wall-clock cost of the steps a user actually pays for:

* export / ``load_atomistic_model``
* ``MLPotential.createSystem`` / ``createMixedSystem`` (includes wrapping the
  model in a ``PythonForce``)
* ``Context`` construction
* first energy+force (cold)
* subsequent ``getState`` calls
* short Langevin MD
* the native-plugin stand-in: C++ ``execute_model`` spike, and a direct
  TorchScript call with no OpenMM

Systems: vacuum water and toluene (pure ML), toluene in explicit water
(ML/MM), and the same vacuum well with a vesin neighbor list. Devices:
OpenMM CPU/Reference, Torch CPU, and Torch CUDA when a GPU is visible.
"""

import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch

from _harmonic import SYSTEMS
from _openmm import (
    direct_energy_forces,
    export_openmm_model,
    format_ms,
    make_potential,
    mixed_system,
    openmm_ml_data,
    preferred_platforms,
    spike_binary,
    timed,
    vacuum_topology,
    symbols_from_numbers,
)
from _petmad import (
    WATER_NM,
    WATER_NUMBERS,
    WATER_SYMBOLS,
    ensure_model,
    evaluate_exported,
)

NM_TO_ANG = 10.0
EVALS = 25
MD_STEPS = 30


def time_openmm(label, topology, positions, model_path, platform, mixed=None, n_eval=None, n_md=None):
    n_eval = EVALS if n_eval is None else n_eval
    n_md = MD_STEPS if n_md is None else n_md
    potential = make_potential(model_path, device="cpu")
    if mixed is None:
        _, t_sys = timed(lambda: potential.createSystem(topology), repeat=1)
        system = potential.createSystem(topology)
    else:
        mm_system, ml_atoms = mixed

        def _mixed():
            return mixed_system(potential, topology, mm_system, ml_atoms)

        _, t_sys = timed(_mixed, repeat=1)
        system = _mixed()

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
    )
    _, t_ctx = timed(
        lambda: mm.Context(system, mm.VerletIntegrator(0.001), platform),
        repeat=1,
    )
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    def energy():
        return simulation.context.getState(getEnergy=True, getForces=True)

    _, t_first = timed(energy, repeat=1)
    _, t_eval = timed(energy, repeat=n_eval, warmup=2)
    try:
        _, t_md = timed(lambda: simulation.step(n_md), repeat=1)
        md_ms = float(np.median(t_md) * 1e3) / n_md
        md_str = format_ms(t_md)
    except Exception as exc:
        t_md = np.array([np.nan])
        md_ms = np.nan
        md_str = f"skipped ({type(exc).__name__})"
    print(
        f"{label:<28} {platform.getName():<10}  "
        f"createSystem {format_ms(t_sys)}  Context {format_ms(t_ctx)}  "
        f"first {format_ms(t_first)}  eval {format_ms(t_eval)}  "
        f"{n_md} MD steps {md_str}"
    )
    return {
        "label": label,
        "platform": platform.getName(),
        "create": float(np.median(t_sys) * 1e3),
        "context": float(np.median(t_ctx) * 1e3),
        "first": float(np.median(t_first) * 1e3),
        "eval": float(np.median(t_eval) * 1e3),
        "md": md_ms,
    }


def time_direct(label, model_path, numbers, positions, device):
    load_atomistic = __import__("metatomic.torch", fromlist=["load_atomistic_model"]).load_atomistic_model
    _, t_load = timed(lambda: load_atomistic(model_path), repeat=1)
    _, t_eval = timed(
        lambda: direct_energy_forces(model_path, numbers, positions, device=device),
        repeat=EVALS,
        warmup=2,
    )
    print(
        f"{label:<28} torch/{device:<6}  load {format_ms(t_load)}  "
        f"eval {format_ms(t_eval)}"
    )
    return {
        "label": f"{label} torch/{device}",
        "platform": f"torch/{device}",
        "create": float(np.median(t_load) * 1e3),
        "context": 0.0,
        "first": float(t_eval[0] * 1e3) if len(t_eval) else 0.0,
        "eval": float(np.median(t_eval) * 1e3),
        "md": np.nan,
    }


records = []
platform = preferred_platforms()[0]
print(f"OpenMM platforms: {[p.getName() for p in preferred_platforms()]}")
print(f"torch cuda: {torch.cuda.is_available()}")
print(f"repeats: {EVALS} energy/force evals, {MD_STEPS} Langevin steps\n")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    water = SYSTEMS["water"]
    w_numbers = water["types"]
    w_rest = np.asarray(water["rest"]) * NM_TO_ANG
    w_pos = np.asarray(water["positions"]) * NM_TO_ANG
    w_top = vacuum_topology(symbols_from_numbers(w_numbers))
    w_model = export_openmm_model(str(tmp / "water.pt"), w_rest, w_numbers)
    w_nl = export_openmm_model(
        str(tmp / "water-nl.pt"), w_rest, w_numbers, neighbor_list=True
    )

    for plat in preferred_platforms():
        records.append(
            time_openmm(
                "water vacuum ML",
                w_top,
                w_pos * unit.angstrom,
                w_model,
                plat,
            )
        )
    records.append(
        time_openmm(
            "water + vesin NL",
            w_top,
            w_pos * unit.angstrom,
            w_nl,
            platform,
        )
    )
    records.append(time_direct("water vacuum", w_model, w_numbers, w_pos, "cpu"))
    if torch.cuda.is_available():
        records.append(time_direct("water vacuum", w_model, w_numbers, w_pos, "cuda"))

    _, t_cons = timed(
        lambda: make_potential(w_model, check_consistency=True).createSystem(w_top),
        repeat=1,
    )
    print(f"{'water createSystem consistency=on':<28} {format_ms(t_cons)}")

    data = openmm_ml_data()
    if data is not None:
        pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
        numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]
        rest = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        t_model = export_openmm_model(str(tmp / "toluene.pt"), rest, numbers)
        displaced = rest + 0.1
        records.append(
            time_openmm(
                "toluene vacuum ML",
                pdb.topology,
                displaced * unit.angstrom,
                t_model,
                platform,
            )
        )
        records.append(time_direct("toluene vacuum", t_model, numbers, displaced, "cpu"))

        prm = data / "toluene" / "toluene-explicit.prm7"
        rst = data / "toluene" / "toluene-explicit.rst7"
        if prm.is_file():
            prmtop = app.AmberPrmtopFile(str(prm))
            inpcrd = app.AmberInpcrdFile(str(rst))
            mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
            records.append(
                time_openmm(
                    f"toluene ML/MM N={prmtop.topology.getNumAtoms()}",
                    prmtop.topology,
                    inpcrd.positions,
                    t_model,
                    platform,
                    mixed=(mm_system, list(range(15))),
                )
            )

# PET-MAD-XS: real Hub model, fewer repeats because each call is a foundation-model eval.
try:
    pet_pt = ensure_model()
except Exception as exc:
    print(f"PET-MAD skipped: {exc}")
    pet_pt = None
if pet_pt is not None and pet_pt.is_file():
    pet_top = vacuum_topology(WATER_SYMBOLS)
    pet_pos = WATER_NM * unit.nanometers
    records.append(
        time_openmm(
            "PET-MAD-XS water",
            pet_top,
            pet_pos,
            str(pet_pt),
            platform,
            n_eval=8,
            n_md=12,
        )
    )
    _, t_direct = timed(
        lambda: evaluate_exported(str(pet_pt), WATER_NUMBERS, WATER_NM * 10.0, device="cpu"),
        repeat=8,
        warmup=1,
    )
    print(f"{'PET-MAD-XS water':<28} torch/cpu   eval {format_ms(t_direct)}")
    records.append(
        {
            "label": "PET-MAD-XS water torch/cpu",
            "platform": "torch/cpu",
            "create": 0.0,
            "context": 0.0,
            "first": float(t_direct[0] * 1e3),
            "eval": float(np.median(t_direct) * 1e3),
            "md": np.nan,
        }
    )
    if torch.cuda.is_available():
        _, t_cuda = timed(
            lambda: evaluate_exported(
                str(pet_pt), WATER_NUMBERS, WATER_NM * 10.0, device="cuda"
            ),
            repeat=8,
            warmup=1,
        )
        print(f"{'PET-MAD-XS water':<28} torch/cuda  eval {format_ms(t_cuda)}")
        records.append(
            {
                "label": "PET-MAD-XS water torch/cuda",
                "platform": "torch/cuda",
                "create": 0.0,
                "context": 0.0,
                "first": float(t_cuda[0] * 1e3),
                "eval": float(np.median(t_cuda) * 1e3),
                "md": np.nan,
            }
        )

spike = spike_binary()
if spike is not None:
    cmd = [str(spike), "--bench", "80"]
    pt = repo_pt = spike.parent / "harmonic.pt"
    if pt.is_file():
        cmd.append(str(pt))
    print(f"\nC++ spike: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = result.stdout + result.stderr
    for line in out.splitlines():
        if line.startswith("bench-") or "backend" in line.lower() and "match" in line.lower():
            print(line)
    if result.returncode != 0:
        print("spike bench failed:\n", out[-2000:])
else:
    print("C++ spike binary not found; skip native execute_model timings")

labels = [f"{r['label']}\n{r['platform']}" for r in records]
x = np.arange(len(records))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].bar(x, [r["eval"] for r in records], color="C0")
axes[0].set_xticks(x, labels, fontsize=7)
axes[0].set_ylabel("median energy+force / ms")
axes[0].set_title("Per-call evaluation")
axes[0].tick_params(axis="x", rotation=25)

width = 0.25
axes[1].bar(x - width, [r["create"] for r in records], width, label="createSystem / load")
axes[1].bar(x, [r["context"] for r in records], width, label="Context")
axes[1].bar(x + width, [r["first"] for r in records], width, label="first eval")
axes[1].set_xticks(x, labels, fontsize=7)
axes[1].set_ylabel("ms")
axes[1].set_title("Setup (once per run)")
axes[1].legend(fontsize=8)
axes[1].tick_params(axis="x", rotation=25)
fig.tight_layout()
