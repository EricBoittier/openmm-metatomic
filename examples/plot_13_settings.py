"""
Settings
========

Wall-clock cost of ``MetatomicForce`` across the knobs the public object
actually exposes: backend, device, ``checkConsistency``, neighbor list
(vesin vs the O(N²) fallback), OpenMM platform, and eval vs short MD.

When ``openmm-metatomic-bench-settings`` is in ``build/``, this example
runs it and plots the table. Otherwise it times the same matrix through
the Python wrapper.
"""

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _native import (
    make_system,
    petmad_path,
    settings_binary,
    soap_pt,
    water_box,
)
from _openmm import format_ms, preferred_platforms, timed
from _petmad import WATER_NM, WATER_NUMBERS


def parse_cpp_table(text: str):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0] == "case" or parts[0].endswith("skipped"):
            continue
        try:
            n = int(parts[-4])
            eval_ms = float(parts[-3])
            nve_ms = float(parts[-2]) if parts[-2] != "skipped" else np.nan
            # last columns: eval nve nvt drift — parser uses the emit() layout
        except ValueError:
            continue
        label = parts[0]
        try:
            n = int(parts[1])
            eval_ms = float(parts[2])
            nve_ms = float(parts[3])
            nvt_ms = float(parts[4])
            drift = float(parts[5]) if len(parts) > 5 else np.nan
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "label": label,
                "n": n,
                "eval": eval_ms,
                "nve": nve_ms,
                "nvt": nvt_ms,
                "drift": drift,
            }
        )
    return rows


def time_python_matrix():
    import openmm as mm
    import openmm.unit as unit

    records = []
    types = WATER_NUMBERS
    pos = WATER_NM
    platforms = preferred_platforms()
    jobs = [("harmonic", "harmonic", "core", False, None)]
    pt = Path(__file__).resolve().parents[1] / "build" / "harmonic.pt"
    if pt.is_file():
        jobs.append(("harmonic", str(pt), "torch", False, None))
    pet = petmad_path()
    jobs.append(("PET-MAD-XS", str(pet), "torch", False, None))
    soap = soap_pt()
    if soap is not None:
        jobs.append(("SOAP-BPNN", str(soap), "torch", False, None))
    box_types, box_pos, box = water_box(8)
    jobs.append(("harmonic-nl", "harmonic-nl", "core", True, box))

    for title, path, backend, periodic, cell in jobs:
        for plat in platforms[:2]:
            for cons in (False, True):
                for device in ("cpu",):
                    system = make_system(
                        box_types if periodic else types,
                        path,
                        backend=backend,
                        device=device,
                        check_consistency=cons,
                        periodic=periodic,
                        box_nm=cell,
                    )
                    integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
                    context = mm.Context(system, integrator, plat)
                    xyz = box_pos if periodic else pos
                    context.setPositions(xyz * unit.nanometers)
                    if cell is not None:
                        context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in cell))

                    def energy():
                        return context.getState(getEnergy=True, getForces=True)

                    energy()
                    _, t_eval = timed(energy, repeat=8, warmup=2)
                    label = (
                        f"{title}/{backend}/{device}/{plat.getName()}"
                        f"/{'cons' if cons else 'nocon'}"
                    )
                    records.append(
                        {
                            "label": label,
                            "n": len(xyz),
                            "eval": float(np.median(t_eval) * 1e3),
                            "nve": np.nan,
                            "nvt": np.nan,
                            "drift": np.nan,
                        }
                    )
                    print(f"{label:<52} eval {format_ms(t_eval)}")
    return records


rows = []
binary = settings_binary()
root = Path(__file__).resolve().parents[1]
if binary is not None:
    cmd = [
        str(binary),
        "--models",
        "harmonic,harmonic-nl,pet-mad",
        "--backends",
        "core,torch",
        "--platforms",
        "Reference",
        "--nl",
        "vesin,naive",
        "--nve",
        "8",
        "--nvt",
        "8",
        "--eval",
        "6",
        "--n-mol",
        "8",
    ]
    harmonic = root / "build" / "harmonic.pt"
    pairlist = root / "build" / "pairlist.pt"
    soap = root / "build" / "soap-bpnn.pt"
    pet = petmad_path()
    toluene = root.parent / "openmm-ml" / "test" / "data" / "toluene" / "toluene.pdb"
    if harmonic.is_file():
        cmd += ["--torch-harmonic", str(harmonic)]
    if pairlist.is_file():
        cmd += ["--torch-nl", str(pairlist)]
    if soap.is_file():
        cmd += ["--soap", str(soap)]
    cmd += ["--petmad", str(pet)]
    if toluene.is_file():
        cmd += ["--toluene", str(toluene)]
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-2000:])
    rows = parse_cpp_table(result.stdout)
if not rows:
    print("C++ bench-settings missing or empty; timing the Python wrapper")
    rows = time_python_matrix()

labels = [r["label"] for r in rows]
x = np.arange(len(rows))
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
axes[0].bar(x, [r["eval"] for r in rows], color="C0")
axes[0].set_xticks(x, labels, fontsize=6)
axes[0].set_ylabel("eval / ms")
axes[0].set_title("Energy+force")
axes[0].tick_params(axis="x", rotation=75)
axes[1].bar(x - 0.2, [r["nve"] for r in rows], 0.4, label="NVE")
axes[1].bar(x + 0.2, [r["nvt"] for r in rows], 0.4, label="NVT")
axes[1].set_xticks(x, labels, fontsize=6)
axes[1].set_ylabel("ms / step")
axes[1].set_title("Short MD")
axes[1].legend(fontsize=8)
axes[1].tick_params(axis="x", rotation=75)
fig.tight_layout()
