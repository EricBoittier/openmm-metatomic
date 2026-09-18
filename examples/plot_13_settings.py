"""
Settings
========

Wall-clock cost of ``MetatomicForce`` across backend, device, neighbor
list (vesin vs the O(N²) fallback), and eval vs short MD. Systems are
large enough that a cold Context is not the number being reported:

* harmonic well at 3k / 15k / 60k atoms
* harmonic-nl water boxes at 256 / 1,000 / 5,000 molecules
* PET-MAD-XS periodic boxes at 32 and 96 waters

Timing method (C++ ``openmm-metatomic-bench-settings``): one Context,
warmup evals + warmup steps, median ``getState``, then
``Integrator.step(N)`` with no per-step energy query. Set
``OPENMM_METATOMIC_BENCH_LAUNCHES=3`` for a median across process
launches.
"""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _native import (
    ensure_harmonic_pt,
    make_system,
    petmad_path,
    save_figure,
    settings_binary,
    soap_pt,
    water_box,
)
from _openmm import format_ms, preferred_platforms, repo_root, timed


def parse_cpp_table(text: str):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[0] == "case" or "skipped" in line:
            continue
        try:
            label = parts[0]
            n = int(parts[1])
            eval_ms = float(parts[2])
            nve_ms = float(parts[3])
            nvt_ms = float(parts[4])
            drift = float(parts[5])
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


def median_launches(tables):
    by_label = {}
    for rows in tables:
        for row in rows:
            by_label.setdefault(row["label"], []).append(row)
    out = []
    for label, group in by_label.items():
        out.append(
            {
                "label": label,
                "n": group[0]["n"],
                "eval": float(np.median([r["eval"] for r in group])),
                "nve": float(np.median([r["nve"] for r in group])),
                "nvt": float(np.median([r["nvt"] for r in group])),
                "drift": float(np.median([r["drift"] for r in group])),
            }
        )
    return out


def time_python_matrix():
    import openmm as mm
    import openmm.unit as unit

    records = []
    platforms = preferred_platforms()[:1]
    cloud_n = 3000
    rng = np.random.default_rng(1)
    types = np.array([1, 6, 8] * ((cloud_n + 2) // 3), dtype=np.int32)[:cloud_n]
    pos = rng.uniform(-0.3, 0.3, size=(cloud_n, 3))
    box_types, box_pos, box = water_box(256)
    jobs = [
        ("harmonic", "harmonic", "core", False, types, pos, None),
        ("harmonic-nl", "harmonic-nl", "core", True, box_types, box_pos, box),
    ]
    pet = petmad_path()
    p_types, p_pos, p_box = water_box(32)
    jobs.append(("PET-MAD-XS", str(pet), "torch", True, p_types, p_pos, p_box))

    for title, path, backend, periodic, xyz_types, xyz, cell in jobs:
        for plat in platforms:
            system = make_system(
                xyz_types,
                path,
                backend=backend,
                periodic=periodic,
                box_nm=cell,
            )
            integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
            context = mm.Context(system, integrator, plat)
            context.setPositions(xyz * unit.nanometers)
            if cell is not None:
                context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in cell))

            def energy():
                return context.getState(getEnergy=True, getForces=True)

            for _ in range(4):
                energy()
            integrator.step(4)
            _, t_eval = timed(energy, repeat=7, warmup=2)

            def md():
                integrator.step(8)

            _, t_md = timed(md, repeat=1, warmup=0)
            label = f"{title}/{backend}/cpu/{plat.getName()}/nocon/vesin/N{len(xyz)}"
            records.append(
                {
                    "label": label,
                    "n": len(xyz),
                    "eval": float(np.median(t_eval) * 1e3),
                    "nve": float(t_md[0] / 8.0 * 1e3),
                    "nvt": np.nan,
                    "drift": np.nan,
                }
            )
            print(f"{label:<56} eval {format_ms(t_eval)}")
    return records


def bench_command():
    root = repo_root()
    binary = settings_binary()
    sizes = [3000, 15000, 60000]
    for n in sizes:
        ensure_harmonic_pt(n)
    cmd = [
        str(binary),
        "--models",
        "harmonic,harmonic-nl,soap-bpnn,pet-mad-box",
        "--backends",
        "core,torch",
        "--platforms",
        "Reference",
        "--nl",
        "vesin,naive",
        "--consistency",
        "0",
        "--nve",
        "20",
        "--nvt",
        "20",
        "--eval",
        "11",
        "--warmup-eval",
        "8",
        "--warmup-step",
        "10",
        "--harmonic-atoms",
        "3000,15000,60000",
        "--box-mol",
        "256,1000,5000",
        "--pet-mol",
        "32,96",
        "--naive-max-n",
        "800",
        "--torch-harmonic-dir",
        str(root / "build" / "scaling"),
    ]
    import torch

    if torch.cuda.is_available():
        cmd += ["--devices", "cpu,cuda"]
    pairlist = root / "build" / "pairlist.pt"
    soap = soap_pt()
    if pairlist.is_file():
        cmd += ["--torch-nl", str(pairlist)]
    if soap is not None:
        cmd += ["--soap", str(soap)]
    cmd += ["--petmad", str(petmad_path())]
    return cmd


def load_saved_tables(path: Path):
    parsed = parse_cpp_table(path.read_text())
    return [parsed] if parsed else []


rows = []
binary = settings_binary()
launches = max(1, int(os.environ.get("OPENMM_METATOMIC_BENCH_LAUNCHES", "1")))
try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path(os.environ.get("OPENMM_METATOMIC_ROOT", ".")).resolve() / "examples"
replay = _here / "plots" / "bench_settings.txt"
if os.environ.get("OPENMM_METATOMIC_BENCH_REPLAY") and replay.is_file():
    tables = load_saved_tables(replay)
    rows = median_launches(tables) if tables else []
    print(f"replaying {len(tables)} launches from {replay}")
elif binary is not None:
    cmd = bench_command()
    print(" ".join(cmd))
    tables = []
    stdout_all = []
    for launch in range(launches):
        print(f"launch {launch + 1}/{launches}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        stdout_all.append(result.stdout)
        if result.returncode != 0:
            print(result.stderr[-2000:])
        parsed = parse_cpp_table(result.stdout)
        if parsed:
            tables.append(parsed)
    if tables:
        rows = median_launches(tables) if launches > 1 else tables[0]
        out = _here / "plots"
        out.mkdir(exist_ok=True)
        (out / "bench_settings.txt").write_text("\n\n".join(stdout_all))
if not rows:
    print("C++ bench-settings missing or empty; timing the Python wrapper")
    rows = time_python_matrix()

harmonic = [r for r in rows if r["label"].startswith("harmonic/")]
other = [r for r in rows if not r["label"].startswith("harmonic/")]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
styles = {
    "core": dict(marker="o", color="C0"),
    "torch": dict(marker="s", color="C1"),
}
for backend, style in styles.items():
    subset = [r for r in harmonic if f"/{backend}/" in r["label"]]
    subset = sorted(subset, key=lambda r: r["n"])
    if not subset:
        continue
    ns = [r["n"] for r in subset]
    axes[0].plot(ns, [r["eval"] for r in subset], label=f"{backend} eval", **style)
    axes[0].plot(
        ns,
        [r["nve"] for r in subset],
        label=f"{backend} NVE",
        linestyle="--",
        **style,
    )
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("N atoms")
axes[0].set_ylabel("ms")
axes[0].set_title("Harmonic well, hot Context")
axes[0].legend(fontsize=8)

if other:
    def short_label(label: str) -> str:
        parts = label.split("/")
        model = (
            parts[0]
            .replace("harmonic-nl", "nl")
            .replace("pet-mad-box", "pet")
            .replace("soap-bpnn", "soap")
        )
        backend = parts[1] if len(parts) > 1 else ""
        device = parts[2] if len(parts) > 2 else ""
        nl = parts[5] if len(parts) > 5 else ""
        n = parts[6] if len(parts) > 6 else ""
        who = "cuda" if device == "cuda" else backend
        return f"{model}/{who}/{nl}/{n}"

    labels = [short_label(r["label"]) for r in other]
    x = np.arange(len(other))
    axes[1].bar(x - 0.2, [r["eval"] for r in other], 0.4, label="eval")
    axes[1].bar(x + 0.2, [r["nve"] for r in other], 0.4, label="NVE")
    axes[1].set_xticks(x, labels, fontsize=6)
    axes[1].tick_params(axis="x", rotation=75)
    axes[1].set_ylabel("ms")
    axes[1].set_yscale("log")
    axes[1].set_title("Periodic boxes (log scale)")
    axes[1].legend(fontsize=8)
else:
    axes[1].set_visible(False)

fig.tight_layout()
save_figure(fig, _here / "plot_13_settings.py")

print(f"{'case':<56} {'N':>8} {'eval':>10} {'nve':>10} {'nvt':>10} {'drift':>12}")
for row in rows:
    print(
        f"{row['label']:<56} {row['n']:8d} {row['eval']:10.3f} "
        f"{row['nve']:10.3f} {row['nvt']:10.3f} {row['drift']:12.4g}"
    )
