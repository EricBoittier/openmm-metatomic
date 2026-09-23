"""NVE PET-MAD-XS water box → extended XYZ.

Matches the CUDA conservation runs: native MetatomicForce, Verlet, 0.5 fs.
Default is 512 waters, 10 ps, a frame every 0.01 ps.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as unit

from _native import make_system, native_available, preferred_platform, water_box
from _openmm import repo_root
from _petmad import ensure_model

NM_TO_ANG = 10.0
Z_TO_SYM = {1: "H", 8: "O"}
R = 8.314462618e-3


def wrap_into_box(pos_nm: np.ndarray, box_nm: float) -> np.ndarray:
    return pos_nm - box_nm * np.floor(pos_nm / box_nm)


def write_frame(fh, symbols, pos_ang, box_ang, comment: str) -> None:
    n = len(symbols)
    lattice = f"{box_ang:.8f} 0 0 0 {box_ang:.8f} 0 0 0 {box_ang:.8f}"
    fh.write(f"{n}\n")
    fh.write(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 {comment}\n')
    for sym, (x, y, z) in zip(symbols, pos_ang):
        fh.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mol", type=int, default=512)
    parser.add_argument("--ps", type=float, default=10.0)
    parser.add_argument("--dt-ps", type=float, default=0.0005)
    parser.add_argument("--stride-ps", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--platform", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not native_available():
        print("native plugin not importable", file=sys.stderr)
        return 1

    for plugin_dir in (
        Path(getattr(mm.version, "openmm_library_path", "") or "") / "plugins",
        repo_root() / "build",
        Path("/home/ericb/metawork/openmm_meta/install/lib/plugins"),
    ):
        if plugin_dir.is_dir():
            mm.Platform.loadPluginsFromDirectory(str(plugin_dir))

    types, positions, box = water_box(args.n_mol)
    box_nm = float(box[0, 0])
    symbols = [Z_TO_SYM[int(z)] for z in types]
    n_steps = int(round(args.ps / args.dt_ps))
    stride = max(1, int(round(args.stride_ps / args.dt_ps)))
    model = ensure_model()
    out = Path(
        args.out
        or (
            repo_root()
            / "examples"
            / "plots"
            / f"petmad_xs_n{args.n_mol}_nve_{args.ps:g}ps.xyz"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".csv")

    system = make_system(
        types,
        str(model),
        backend="torch",
        device=args.device,
        periodic=True,
        box_nm=box,
    )
    if args.platform:
        try:
            platform = mm.Platform.getPlatformByName(args.platform)
        except Exception:
            platform = preferred_platform()
            print(f"platform {args.platform} missing; using {platform.getName()}")
    else:
        platform = preferred_platform()

    integrator = mm.VerletIntegrator(args.dt_ps * unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions * unit.nanometers)
    context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in box))
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)

    n_atoms = len(types)
    n_frames = n_steps // stride + 1
    print(
        f"PET-MAD-XS NVE  n_mol={args.n_mol}  N={n_atoms}  "
        f"t={args.ps:g} ps  dt={args.dt_ps * 1000:g} fs  "
        f"frames={n_frames}  platform={platform.getName()}  "
        f"device={args.device}  -> {out}",
        flush=True,
    )

    t0 = time.perf_counter()
    e0 = None
    with out.open("w") as xyz, log_path.open("w") as csv:
        csv.write("t_ps,pe,ke,etot,T\n")
        frame = 0
        remaining = n_steps
        while True:
            state = context.getState(getEnergy=True, getPositions=True)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            pos = np.array(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            )
            pos = wrap_into_box(pos, box_nm)
            t_ps = (n_steps - remaining) * args.dt_ps
            temp = 2.0 * ke / (3 * n_atoms * R)
            etot = pe + ke
            if e0 is None:
                e0 = etot
            write_frame(
                xyz,
                symbols,
                pos * NM_TO_ANG,
                box_nm * NM_TO_ANG,
                f"Time={t_ps:.6f} energy={pe:.6f} ke={ke:.6f} T={temp:.2f}",
            )
            csv.write(f"{t_ps:.6f},{pe:.6f},{ke:.6f},{etot:.6f},{temp:.4f}\n")
            xyz.flush()
            csv.flush()
            frame += 1
            elapsed = time.perf_counter() - t0
            if frame == 1 or frame % 50 == 0 or remaining == 0:
                print(
                    f"  frame {frame:>4}/{n_frames}  t={t_ps:6.3f} ps  "
                    f"E={etot:.3f}  dE={etot - e0:.3f}  T={temp:.1f} K  "
                    f"wall={elapsed:.0f}s",
                    flush=True,
                )
            if remaining == 0:
                break
            chunk = min(stride, remaining)
            integrator.step(chunk)
            remaining -= chunk

    print(f"wrote {out}  ({frame} frames)", flush=True)
    print(f"wrote {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
