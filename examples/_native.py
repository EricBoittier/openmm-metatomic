"""Helpers for driving the native MetatomicForce Python wrapper."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from _openmm import repo_root, vacuum_topology, symbols_from_numbers
from _petmad import WATER_NM, WATER_NUMBERS, WATER_SYMBOLS, ensure_model

NM_TO_ANG = 10.0
SPACING_NM = 0.35
MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}


def _ensure_python_path() -> None:
    root = repo_root()
    built = root / "build" / "python"
    if built.is_dir():
        text = str(built)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    plugin = root / "build"
    if (plugin / "libOpenMMMetatomic.so").is_file():
        os.environ.setdefault("OPENMM_PLUGIN_DIR", str(plugin))


def native_available() -> bool:
    """True when the SWIG module can be imported (plugin built in ``build/``)."""
    _ensure_python_path()
    try:
        import openmmmetatomic  # noqa: F401
    except ImportError:
        return False
    return True


def MetatomicForce():
    _ensure_python_path()
    from openmmmetatomic import MetatomicForce as cls
    return cls


def mass_for(atomic_number: int) -> float:
    return MASSES.get(int(atomic_number), 12.0)


def water_box(n_molecules: int, spacing: float = SPACING_NM):
    n_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    types = []
    positions = []
    count = 0
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if count >= n_molecules:
                    break
                off = np.array([ix, iy, iz], dtype=np.float64) * spacing
                for t, p in zip(WATER_NUMBERS, WATER_NM):
                    types.append(t)
                    positions.append(p + off)
                count += 1
    box = n_side * spacing
    return (
        np.asarray(types, dtype=np.int32),
        np.asarray(positions, dtype=np.float64),
        np.eye(3) * box,
    )


def harmonic_cloud(n_atoms: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    types = np.array([1, 6, 8] * ((n_atoms + 2) // 3), dtype=np.int32)[:n_atoms]
    positions = rng.uniform(-0.3, 0.3, size=(n_atoms, 3))
    return types, positions


def save_figure(fig, script_path=None, suffix: str = ""):
    from _openmm import save_figure as _save

    return _save(fig, script_path, suffix=suffix)


def periodic_topology(types: Sequence[int], box_nm: np.ndarray):
    import openmm.unit as unit

    topology = vacuum_topology(symbols_from_numbers(types))
    topology.setPeriodicBoxVectors(box_nm * unit.nanometers)
    return topology


def add_force(
    system,
    model_path: str,
    types: Sequence[int],
    backend: str = "auto",
    device: str = "",
    check_consistency: bool = False,
    periodic: bool = False,
):
    force_cls = MetatomicForce()
    force = force_cls(str(model_path))
    force.setBackend(backend)
    if device:
        force.setDevice(device)
    force.setCheckConsistency(check_consistency)
    force.setAtomicTypes([int(t) for t in types])
    force.setUsesPeriodicBoundaryConditions(periodic)
    system.addForce(force)
    return force


def make_system(
    types: Sequence[int],
    model_path: str,
    backend: str = "auto",
    device: str = "",
    check_consistency: bool = False,
    periodic: bool = False,
    box_nm: Optional[np.ndarray] = None,
):
    import openmm as mm

    system = mm.System()
    for z in types:
        system.addParticle(mass_for(z))
    if periodic and box_nm is not None:
        system.setDefaultPeriodicBoxVectors(
            mm.Vec3(*box_nm[0]), mm.Vec3(*box_nm[1]), mm.Vec3(*box_nm[2])
        )
    add_force(
        system,
        model_path,
        types,
        backend=backend,
        device=device,
        check_consistency=check_consistency,
        periodic=periodic,
    )
    return system


def preferred_platform():
    from _openmm import preferred_platforms

    plats = preferred_platforms()
    if not plats:
        import openmm as mm

        return mm.Platform.getPlatformByName("Reference")
    return plats[0]


def energies(context, n_atoms: int, remove_cm: bool = False):
    import openmm.unit as unit

    state = context.getState(getEnergy=True)
    pot = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    kin = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    dof = 3 * n_atoms - (3 if remove_cm else 0)
    r = 8.314462618e-3
    temp = (2.0 * kin / (dof * r)) if dof else 0.0
    return float(pot), float(kin), float(temp)


def run_nve(system, positions, platform, n_steps: int, dt_ps: float = 0.0005, temperature: float = 300.0, box=None):
    import openmm as mm
    import openmm.unit as unit

    integrator = mm.VerletIntegrator(dt_ps * unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions * unit.nanometers)
    if box is not None:
        context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in box))
    context.setVelocitiesToTemperature(temperature * unit.kelvin, 1)
    n = len(positions)
    trace = []
    for _ in range(n_steps + 1):
        pot, kin, temp = energies(context, n, False)
        trace.append((pot, kin, pot + kin, temp))
        integrator.step(1)
    return np.asarray(trace, dtype=np.float64)


def run_nvt(system, positions, platform, n_steps: int, dt_ps: float = 0.0005, temperature: float = 300.0, box=None):
    import openmm as mm
    import openmm.unit as unit

    system.addForce(mm.CMMotionRemover())
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picoseconds
    )
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions * unit.nanometers)
    if box is not None:
        context.setPeriodicBoxVectors(*(mm.Vec3(*row) for row in box))
    context.setVelocitiesToTemperature(temperature * unit.kelvin, 2)
    n = len(positions)
    trace = []
    for _ in range(n_steps + 1):
        pot, kin, temp = energies(context, n, True)
        trace.append((pot, kin, pot + kin, temp))
        integrator.step(1)
    return np.asarray(trace, dtype=np.float64)


def run_md_binary() -> Optional[Path]:
    path = repo_root() / "build" / "openmm-metatomic-run-md"
    return path if path.is_file() else None


def settings_binary() -> Optional[Path]:
    path = repo_root() / "build" / "openmm-metatomic-bench-settings"
    return path if path.is_file() else None


def petmad_path() -> Path:
    return ensure_model()


def soap_pt() -> Optional[Path]:
    path = repo_root() / "build" / "soap-bpnn.pt"
    return path if path.is_file() else None


def harmonic_pt() -> Optional[Path]:
    path = repo_root() / "build" / "harmonic.pt"
    return path if path.is_file() else None


def pairlist_pt() -> Optional[Path]:
    path = repo_root() / "build" / "pairlist.pt"
    return path if path.is_file() else None


def ensure_harmonic_pt(n_atoms: int) -> Optional[Path]:
    root = repo_root()
    out = root / "build" / "scaling" / f"harmonic-{n_atoms}.pt"
    if out.is_file():
        return out
    script = root / "spike" / "export_scaling_models.py"
    if not script.is_file():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(
        [sys.executable, str(script), str(out.parent), str(n_atoms)],
        check=True,
    )
    return out if out.is_file() else None
