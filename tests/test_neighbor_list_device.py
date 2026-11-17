"""The CUDA neighbor list must agree with the host one, and must actually run.

A silent fall back to the host list would make every agreement check here pass
for the wrong reason, so each case asserts on the fallback message too.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
import openmm.app as app
import openmm.unit as unit

MODEL = os.path.abspath(
    os.environ.get(
        "OPENMM_METATOMIC_TEST_MODEL",
        os.path.join(os.path.dirname(__file__), "..", "models", "pet-mad-xs-v1.5.0.pt"),
    )
)

pytestmark = pytest.mark.skipif(
    "CUDA" not in [
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    ]
    or not os.path.exists(MODEL),
    reason="needs the CUDA platform and a metatomic model",
)

# Run each case in its own process: the neighbor-list device is an environment
# variable read by the plugin, and vesin's CUDA capacity is global to a process.
DRIVER = textwrap.dedent(
    """
    import sys, numpy as np, openmm as mm, openmm.app as app, openmm.unit as unit
    import openmmmetatomic
    from openmmmetatomic.openmmml import register
    register()
    from openmmml import MLPotential

    import pickle

    model, geometry, steps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    seed = sys.argv[5] if len(sys.argv) > 5 else ""
    top, positions, boxVectors = pickle.load(open(geometry, "rb"))
    if seed:
        # Evaluate at positions another run ended on, so the comparison is of
        # two neighbor lists at one geometry and not of two trajectories.
        positions = np.load(seed)["positions"] * unit.nanometer

    system = MLPotential("metatomic-native", modelPath=model, backend="torch").createSystem(top)
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.0005 * unit.picoseconds
    )
    simulation = app.Simulation(
        top, system, integrator, mm.Platform.getPlatformByName("CUDA"),
        {"DeviceIndex": "0", "Precision": "mixed"},
    )
    simulation.context.setPositions(positions)
    if boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*boxVectors)
    if steps:
        # Walk the system so vesin's skin cache is built, hit and rebuilt; a
        # single evaluation would never exercise the cached path at all.
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
        simulation.step(steps)
    state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    np.savez(
        out,
        energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        forces=state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        positions=state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
    )
    """
)

FALLBACK = "falling back to a host neighbor list"
ON_CUDA = "neighbor list on CUDA device"


def _run(tmp_path, geometry, device, steps, extra_env=None, seed=None):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    out = tmp_path / f"{device}-{steps}-{'seeded' if seed else 'fresh'}.npz"
    env = dict(
        os.environ,
        OPENMM_METATOMIC_NEIGHBOR_DEVICE=device,
        OPENMM_METATOMIC_NEIGHBOR_VERBOSE="1",
    )
    env.pop("VESIN_CUDA_MAX_PAIRS_PER_POINT", None)
    env.update(extra_env or {})
    command = [sys.executable, str(script), MODEL, str(geometry), str(steps), str(out)]
    if seed is not None:
        command.append(str(seed))
    # Run from a scratch directory: the wrapper resolves libOpenMMMetatomic.so
    # through a relative path, so a cwd that happens to hold a build/ directory
    # loads that library instead of the one on PYTHONPATH.
    done = subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=str(tmp_path)
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return np.load(out), done.stderr


def _water(tmp_path, n_waters, scale=1.0, triclinic=False):
    import pickle

    forcefield = app.ForceField("amber14/tip3pfb.xml")
    side = (n_waters / 33.4) ** (1 / 3) * scale
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(
        forcefield, boxSize=openmm.Vec3(side, side, side) * unit.nanometer
    )
    vectors = None
    if triclinic:
        # a 70-degree monoclinic cell, which is what sends vesin's "auto" to a
        # cell list no matter the point count
        vectors = (
            openmm.Vec3(side, 0, 0),
            openmm.Vec3(side * np.cos(np.deg2rad(70)), side * np.sin(np.deg2rad(70)), 0),
            openmm.Vec3(0, 0, side),
        ) * unit.nanometer
        modeller.topology.setPeriodicBoxVectors(vectors)
    path = tmp_path / f"water-{n_waters}-{scale}-{triclinic}.pkl"
    with open(path, "wb") as handle:
        pickle.dump((modeller.topology, modeller.positions, vectors), handle)
    return path


def _assert_agrees(cuda, host, tolerance=1e-5):
    assert cuda["energy"] == pytest.approx(host["energy"], rel=tolerance)
    scale = max(np.abs(host["forces"]).max(), 1.0)
    assert np.abs(cuda["forces"] - host["forces"]).max() / scale < tolerance


def _assert_ran_on_cuda(stderr):
    """Without this the agreement tests pass whenever the GPU list never ran."""
    assert FALLBACK not in stderr
    assert ON_CUDA in stderr, "the CUDA list did not run, so agreement proves nothing"


@pytest.mark.parametrize("n_waters", [20, 233])
def test_cuda_list_matches_host_list(tmp_path, n_waters):
    geometry = _water(tmp_path, n_waters)
    cuda, stderr = _run(tmp_path, geometry, "cuda", 0)
    _assert_ran_on_cuda(stderr)
    host, _ = _run(tmp_path, geometry, "cpu", 0)
    _assert_agrees(cuda, host)


def _assert_stepped_run_agrees(tmp_path, geometry, steps, tolerance=1e-4, env=None):
    """Step on CUDA so vesin's skin cache is built, hit and rebuilt, then check
    the cached list against a fresh host list at the geometry it ended on.

    Comparing two trajectories instead would fail on physics: a 4e-7 force
    difference grows to 0.025 nm within 60 Langevin steps.
    """
    cuda, stderr = _run(tmp_path, geometry, "cuda", steps, env)
    host, _ = _run(tmp_path, geometry, "cpu", 0, env, seed=tmp_path / f"cuda-{steps}-fresh.npz")
    assert np.abs(cuda["positions"] - host["positions"]).max() < 1e-9
    _assert_agrees(cuda, host, tolerance)
    return stderr


def test_cuda_list_matches_host_list_after_many_steps(tmp_path):
    """The cached (skin > 0) path is the default, so it is the one that matters."""
    stderr = _assert_stepped_run_agrees(tmp_path, _water(tmp_path, 233), 60)
    _assert_ran_on_cuda(stderr)


def test_cuda_list_handles_a_triclinic_cell(tmp_path):
    """vesin throws on brute force in a non-orthogonal cell, so "auto" has to
    survive below the cell-list threshold."""
    stderr = _assert_stepped_run_agrees(tmp_path, _water(tmp_path, 60, triclinic=True), 5)
    _assert_ran_on_cuda(stderr)


def test_cuda_list_handles_a_cutoff_past_half_the_box(tmp_path):
    stderr = _assert_stepped_run_agrees(tmp_path, _water(tmp_path, 8, scale=0.75), 5)
    _assert_ran_on_cuda(stderr)


def test_an_unfittable_capacity_falls_back_instead_of_failing(tmp_path):
    """A capacity the caller chose is honoured, and an impossible one must not
    take the run down with it."""
    stderr = _assert_stepped_run_agrees(
        tmp_path, _water(tmp_path, 233), 5, env={"VESIN_CUDA_MAX_PAIRS_PER_POINT": "4"}
    )
    assert FALLBACK in stderr
    assert ON_CUDA not in stderr
