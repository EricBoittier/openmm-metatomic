"""OpenMM / OpenMM-ML helpers that match the current end-user path.

The native plugin is a first-class Python entry point:
``MLPotential("metatomic-native")`` after ``openmmmetatomic.register()`` (or
the ``openmmml.potentials`` entry point when the package is installed).
``MLPotential("metatomic")`` still installs a ``PythonForce``. These helpers
follow that API, plus the same Amber mixed systems used by OpenMM-ML's
metatomic tests.
"""

import inspect
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
    unit_conversion_factor,
)

K_EV = 1.0
EV_TO_KJ = float(unit_conversion_factor("eV", "kJ/mol"))


def repo_root() -> Path:
    env = os.environ.get("OPENMM_METATOMIC_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "CMakeLists.txt").is_file() and (parent / "openmmapi").is_dir():
            return parent
    return Path("/home/ericb/metawork/openmm-metatomic")


def openmm_ml_data() -> Optional[Path]:
    env = os.environ.get("OPENMM_ML_TEST_DATA")
    if env:
        path = Path(env)
        return path if path.is_dir() else None
    candidates = [
        repo_root().parent / "openmm-ml" / "test" / "data",
        Path("/home/ericb/metawork/openmm-ml/test/data"),
    ]
    for path in candidates:
        if (path / "toluene" / "toluene.pdb").is_file():
            return path.resolve()
    return None


def spike_binary() -> Optional[Path]:
    path = repo_root() / "build" / "openmm-metatomic-spike"
    return path if path.is_file() else None


def supported_devices() -> List[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


class OpenMMHarmonic(torch.nn.Module):
    """Independent-atom well in OpenMM-ML units: Å and eV, ``E = k Σ||Δr||²``."""

    def __init__(self, k: float, rest: torch.Tensor):
        super().__init__()
        self.k = float(k)
        self.register_buffer("rest", rest.to(dtype=torch.float64))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        del selected_atoms, outputs
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            rest = self.rest.to(dtype=system.positions.dtype, device=system.positions.device)
            energy[i] = self.k * ((system.positions - rest) ** 2).sum()
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


class NeighborHarmonic(OpenMMHarmonic):
    def __init__(self, k: float, rest: torch.Tensor, cutoff: float = 4.0):
        super().__init__(k, rest)
        self._nl = NeighborListOptions(cutoff, False, True)

    def requested_neighbor_lists(self):
        return [self._nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        del selected_atoms, outputs
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            neighbors = system.get_neighbor_list(self._nl)
            energy[i] = neighbors.values.sum() * 0.0
            rest = self.rest.to(dtype=system.positions.dtype, device=system.positions.device)
            energy[i] = energy[i] + self.k * ((system.positions - rest) ** 2).sum()
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


def export_openmm_model(
    path: str,
    rest_angstrom,
    atomic_types: Sequence[int],
    k: float = K_EV,
    neighbor_list: bool = False,
    devices: Optional[Sequence[str]] = None,
) -> str:
    rest = torch.tensor(np.asarray(rest_angstrom, dtype=np.float64))
    model = NeighborHarmonic(k, rest) if neighbor_list else OpenMMHarmonic(k, rest)
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
        atomic_types=sorted(set(atomic_types)),
        interaction_range=4.0 if neighbor_list else 0.0,
        length_unit="Angstrom",
        supported_devices=list(devices or supported_devices()),
        dtype="float64",
    )
    AtomisticModel(model.eval(), ModelMetadata(name="gallery-openmm-harmonic"), capabilities).save(
        path
    )
    return path


def direct_energy_forces(model_path: str, numbers, positions_angstrom, device: str = "cpu"):
    model = load_atomistic_model(model_path)
    model = model.to(device)
    dtype = torch.float64
    types = torch.tensor(list(numbers), dtype=torch.int32, device=device)
    pos = torch.tensor(positions_angstrom, dtype=dtype, device=device, requires_grad=True)
    cell = torch.zeros((3, 3), dtype=dtype, device=device)
    pbc = torch.tensor([False, False, False], device=device)
    system = System(types, pos, cell, pbc)
    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
    )
    energy = model([system], options, False)["energy"].block().values.sum()
    energy.backward()
    forces = (-pos.grad * EV_TO_KJ * 10.0).detach().cpu().numpy()
    return float(energy.detach()) * EV_TO_KJ, forces


def vacuum_topology(symbols: Sequence[str]):
    import openmm.app as app

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    for i, symbol in enumerate(symbols):
        element = app.Element.getBySymbol(symbol)
        topology.addAtom(f"{symbol}{i+1}", element, residue)
    return topology


def symbols_from_numbers(numbers: Sequence[int]) -> List[str]:
    table = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}
    return [table[int(z)] for z in numbers]


def openmm_platforms(include: Optional[Iterable[str]] = None) -> list:
    import openmm as mm

    wanted = None if include is None else set(include)
    out = []
    for i in range(mm.Platform.getNumPlatforms()):
        platform = mm.Platform.getPlatform(i)
        if wanted is not None and platform.getName() not in wanted:
            continue
        out.append(platform)
    return out


def preferred_platforms() -> list:
    """CPU and Reference are the reliable gallery platforms; OpenCL is optional."""
    names = ["CPU", "Reference"]
    found = {p.getName(): p for p in openmm_platforms()}
    return [found[name] for name in names if name in found]


def ensure_openmm_ml_embeddings():
    """Register mechanical embedding when the package was not installed with
    entry points (editable / PYTHONPATH checkout)."""
    from openmmml import MLPotential

    factories = getattr(MLPotential, "_embeddingFactories", None)
    if factories is None:
        return
    if "mechanical" in factories:
        return
    try:
        from openmmml.embeddings.mechanicalembedding import MechanicalEmbeddingFactory
    except ImportError:
        return
    MLPotential.registerEmbeddingFactory("mechanical", MechanicalEmbeddingFactory())


def make_potential(
    model_path: str,
    device: str = "cpu",
    check_consistency: bool = False,
    **kwargs,
):
    from openmmml import MLPotential

    ensure_openmm_ml_embeddings()
    return MLPotential(
        "metatomic",
        modelPath=model_path,
        device=device,
        checkConsistency=check_consistency,
        **kwargs,
    )


def ensure_native_backend():
    """Register ``MLPotential("metatomic-native")`` from the build tree."""
    import sys

    built = repo_root() / "build" / "python"
    if built.is_dir() and str(built) not in sys.path:
        sys.path.insert(0, str(built))
    plugin = repo_root() / "build"
    if (plugin / "libOpenMMMetatomic.so").is_file():
        os.environ.setdefault("OPENMM_PLUGIN_DIR", str(plugin))
    import openmmmetatomic

    return openmmmetatomic.register()


def make_native_potential(
    model_path: str,
    device: str = "cpu",
    check_consistency: bool = False,
    **kwargs,
):
    from openmmml import MLPotential

    ensure_openmm_ml_embeddings()
    name = ensure_native_backend()
    return MLPotential(
        name,
        modelPath=model_path,
        device=device,
        checkConsistency=check_consistency,
        **kwargs,
    )


def mixed_system(potential, topology, system, atoms, **kwargs):
    """``createMixedSystem`` for either backend.

    The native backend reports ``getMLLongRange() is False``, and mechanical
    embedding then rejects an explicit ``mlLongRange``; the PythonForce backend
    reports ``None`` and has to be told. Both end up with a short-ranged ML part.
    """
    if potential._impl.getMLLongRange() is None:
        kwargs.setdefault("mlLongRange", False)
    return potential.createMixedSystem(topology, system, atoms, **kwargs)


def _example_script_path(script_path=None) -> Path:
    """Path of the calling ``plot_*.py``. Sphinx-gallery often has no ``__file__``."""
    if script_path is not None:
        try:
            path = Path(script_path)
            if path.suffix == ".py":
                return path
        except TypeError:
            pass
    for frame in inspect.stack()[1:]:
        name = Path(frame.filename).name
        if name.startswith("plot_") and name.endswith(".py"):
            return Path(frame.filename)
    return repo_root() / "examples" / "gallery.py"


def save_figure(fig, script_path=None, suffix: str = ""):
    path = _example_script_path(script_path)
    out = path.resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    dest = out / (path.stem + suffix + ".png")
    fig.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return dest


def energy_forces(context) -> Tuple[float, np.ndarray]:
    import openmm.unit as unit

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    return float(energy), np.asarray(forces, dtype=np.float64)


def timed(fn: Callable, repeat: int = 1, warmup: int = 0):
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    array = np.asarray(times, dtype=np.float64)
    return result, array


def ms(seconds: np.ndarray) -> Tuple[float, float]:
    return float(np.median(seconds) * 1e3), float(np.min(seconds) * 1e3)


def format_ms(seconds: np.ndarray) -> str:
    med, lo = ms(seconds)
    return f"{med:8.3f} ms  (min {lo:.3f})"
