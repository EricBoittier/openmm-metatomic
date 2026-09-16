"""Shared harmonic well, demo systems, and evaluation helpers for the gallery."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    load_atomistic_model,
)

K = 1.0

# Positions are nanometers. Energy is kJ/mol. E = 0.5 k ||r - r0||^2
SYSTEMS: Dict[str, dict] = {
    "water": {
        "title": "water (HOH)",
        "types": [1, 1, 8],
        "rest": [
            [0.0757, 0.0586, 0.0],
            [-0.0757, 0.0586, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "positions": [
            [0.0857, 0.0586, 0.0],
            [-0.0757, 0.0486, 0.01],
            [0.0, 0.01, -0.005],
        ],
        "periodic": False,
        "cell": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    },
    "methane": {
        "title": "methane (CH4)",
        "types": [6, 1, 1, 1, 1],
        "rest": [
            [0.0, 0.0, 0.0],
            [0.06293, 0.06293, 0.06293],
            [0.06293, -0.06293, -0.06293],
            [-0.06293, 0.06293, -0.06293],
            [-0.06293, -0.06293, 0.06293],
        ],
        "positions": [
            [0.005, 0.0, -0.004],
            [0.07293, 0.06293, 0.06293],
            [0.06293, -0.05293, -0.06293],
            [-0.06293, 0.06293, -0.05293],
            [-0.07293, -0.06293, 0.06293],
        ],
        "periodic": False,
        "cell": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    },
    "co2": {
        "title": "carbon dioxide",
        "types": [6, 8, 8],
        "rest": [
            [0.0, 0.0, 0.0],
            [0.116, 0.0, 0.0],
            [-0.116, 0.0, 0.0],
        ],
        "positions": [
            [0.0, 0.008, 0.0],
            [0.126, 0.0, 0.004],
            [-0.106, -0.006, 0.0],
        ],
        "periodic": False,
        "cell": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    },
    "carbon8": {
        "title": "eight-atom carbon cube",
        "types": [6] * 8,
        "rest": [
            [sx * 0.07, sy * 0.07, sz * 0.07]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        "positions": [
            [sx * 0.07 + 0.008, sy * 0.07 - 0.005, sz * 0.07 + 0.003]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        "periodic": False,
        "cell": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    },
    "water_pbc": {
        "title": "water in a 1.5 nm cubic box",
        "types": [1, 1, 8],
        "rest": [
            [0.0757, 0.0586, 0.0],
            [-0.0757, 0.0586, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "positions": [
            [0.0857, 0.0586, 0.0],
            [-0.0757, 0.0486, 0.01],
            [0.0, 0.01, -0.005],
        ],
        "periodic": True,
        "cell": [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
    },
}


def as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def analytic(system: dict, k: float = K) -> Tuple[float, np.ndarray]:
    rest = as_array(system["rest"])
    positions = as_array(system["positions"])
    delta = positions - rest
    energy = 0.5 * k * float((delta * delta).sum())
    forces = -k * delta
    return energy, forces


def finite_difference_forces(energy_fn, positions: np.ndarray, h: float = 1e-5) -> np.ndarray:
    forces = np.zeros_like(positions)
    pos = positions.copy()
    for i in range(positions.shape[0]):
        for c in range(3):
            pos[i, c] = positions[i, c] + h
            plus = energy_fn(pos)
            pos[i, c] = positions[i, c] - h
            minus = energy_fn(pos)
            pos[i, c] = positions[i, c]
            forces[i, c] = -(plus - minus) / (2.0 * h)
    return forces


class Harmonic(torch.nn.Module):
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
        if list(outputs.keys()) != ["energy"]:
            raise ValueError("this model only computes energy")
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")
        if outputs["energy"].sample_kind == "atom":
            raise NotImplementedError("per-atom energy is not implemented")

        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            delta = system.positions - self.rest.to(
                dtype=system.positions.dtype, device=system.positions.device
            )
            energy[i] = 0.5 * self.k * (delta * delta).sum()

        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


def export_system(system: dict, path: str, k: float = K) -> str:
    rest = torch.tensor(system["rest"], dtype=torch.float64)
    model = Harmonic(k, rest)
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
        atomic_types=sorted(set(system["types"]) | {1, 6, 8}),
        interaction_range=0.0,
        length_unit="nm",
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name=f"openmm-metatomic-harmonic-{system.get('title', 'system')}",
        description="Gallery TorchScript harmonic well",
    )
    AtomisticModel(model.eval(), metadata, capabilities).save(path)
    return path


def evaluate_core(system: dict, k: float = K) -> Tuple[float, np.ndarray]:
    """Same independent-atom well as the C++ ``HarmonicModel`` / metatomic-core path."""
    return analytic(system, k)


def _make_wrapper(system: dict, k: float = K) -> AtomisticModel:
    rest = torch.tensor(system["rest"], dtype=torch.float64)
    return AtomisticModel(
        Harmonic(k, rest).eval(),
        ModelMetadata(name="gallery-harmonic"),
        ModelCapabilities(
            outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
            atomic_types=sorted(set(system["types"]) | {1, 6, 8}),
            interaction_range=0.0,
            length_unit="nm",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )


def _evaluate_wrapper(wrapper: AtomisticModel, system: dict) -> Tuple[float, np.ndarray]:
    positions = torch.tensor(system["positions"], dtype=torch.float64, requires_grad=True)
    types = torch.tensor(system["types"], dtype=torch.int32)
    cell = torch.tensor(system["cell"], dtype=torch.float64)
    periodic = bool(system["periodic"])
    pbc = torch.tensor([periodic, periodic, periodic])
    options = ModelEvaluationOptions(
        length_unit="nm",
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
    )
    frame = System(types, positions, cell, pbc)
    output = wrapper([frame], options, check_consistency=True)
    energy = output["energy"].block().values.sum()
    energy.backward()
    forces = (-positions.grad).detach().cpu().numpy()
    return float(energy.item()), forces


def evaluate_torch(system: dict, k: float = K) -> Tuple[float, np.ndarray]:
    """In-process TorchScript ``AtomisticModel`` (not yet saved to ``.pt``)."""
    return _evaluate_wrapper(_make_wrapper(system, k), system)


def evaluate_pt(system: dict, path: Union[str, Path], k: float = K) -> Tuple[float, np.ndarray]:
    """Load an exported ``.pt`` with ``load_atomistic_model`` — the plugin torch backend."""
    del k
    return _evaluate_wrapper(load_atomistic_model(str(path)), system)
