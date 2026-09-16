"""SOAP-BPNN twin of metatrain's ``soap_bpnn`` (legacy path).

Metatrain's Behler–Parrinello network uses SOAP power spectra from torch-spex
instead of the original ACSFs, then a per-species MLP with SiLU and no bias
(see ``metatrain.soap_bpnn.model.MLPMap`` and ``SoapPowerSpectrum``).

This module is a small, fully specified twin of that stack so the same weights
can run in metatomic-core C++:

* Shifted-cosine cutoff (``SOAPCutoffConfig``)
* Bernstein radial basis (a torch-spex radial option; Laplacian eigenstates
  are splined in spex and not practical to duplicate here)
* real spherical harmonics, ``max_angular = 1``
* power-spectrum contraction ``sum_m c_{n m} c_{n' m}`` (same einsum as
  ``SoapPowerSpectrum``)
* orthogonal neighbor species (``legacy: true``)
* per-species ``Linear(bias=False) → SiLU → Linear(bias=False)`` onto atomic
  energy, then a sum over atoms

Units are nm and kJ/mol. Neighbor lists are built from positions (minimum
image on orthogonal cells), matching the C++ ``BpnnModel``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

from _harmonic import SYSTEMS, finite_difference_forces

SPECIES = (1, 6, 8)
CUTOFF = 0.5
WIDTH = 0.05
N_RADIAL = 2
N_SPECIES = len(SPECIES)
SOAP_L0 = N_SPECIES * N_RADIAL
SOAP_SIZE = SOAP_L0 * SOAP_L0 * 2  # l = 0 and l = 1
HIDDEN = 8
SPECIES_INDEX = {1: 0, 6: 1, 8: 2}


def default_weights() -> Dict[int, dict]:
    rng = np.random.RandomState(1)
    out = {}
    for z in SPECIES:
        out[int(z)] = {
            "w1": rng.randn(HIDDEN, SOAP_SIZE) * 0.05,
            "w2": rng.randn(HIDDEN) * 0.05,
        }
    return out


def shifted_cosine(r, rc: float = CUTOFF, width: float = WIDTH):
    r = np.asarray(r, dtype=np.float64)
    inner = rc - width
    out = np.ones_like(r)
    out = np.where(r >= rc, 0.0, out)
    mid = (r >= inner) & (r < rc)
    out = np.where(mid, 0.5 * (1.0 + np.cos(np.pi * (r - inner) / width)), out)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def displacements(positions: np.ndarray, cell: np.ndarray, periodic: bool) -> np.ndarray:
    delta = np.asarray(positions, dtype=np.float64)[None, :, :] - np.asarray(
        positions, dtype=np.float64
    )[:, None, :]
    if periodic:
        lengths = np.diag(np.asarray(cell, dtype=np.float64))
        for c, length in enumerate(lengths):
            if length > 0.0:
                delta[..., c] -= length * np.round(delta[..., c] / length)
    return delta


def soap_features(types: Sequence[int], positions, cell, periodic) -> np.ndarray:
    types = np.asarray(types, dtype=np.int32)
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    delta = displacements(pos, cell, periodic)
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, np.inf)
    valid = (dist <= CUTOFF) & (dist > 1e-14)
    hat = np.zeros_like(delta)
    hat[valid] = delta[valid] / dist[valid][:, None]
    dist_cut = np.where(valid, dist, CUTOFF)
    fc = shifted_cosine(dist_cut) * valid
    x = np.clip(dist_cut / CUTOFF, 0.0, 1.0)
    radial = np.stack([(1.0 - x) * fc, x * fc], axis=-1)
    onehot = np.stack([(types == z).astype(np.float64) for z in SPECIES], axis=-1)
    c0 = np.einsum("ijn,js->isn", radial, onehot)
    c1 = np.einsum("ijn,ijm,js->isnm", radial, hat, onehot)
    c0f = c0.reshape(n, SOAP_L0)
    c1f = c1.reshape(n, SOAP_L0, 3)
    p0 = np.einsum("ia,ib->iab", c0f, c0f).reshape(n, -1)
    p1 = np.einsum("iam,ibm->iab", c1f, c1f).reshape(n, -1)
    return np.concatenate([p0, p1], axis=-1)


def numpy_energy(types, positions, cell, periodic, weights=None) -> float:
    weights = default_weights() if weights is None else weights
    features = soap_features(types, positions, cell, periodic)
    energy = 0.0
    for i, z in enumerate(types):
        w = weights[int(z)]
        hidden = silu(w["w1"] @ features[i])
        energy += float(w["w2"] @ hidden)
    return energy


class SoapBpnn(torch.nn.Module):
    """TorchScript twin of ``OpenMMMetatomic::BpnnModel`` / metatrain SOAP-BPNN."""

    def __init__(self, weights=None):
        super().__init__()
        weights = default_weights() if weights is None else weights
        w1 = np.stack([weights[z]["w1"] for z in SPECIES], axis=0)
        w2 = np.stack([weights[z]["w2"] for z in SPECIES], axis=0)
        self.register_buffer("w1", torch.tensor(w1, dtype=torch.float64))
        self.register_buffer("w2", torch.tensor(w2, dtype=torch.float64))
        self.cutoff = float(CUTOFF)
        self.width = float(WIDTH)

    def _min_image(self, delta: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
        out = delta
        lengths = torch.stack([cell[0, 0], cell[1, 1], cell[2, 2]])
        pbc_f = pbc.to(dtype=delta.dtype)
        one = torch.ones((), dtype=delta.dtype, device=delta.device)
        for c in range(3):
            length = lengths[c]
            apply = pbc_f[c] * (length > 0).to(dtype=delta.dtype)
            safe = torch.where(length > 0, length, one)
            shifted = out.clone()
            shifted[..., c] = out[..., c] - apply * length * torch.round(out[..., c] / safe)
            out = shifted
        return out

    def _shifted_cosine(self, r: torch.Tensor) -> torch.Tensor:
        inner = self.cutoff - self.width
        pi = torch.tensor(3.141592653589793, dtype=r.dtype, device=r.device)
        mid = 0.5 * (1.0 + torch.cos(pi * (r - inner) / self.width))
        return torch.where(
            r >= self.cutoff,
            torch.zeros_like(r),
            torch.where(r >= inner, mid, torch.ones_like(r)),
        )

    def _soap(self, types: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor):
        n = positions.shape[0]
        delta = self._min_image(positions.unsqueeze(0) - positions.unsqueeze(1), cell, pbc)
        dist = torch.linalg.norm(delta, dim=-1)
        eye = torch.eye(n, dtype=torch.bool, device=positions.device)
        valid = (~eye) & (dist <= self.cutoff) & (dist > 1e-14)
        r_safe = torch.where(valid, dist, torch.ones_like(dist))
        hat = torch.where(valid.unsqueeze(-1), delta / r_safe.unsqueeze(-1), torch.zeros_like(delta))
        fc = self._shifted_cosine(dist) * valid.to(dtype=dist.dtype)
        x = (dist / self.cutoff).clamp(0.0, 1.0)
        radial = torch.stack([(1.0 - x) * fc, x * fc], dim=-1)
        onehot = torch.stack(
            [
                (types == 1).to(dtype=dist.dtype),
                (types == 6).to(dtype=dist.dtype),
                (types == 8).to(dtype=dist.dtype),
            ],
            dim=-1,
        )
        c0 = torch.einsum("ijn,js->isn", radial, onehot)
        c1 = torch.einsum("ijn,ijm,js->isnm", radial, hat, onehot)
        c0f = c0.reshape(n, 6)
        c1f = c1.reshape(n, 6, 3)
        p0 = torch.einsum("ia,ib->iab", c0f, c0f).reshape(n, -1)
        p1 = torch.einsum("iam,ibm->iab", c1f, c1f).reshape(n, -1)
        return torch.cat([p0, p1], dim=-1)

    def _atomic_energies(self, types: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        energy = features.new_zeros((features.shape[0],))
        hidden0 = torch.nn.functional.silu(features @ self.w1[0].T)
        energy = torch.where(types == 1, hidden0 @ self.w2[0], energy)
        hidden1 = torch.nn.functional.silu(features @ self.w1[1].T)
        energy = torch.where(types == 6, hidden1 @ self.w2[1], energy)
        hidden2 = torch.nn.functional.silu(features @ self.w1[2].T)
        energy = torch.where(types == 8, hidden2 @ self.w2[2], energy)
        return energy

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")
        if "energy" not in outputs:
            raise ValueError("this model only computes energy")
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            features = self._soap(system.types, system.positions, system.cell, system.pbc)
            energy[i] = self._atomic_energies(system.types, features).sum()
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


def capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
        atomic_types=list(SPECIES),
        interaction_range=CUTOFF,
        length_unit="nm",
        supported_devices=["cpu"],
        dtype="float64",
    )


def wrap(model: Optional[SoapBpnn] = None) -> AtomisticModel:
    return AtomisticModel(
        (model or SoapBpnn()).eval(),
        ModelMetadata(
            name="openmm-metatomic-soap-bpnn",
            description="SOAP-BPNN twin of metatrain soap_bpnn (legacy MLP)",
        ),
        capabilities(),
    )


def export_bpnn(path: str, model: Optional[SoapBpnn] = None) -> str:
    wrap(model).save(path)
    return path


def evaluate_torch(system: dict, wrapper: Optional[AtomisticModel] = None) -> Tuple[float, np.ndarray]:
    wrapper = wrap() if wrapper is None else wrapper
    positions = torch.tensor(system["positions"], dtype=torch.float64, requires_grad=True)
    frame = System(
        torch.tensor(system["types"], dtype=torch.int32),
        positions,
        torch.tensor(system["cell"], dtype=torch.float64),
        torch.tensor([bool(system["periodic"])] * 3),
    )
    options = ModelEvaluationOptions(
        length_unit="nm",
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
    )
    energy = wrapper([frame], options, check_consistency=True)["energy"].block().values.sum()
    energy.backward()
    forces = (-positions.grad).detach().cpu().numpy()
    return float(energy.item()), forces


def evaluate_pt(system: dict, path: Union[str, Path]) -> Tuple[float, np.ndarray]:
    return evaluate_torch(system, load_atomistic_model(str(path)))


def evaluate_numpy(system: dict) -> float:
    return numpy_energy(system["types"], system["positions"], system["cell"], system["periodic"])


def emit_cpp_weights(path: Union[str, Path]) -> None:
    weights = default_weights()
    lines = [
        "// Generated by examples/_bpnn.py — do not edit by hand.",
        "#pragma once",
        f"constexpr int kBpnnSoapSize = {SOAP_SIZE};",
        f"constexpr int kBpnnHidden = {HIDDEN};",
        f"constexpr int kBpnnNSpecies = {N_SPECIES};",
        "constexpr double kBpnnW1[3][8][72] = {",
    ]
    for z in SPECIES:
        lines.append("    {")
        for row in weights[z]["w1"]:
            vals = ", ".join(f"{x:.16g}" for x in row)
            lines.append(f"        {{{vals}}},")
        lines.append("    },")
    lines.append("};")
    lines.append("constexpr double kBpnnW2[3][8] = {")
    for z in SPECIES:
        vals = ", ".join(f"{x:.16g}" for x in weights[z]["w2"])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    Path(path).write_text("\n".join(lines) + "\n")


def _check_systems(export_path: Optional[Path] = None) -> None:
    wrapper = wrap()
    pt = None
    if export_path is not None:
        pt = export_bpnn(str(export_path))
    print(
        f"{'system':<12} {'N':>3} {'E_np':>12} {'E_torch':>12} {'ΔE':>10} {'ΔF_FD':>10}  pbc"
    )
    for name, system in SYSTEMS.items():
        e_np = evaluate_numpy(system)
        e_t, f_t = evaluate_torch(system, wrapper)
        f_fd = finite_difference_forces(
            lambda pos, system=system: evaluate_numpy({**system, "positions": pos}),
            np.asarray(system["positions"], dtype=np.float64),
            h=1e-6,
        )
        d_e = abs(e_t - e_np)
        d_f = float(np.max(np.abs(f_t - f_fd)))
        assert d_e < 1e-10, (name, d_e, e_np, e_t)
        assert d_f < 5e-5, (name, d_f)
        if pt is not None:
            e_p, f_p = evaluate_pt(system, pt)
            assert abs(e_p - e_np) < 1e-10
            assert float(np.max(np.abs(f_p - f_t))) < 1e-8
        print(
            f"{name:<12} {len(system['types']):>3} {e_np:12.6e} {e_t:12.6e} "
            f"{d_e:10.3e} {d_f:10.3e}  {system['periodic']}"
        )
    print("SOAP-BPNN torch matches numpy; autograd forces match finite differences")


if __name__ == "__main__":
    _check_systems()
