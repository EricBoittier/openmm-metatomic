"""SOAP-BPNN twin of metatrain ``soap_bpnn`` (legacy path).

SOAP is the torch-spex spherical expansion (Laplacian eigenstates, Wikipedia
real spherical harmonics via sphericart, orthogonal species, shifted-cosine
cutoff) contracted with the same ``einsum("smn,smN->snN")`` power spectrum as
``metatrain.soap_bpnn.modules.power_spectrum.SoapPowerSpectrum``. The BPNN is
the per-species ``Linear(bias=False) → SiLU → Linear(bias=False)`` map.

The C++ ``BpnnModel`` uses sphericart and the dumped cubic-Hermite spline of
the same Laplacian eigenstates so core, TorchScript, torch-spex, and metatrain
agree numerically.

Units are nm and kJ/mol. Neighbor lists are brute-force minimum-image
displacements ``R_ij = R_j - R_i`` with ``0 < r ≤ cutoff``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sphericart
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
from spex.radial.physical.laplacian_eigenstates import LaplacianEigenstates
from spex.spherical_expansion import SphericalExpansion

from _harmonic import SYSTEMS, finite_difference_forces

SPECIES = (1, 6, 8)
CUTOFF = 0.5
WIDTH = 0.05
MAX_ANGULAR = 1
MAX_RADIAL = 2
N_PER_L = (3, 2)
N_SPECIES = len(SPECIES)
N_RADIAL_FEAT = sum(N_PER_L)
SOAP_SIZE = sum(n * n * N_SPECIES * N_SPECIES for n in N_PER_L)
HIDDEN = 8
SPECIES_INDEX = {1: 0, 6: 1, 8: 2}

_SPLINE = None
_SPHERICART = sphericart.SphericalHarmonics(l_max=MAX_ANGULAR)


def laplacian_spline() -> dict:
    """Dump torch-spex LaplacianEigenstates cubic Hermite coefficients."""
    global _SPLINE
    if _SPLINE is None:
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            radial = LaplacianEigenstates(
                cutoff=CUTOFF, max_angular=MAX_ANGULAR, max_radial=MAX_RADIAL
            )
        finally:
            torch.set_default_dtype(prev)
        n_per_l = tuple(int(n) for n in radial.n_per_l)
        if n_per_l != N_PER_L:
            raise RuntimeError(f"unexpected n_per_l {n_per_l}, expected {N_PER_L}")
        _SPLINE = {
            "n_per_l": n_per_l,
            "values": radial.spliner.spline_values.detach().cpu().to(torch.float64).numpy(),
            "derivs": radial.spliner.spline_derivatives.detach().cpu().to(torch.float64).numpy(),
            "spacing": float(radial.spliner.spline_spacing.detach().cpu().to(torch.float64)),
        }
    return _SPLINE


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
    onset = rc - width
    cosine = 0.5 * (1.0 + np.cos(np.pi * (r - onset) / width))
    return np.where(r < onset, 1.0, cosine) * np.where(r < rc, 1.0, 0.0)


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


def evaluate_splines(r: np.ndarray) -> np.ndarray:
    spline = laplacian_spline()
    values = spline["values"]
    derivs = spline["derivs"]
    dx = spline["spacing"]
    x = np.clip(np.asarray(r, dtype=np.float64), 0.0, CUTOFF)
    n = np.floor(x / dx).astype(np.int64)
    n = np.clip(n, 0, values.shape[0] - 2)
    t = (x - n.astype(np.float64) * dx) / dx
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    pk = values[n]
    pk1 = values[n + 1]
    mk = derivs[n]
    mk1 = derivs[n + 1]
    return (
        h00[:, None] * pk
        + h10[:, None] * dx * mk
        + h01[:, None] * pk1
        + h11[:, None] * dx * mk1
    )


def _pairs(types, positions, cell, periodic):
    types = np.asarray(types, dtype=np.int32)
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    delta = displacements(pos, cell, periodic)
    rij = []
    ii = []
    jj = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dr = delta[i, j]
            r = float(np.linalg.norm(dr))
            if r <= CUTOFF and r > 1e-14:
                rij.append(dr)
                ii.append(i)
                jj.append(j)
    if not rij:
        return types, n, np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return types, n, np.stack(rij), np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64)


def _power_spectrum(expansion: Sequence[np.ndarray]) -> np.ndarray:
    parts = []
    for tensor in expansion:
        n, m, nrad, nspec = tensor.shape
        flat = tensor.reshape(n, m, nrad * nspec)
        parts.append(np.einsum("smn,smN->snN", flat, flat).reshape(n, -1))
    return np.concatenate(parts, axis=1)


def soap_features(types: Sequence[int], positions, cell, periodic) -> np.ndarray:
    """SOAP from dumped Laplacian spline + sphericart (C++ twin)."""
    types, n, rij, ii, jj = _pairs(types, positions, cell, periodic)
    size = SOAP_SIZE
    if rij.shape[0] == 0:
        return np.zeros((n, size))
    radial = evaluate_splines(np.linalg.norm(rij, axis=-1))
    fc = shifted_cosine(np.linalg.norm(rij, axis=-1))
    radial = radial * fc[:, None]
    ylm = _SPHERICART.compute(rij)
    onehot = np.zeros((rij.shape[0], N_SPECIES))
    for p, j in enumerate(jj):
        onehot[p, SPECIES_INDEX[int(types[j])]] = 1.0
    expansion = []
    n_off = 0
    y_off = 0
    for ell, nrad in enumerate(N_PER_L):
        m = 2 * ell + 1
        R = radial[:, n_off : n_off + nrad]
        S = ylm[:, y_off : y_off + m]
        n_off += nrad
        y_off += m
        contrib = np.einsum("pn,pm,pc->pmnc", R, S, onehot)
        block = np.zeros((n, m, nrad, N_SPECIES))
        np.add.at(block, ii, contrib)
        expansion.append(block)
    return _power_spectrum(expansion)


def soap_features_spex(types: Sequence[int], positions, cell, periodic) -> np.ndarray:
    """SOAP from torch-spex ``SphericalExpansion`` (same stack as metatrain)."""
    types, n, rij, ii, jj = _pairs(types, positions, cell, periodic)
    if rij.shape[0] == 0:
        return np.zeros((n, SOAP_SIZE))
    expansion = _spex_calculator().forward(
        torch.tensor(rij, dtype=torch.float64),
        torch.tensor(ii, dtype=torch.int64),
        torch.tensor(jj, dtype=torch.int64),
        torch.tensor(types, dtype=torch.int32),
    )
    parts = []
    for tensor in expansion:
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])
        n_prop = int(tensor.shape[-1] ** 2)
        parts.append(torch.einsum("smn,smN->snN", tensor, tensor).reshape(tensor.shape[0], n_prop))
    return torch.cat(parts, dim=1).detach().cpu().numpy()


def soap_features_metatrain(types: Sequence[int], positions, cell, periodic) -> np.ndarray:
    """SOAP from metatrain ``SoapPowerSpectrum`` (legacy TensorMap layout)."""
    from metatrain.soap_bpnn.modules.power_spectrum import SoapPowerSpectrum

    types, n, rij, ii, jj = _pairs(types, positions, cell, periodic)
    out = np.zeros((n, SOAP_SIZE))
    if rij.shape[0] == 0:
        return out
    calculator = SoapPowerSpectrum(**_spex_spec())
    feat = calculator(
        torch.tensor(rij, dtype=torch.float64),
        torch.tensor(ii, dtype=torch.int64),
        torch.tensor(jj, dtype=torch.int64),
        torch.tensor(types, dtype=torch.int32),
        torch.zeros(n, dtype=torch.int32),
        torch.arange(n, dtype=torch.int32),
    )
    for block in feat.blocks():
        atoms = block.samples.values[:, 1].cpu().numpy()
        values = block.values.detach().cpu().numpy()
        for atom, row in zip(atoms, values):
            out[int(atom)] = row
    return out


def _spex_spec() -> dict:
    return {
        "cutoff": CUTOFF,
        "max_angular": MAX_ANGULAR,
        "radial": {"LaplacianEigenstates": {"max_radial": MAX_RADIAL}},
        "angular": "SphericalHarmonics",
        "species": {"Orthogonal": {"species": list(SPECIES)}},
        "cutoff_function": {"ShiftedCosine": {"width": WIDTH}},
    }


_SPEX = None


def _spex_calculator() -> SphericalExpansion:
    global _SPEX
    if _SPEX is None:
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            _SPEX = SphericalExpansion(**_spex_spec())
        finally:
            torch.set_default_dtype(prev)
    return _SPEX


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
        spline = laplacian_spline()
        w1 = np.stack([weights[z]["w1"] for z in SPECIES], axis=0)
        w2 = np.stack([weights[z]["w2"] for z in SPECIES], axis=0)
        self.register_buffer("w1", torch.tensor(w1, dtype=torch.float64))
        self.register_buffer("w2", torch.tensor(w2, dtype=torch.float64))
        self.register_buffer("spline_values", torch.tensor(spline["values"], dtype=torch.float64))
        self.register_buffer("spline_derivs", torch.tensor(spline["derivs"], dtype=torch.float64))
        self.register_buffer(
            "spline_spacing", torch.tensor(spline["spacing"], dtype=torch.float64)
        )
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
        onset = self.cutoff - self.width
        pi = torch.tensor(3.141592653589793, dtype=r.dtype, device=r.device)
        cosine = 0.5 * (1.0 + torch.cos(pi * (r - onset) / self.width))
        ones = torch.ones_like(r)
        zeros = torch.zeros_like(r)
        return torch.where(r < onset, ones, cosine) * torch.where(r < self.cutoff, ones, zeros)

    def _spline(self, r: torch.Tensor) -> torch.Tensor:
        x = r.clamp(0.0, self.cutoff)
        dx = self.spline_spacing
        n = torch.floor(x / dx).to(dtype=torch.long)
        n = torch.clamp(n, 0, self.spline_values.shape[0] - 2)
        t = (x - n.to(dtype=x.dtype) * dx) / dx
        t2 = t * t
        t3 = t2 * t
        h00 = (2.0 * t3 - 3.0 * t2 + 1.0).unsqueeze(-1)
        h10 = (t3 - 2.0 * t2 + t).unsqueeze(-1)
        h01 = (-2.0 * t3 + 3.0 * t2).unsqueeze(-1)
        h11 = (t3 - t2).unsqueeze(-1)
        pk = self.spline_values[n]
        pk1 = self.spline_values[n + 1]
        mk = self.spline_derivs[n]
        mk1 = self.spline_derivs[n + 1]
        return h00 * pk + h10 * dx * mk + h01 * pk1 + h11 * dx * mk1

    def _ylm(self, delta: torch.Tensor, dist: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((), dtype=delta.dtype, device=delta.device)
        r_safe = torch.where(valid, dist, ones)
        inv = 1.0 / r_safe
        y00 = torch.full_like(dist, 0.28209479177387814)
        scale = torch.tensor(0.4886025119029199, dtype=delta.dtype, device=delta.device)
        y1m = scale * delta[..., 1] * inv
        y10 = scale * delta[..., 2] * inv
        y11 = scale * delta[..., 0] * inv
        ylm = torch.stack([y00, y1m, y10, y11], dim=-1)
        return torch.where(valid.unsqueeze(-1), ylm, torch.zeros_like(ylm))

    def _soap(self, types: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor):
        n = positions.shape[0]
        delta = self._min_image(positions.unsqueeze(0) - positions.unsqueeze(1), cell, pbc)
        dist = torch.linalg.norm(delta, dim=-1)
        eye = torch.eye(n, dtype=torch.bool, device=positions.device)
        valid = (~eye) & (dist <= self.cutoff) & (dist > 1e-14)
        fc = self._shifted_cosine(dist) * valid.to(dtype=dist.dtype)
        radial = self._spline(dist) * fc.unsqueeze(-1)
        ylm = self._ylm(delta, dist, valid)
        onehot = torch.stack(
            [
                (types == 1).to(dtype=dist.dtype),
                (types == 6).to(dtype=dist.dtype),
                (types == 8).to(dtype=dist.dtype),
            ],
            dim=-1,
        )
        c0 = torch.einsum("ijn,ijm,js->imnc", radial[..., 0:3], ylm[..., 0:1], onehot)
        t0 = c0.reshape(n, 1, 9)
        p0 = torch.einsum("smn,smN->snN", t0, t0).reshape(n, -1)
        c1 = torch.einsum("ijn,ijm,js->imnc", radial[..., 3:5], ylm[..., 1:4], onehot)
        t1 = c1.reshape(n, 3, 6)
        p1 = torch.einsum("smn,smN->snN", t1, t1).reshape(n, -1)
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


def _fmt_rows(rows: np.ndarray, indent: int) -> str:
    pad = " " * indent
    lines = []
    for row in np.atleast_2d(rows):
        vals = ", ".join(f"{float(x):.16g}" for x in np.ravel(row))
        lines.append(f"{pad}{{{vals}}},")
    return "\n".join(lines)


def emit_cpp_weights(path: Union[str, Path]) -> None:
    spline = laplacian_spline()
    weights = default_weights()
    values = spline["values"]
    derivs = spline["derivs"]
    lines = [
        "// Generated by examples/_bpnn.py from torch-spex LaplacianEigenstates.",
        "// Do not edit by hand.",
        "#pragma once",
        f"constexpr int kBpnnSoapSize = {SOAP_SIZE};",
        f"constexpr int kBpnnHidden = {HIDDEN};",
        f"constexpr int kBpnnNSpecies = {N_SPECIES};",
        f"constexpr int kBpnnMaxAngular = {MAX_ANGULAR};",
        f"constexpr int kBpnnNRadialFeat = {N_RADIAL_FEAT};",
        f"constexpr int kBpnnSplineN = {values.shape[0]};",
        f"constexpr int kBpnnNPerL[{len(N_PER_L)}] = {{{', '.join(str(n) for n in N_PER_L)}}};",
        f"constexpr double kBpnnSplineSpacing = {spline['spacing']:.16g};",
        f"constexpr double kBpnnSplineValues[{values.shape[0]}][{values.shape[1]}] = {{",
        _fmt_rows(values, 4),
        "};",
        f"constexpr double kBpnnSplineDerivs[{derivs.shape[0]}][{derivs.shape[1]}] = {{",
        _fmt_rows(derivs, 4),
        "};",
        f"constexpr double kBpnnW1[3][{HIDDEN}][{SOAP_SIZE}] = {{",
    ]
    for z in SPECIES:
        lines.append("    {")
        lines.append(_fmt_rows(weights[z]["w1"], 8))
        lines.append("    },")
    lines.append("};")
    lines.append(f"constexpr double kBpnnW2[3][{HIDDEN}] = {{")
    for z in SPECIES:
        vals = ", ".join(f"{x:.16g}" for x in weights[z]["w2"])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    Path(path).write_text("\n".join(lines) + "\n")


def _check_systems(export_path: Optional[Path] = None) -> None:
    laplacian_spline()
    wrapper = wrap()
    pt = None
    if export_path is not None:
        pt = export_bpnn(str(export_path))
    print(
        f"{'system':<12} {'N':>3} {'E_np':>12} {'E_torch':>12} {'Δspex':>10} "
        f"{'Δmtt':>10} {'ΔF_FD':>10}  pbc"
    )
    for name, system in SYSTEMS.items():
        types, pos, cell, periodic = (
            system["types"],
            system["positions"],
            system["cell"],
            system["periodic"],
        )
        feat_np = soap_features(types, pos, cell, periodic)
        feat_spex = soap_features_spex(types, pos, cell, periodic)
        d_spex = float(np.max(np.abs(feat_np - feat_spex)))
        assert d_spex < 1e-10, (name, d_spex)
        d_mtt = float("nan")
        try:
            feat_mtt = soap_features_metatrain(types, pos, cell, periodic)
            d_mtt = float(np.max(np.abs(feat_np - feat_mtt)))
            assert d_mtt < 1e-10, (name, d_mtt)
        except ImportError:
            pass
        e_np = evaluate_numpy(system)
        e_t, f_t = evaluate_torch(system, wrapper)
        f_fd = finite_difference_forces(
            lambda p, system=system: evaluate_numpy({**system, "positions": p}),
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
        mtt_s = f"{d_mtt:10.3e}" if d_mtt == d_mtt else f"{'n/a':>10}"
        print(
            f"{name:<12} {len(system['types']):>3} {e_np:12.6e} {e_t:12.6e} "
            f"{d_spex:10.3e} {mtt_s} {d_f:10.3e}  {system['periodic']}"
        )
    print("SOAP matches torch-spex/metatrain; torch energy matches numpy; forces match FD")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None)
    args = parser.parse_args()
    if args.emit is not None:
        emit_cpp_weights(args.emit)
        print(f"wrote {args.emit}")
    _check_systems(args.export)
