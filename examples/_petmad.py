"""Download and convert PET-MAD (lab-cosmo/upet) for the gallery."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
    pick_device,
    pick_output,
    unit_conversion_factor,
)

from _openmm import repo_root

HF_REPO = "lab-cosmo/upet"
CKPT_REL = "models/pet-mad-xs-v1.5.0.ckpt"
SIZE = "xs"
VERSION = "1.5.0"

# TIP3P-like water in nanometers, same geometry as etc/openmm-metatomic-ase
WATER_NM = np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.096], [0.0, 0.093, -0.024]],
    dtype=np.float64,
)
WATER_NUMBERS = [8, 1, 1]
WATER_SYMBOLS = ["O", "H", "H"]


def models_dir() -> Path:
    path = repo_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path() -> Path:
    return models_dir() / "hf-upet" / CKPT_REL


def converted_pt() -> Path:
    return models_dir() / f"pet-mad-{SIZE}-v{VERSION}.converted.pt"


def official_pt() -> Path:
    return models_dir() / f"pet-mad-{SIZE}-v{VERSION}.pt"


def download_checkpoint() -> Path:
    dest = checkpoint_path()
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf = shutil.which("hf") or "hf"
    subprocess.run(
        [hf, "download", HF_REPO, CKPT_REL, "--local-dir", str(models_dir() / "hf-upet")],
        check=True,
    )
    if not dest.is_file():
        raise FileNotFoundError(f"Hugging Face download did not produce {dest}")
    return dest


def _mtt_candidates() -> List[Path]:
    env = os.environ.get("OPENMM_METATOMIC_MTT")
    found = []
    if env:
        found.append(Path(env))
    which = shutil.which("mtt")
    if which:
        found.append(Path(which))
    found.append(
        Path(
            "/home/ericb/metawork/atomistic-cookbook/examples/"
            "metatomic-hourglass/.hourglass-env/bin/mtt"
        )
    )
    seen = set()
    out = []
    for path in found:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def convert_checkpoint(ckpt: Path, output: Path) -> Path:
    if output.is_file():
        return output
    extensions = models_dir() / "pet-mad-extensions"
    errors = []
    for mtt in _mtt_candidates():
        if not mtt.exists():
            continue
        proc = subprocess.run(
            [str(mtt), "export", str(ckpt), "-o", str(output), "-e", str(extensions)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and output.is_file():
            print(f"converted with {mtt}")
            return output
        errors.append(f"{mtt}: {proc.stderr[-800:] or proc.stdout[-800:]}")
    raise RuntimeError(
        "mtt export failed for every candidate. The vendored metatrain "
        "2026.5.dev mis-classifies PET-MAD's non-conservative force head; "
        "use metatrain==2026.3.1 or 2026.4 (see models/README.md).\n"
        + "\n".join(errors)
    )


def save_official_pt(output: Path) -> Path:
    """``upet.save_upet`` — the packaged Hugging Face → TorchScript path."""
    if output.is_file():
        return output
    try:
        import upet

        upet.save_upet(model="pet-mad", size=SIZE, version=VERSION, output=str(output))
        return output
    except ImportError:
        pass
    hourglass = Path(
        "/home/ericb/metawork/atomistic-cookbook/examples/"
        "metatomic-hourglass/.hourglass-env/bin/python"
    )
    if hourglass.is_file():
        subprocess.run(
            [
                str(hourglass),
                "-c",
                "import upet; upet.save_upet("
                f"model='pet-mad', size={SIZE!r}, version={VERSION!r}, "
                f"output={str(output)!r})",
            ],
            check=True,
        )
        return output
    raise ImportError(
        "upet is not installed; pip install upet, or set OPENMM_METATOMIC_MTT "
        "and rely on the converted .pt only"
    )


def ensure_models() -> Tuple[Path, Path, Path]:
    ckpt = download_checkpoint()
    converted = convert_checkpoint(ckpt, converted_pt())
    try:
        official = save_official_pt(official_pt())
    except Exception as exc:
        print(f"official upet.save_upet skipped: {exc}")
        official = converted
    return ckpt, converted, official


_LOADED = {}


def evaluate_exported(
    model_path: str,
    numbers: Sequence[int],
    positions_angstrom,
    device: str = "cpu",
    periodic: bool = False,
    cell_angstrom: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Energy (kJ/mol) and forces (kJ/mol/nm) with vesin neighbor lists."""
    import vesin.metatomic

    key = (model_path, device, bool(periodic))
    cached = _LOADED.get(key)
    if cached is None:
        model = load_atomistic_model(model_path)
        capabilities = model.capabilities()
        torch_device = torch.device(pick_device(capabilities.supported_devices, device))
        dtype = getattr(torch, capabilities.dtype)
        model = model.to(device=torch_device)
        energy_key = pick_output("energy", capabilities.outputs, None)
        neighbor_lists = [
            vesin.metatomic.NeighborList(
                options=nl,
                length_unit="angstrom",
                check_consistency=False,
                skin=2.0,
            )
            for nl in model.requested_neighbor_lists()
        ]
        cached = (model, torch_device, dtype, energy_key, neighbor_lists)
        _LOADED[key] = cached
    model, torch_device, dtype, energy_key, neighbor_lists = cached
    types = torch.tensor(list(numbers), dtype=torch.int32, device=torch_device)
    pos = torch.tensor(
        np.asarray(positions_angstrom, dtype=np.float64),
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    if periodic and cell_angstrom is not None:
        cell = torch.tensor(cell_angstrom, dtype=dtype, device=torch_device)
    else:
        cell = torch.zeros((3, 3), dtype=dtype, device=torch_device)
    pbc = torch.tensor([periodic] * 3, dtype=torch.bool, device=torch_device)
    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={energy_key: ModelOutput(unit="eV", sample_kind="system")},
    )
    energy_scale = float(unit_conversion_factor("eV", "kJ/mol"))
    system = System(types, pos, cell, pbc)
    if neighbor_lists:
        work = system if system.device.type in ("cpu", "cuda") else system.to(device="cpu")
        for neighbors in neighbor_lists:
            neighbors.add_neighbor_list(systems=[work], copy=False)
        if work.device != torch_device:
            work = work.to(device=torch_device)
        system = work
    energy = model([system], options, False)[energy_key].block().values.sum()
    energy.backward()
    forces = (-system.positions.grad * energy_scale * 10.0).detach().cpu().numpy()
    return float(energy.detach()) * energy_scale, np.asarray(forces, dtype=np.float64)
