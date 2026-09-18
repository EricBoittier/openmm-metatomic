"""``.mta`` is a serialized metatomic System, not a model.

TorchScript models are ``.pt``. Core plugins are ``.so`` / ``.dylib`` /
``.dll``. This is the format ``metatomic.torch.save`` writes for a
structure (types, positions, cell), the same layout as
``0/system.mta`` inside a DiskDataset zip.
"""

from pathlib import Path

import pytest
import torch

pytest.importorskip("metatomic.torch")

import metatomic.torch as mta  # noqa: E402


def test_round_trip_system_mta(tmp_path: Path):
    system = mta.System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=torch.float64,
        ),
        cell=torch.eye(3, dtype=torch.float64) * 12.0,
        pbc=torch.tensor([True, True, True]),
    )
    path = tmp_path / "water.mta"
    mta.save(str(path), system)

    loaded = mta.load_system(str(path))
    assert torch.equal(loaded.types, system.types)
    assert torch.allclose(loaded.positions, system.positions)
    assert torch.allclose(loaded.cell, system.cell)
    assert torch.equal(loaded.pbc, system.pbc)
