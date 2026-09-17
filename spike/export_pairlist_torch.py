"""Export a TorchScript harmonic well that requests a neighbor list.

Same math and units as HarmonicModel/NeighborHarmonicModel in
HarmonicModel.h (E = 0.5 * sum ||r||^2, k=1, nm/kJ-mol): the pair list is
touched (summed with a zero coefficient) but does not change the energy, so
this has the same analytic solution as the plain harmonic well. Used to
validate the C++ plugin's torch-backend pair-list path (spike/test_pairlist.cpp).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

K = 1.0
CUTOFF_NM = 0.3


class NeighborHarmonic(torch.nn.Module):
    def __init__(self, k: float, n_atoms: int, cutoff: float):
        super().__init__()
        self.k = float(k)
        self.register_buffer("rest", torch.zeros((n_atoms, 3), dtype=torch.float64))
        self._nl = NeighborListOptions(cutoff, False, True)

    def requested_neighbor_lists(self):
        return [self._nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if list(outputs.keys()) != ["energy"]:
            raise ValueError("this model only computes energy")
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            neighbors = system.get_neighbor_list(self._nl)
            # Touch the pair list without perturbing the energy.
            energy[i] = neighbors.values.sum() * 0.0
            rest = self.rest.to(dtype=system.positions.dtype, device=system.positions.device)
            energy[i] = energy[i] + 0.5 * self.k * ((system.positions - rest) ** 2).sum()
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "pairlist.pt"
    n_atoms = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    model = NeighborHarmonic(K, n_atoms, CUTOFF_NM)
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
        atomic_types=[1, 6, 8],
        interaction_range=CUTOFF_NM,
        length_unit="nm",
        supported_devices=["cpu"],
        dtype="float64",
    )
    AtomisticModel(
        model.eval(), ModelMetadata(name="pairlist-neighbor-harmonic"), capabilities
    ).save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
