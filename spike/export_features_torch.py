"""Export a TorchScript model exercising the non-energy features of the plugin.

Everything it computes has a closed form so ``spike/test_features.cpp`` can check
the C++ side without a reference implementation (nm and kJ/mol throughout, no
unit conversion):

* ``energy`` = ``0.5 Σ|r_i|²`` + ``PAIR_ENERGY`` per pair inside ``CUTOFF_NM``
  + ``CHARGE_COEFF·charge`` + ``SPIN_COEFF·(spin - 1)``. The pair term makes the
  energy sensitive to which directions are periodic, and the two scalar inputs
  make ``charge`` / ``spin_multiplicity`` observable.
* ``non_conservative_force`` = ``(i + 1, 2, 3)``, a field with a non-zero net
  force, so an engine that forgets to remove the mean is easy to spot.
* ``energy_uncertainty`` = ``0.01·(i + 1)`` eV per atom.
"""

import sys
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
PAIR_ENERGY = 2.0
CHARGE_COEFF = 10.0
SPIN_COEFF = 100.0


class FeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._nl = NeighborListOptions(CUTOFF_NM, False, True)
        self.k = K
        self.pair_energy = PAIR_ENERGY
        self.charge_coeff = CHARGE_COEFF
        self.spin_coeff = SPIN_COEFF

    def requested_neighbor_lists(self):
        return [self._nl]

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return {
            "charge": ModelOutput(unit="e", sample_kind="system"),
            "spin_multiplicity": ModelOutput(unit="", sample_kind="system"),
        }

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device
        result: Dict[str, TensorMap] = {}

        if "energy" in outputs:
            energy = torch.zeros((len(systems), 1), dtype=dtype, device=device)
            for i, system in enumerate(systems):
                neighbors = system.get_neighbor_list(self._nl)
                n_pairs = float(neighbors.values.shape[0])
                charge = system.get_data("charge").block().values.reshape(())
                spin = system.get_data("spin_multiplicity").block().values.reshape(())
                energy[i] = (
                    0.5 * self.k * (system.positions**2).sum()
                    + self.pair_energy * n_pairs
                    + self.charge_coeff * charge
                    + self.spin_coeff * (spin - 1.0)
                )
            result["energy"] = TensorMap(
                Labels("_", torch.tensor([[0]], device=device)),
                [
                    TensorBlock(
                        values=energy,
                        samples=Labels(
                            "system",
                            torch.arange(len(systems), device=device).reshape(-1, 1),
                        ),
                        components=[],
                        properties=Labels("energy", torch.tensor([[0]], device=device)),
                    )
                ],
            )

        if "non_conservative_force" in outputs:
            forces = []
            for system in systems:
                n = len(system)
                forces.append(
                    torch.stack(
                        [
                            torch.arange(1, n + 1, dtype=dtype, device=device),
                            torch.full((n,), 2.0, dtype=dtype, device=device),
                            torch.full((n,), 3.0, dtype=dtype, device=device),
                        ],
                        dim=1,
                    ).reshape(n, 3, 1)
                )
            result["non_conservative_force"] = self._per_atom(
                systems,
                torch.cat(forces),
                [Labels("xyz", torch.tensor([[0], [1], [2]], device=device))],
                Labels("non_conservative_force", torch.tensor([[0]], device=device)),
            )

        if "energy_uncertainty" in outputs:
            uncertainty = []
            for system in systems:
                n = len(system)
                uncertainty.append(
                    0.01 * torch.arange(1, n + 1, dtype=dtype, device=device).reshape(n, 1)
                )
            result["energy_uncertainty"] = self._per_atom(
                systems,
                torch.cat(uncertainty),
                [],
                Labels("energy_uncertainty", torch.tensor([[0]], device=device)),
            )

        return result

    def _per_atom(
        self,
        systems: List[System],
        values: torch.Tensor,
        components: List[Labels],
        properties: Labels,
    ) -> TensorMap:
        device = systems[0].positions.device
        samples = []
        for i, system in enumerate(systems):
            n = len(system)
            samples.append(
                torch.stack(
                    [
                        torch.full((n,), i, dtype=torch.int32, device=device),
                        torch.arange(n, dtype=torch.int32, device=device),
                    ],
                    dim=1,
                )
            )
        block = TensorBlock(
            values=values,
            samples=Labels(["system", "atom"], torch.cat(samples)),
            components=components,
            properties=properties,
        )
        return TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "features.pt"
    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(unit="kJ/mol", sample_kind="system"),
            "non_conservative_force": ModelOutput(
                unit="kJ/mol/nm", sample_kind="atom"
            ),
            "energy_uncertainty": ModelOutput(unit="eV", sample_kind="atom"),
        },
        atomic_types=[1, 6, 8],
        interaction_range=CUTOFF_NM,
        length_unit="nm",
        # Everything in forward() follows systems[0].positions.device, so the
        # same checkpoint drives the CUDA parity test in spike/test_cuda.cpp.
        supported_devices=["cuda", "cpu"],
        dtype="float64",
    )
    AtomisticModel(
        FeatureModel().eval(), ModelMetadata(name="plugin-features"), capabilities
    ).save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
