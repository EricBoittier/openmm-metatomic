"""Export a TorchScript harmonic well matching the C++ M0 spike."""

from typing import Dict, List, Optional

import argparse
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import AtomisticModel, ModelCapabilities, ModelMetadata, ModelOutput, System

K = 1.0
R0 = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.05],
    ],
    dtype=torch.float64,
)


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


def export(path: str) -> None:
    model = Harmonic(K, R0)
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
        atomic_types=[1, 6, 8],
        interaction_range=0.0,
        length_unit="nm",
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name="openmm-metatomic-harmonic-torchscript",
        description="M0 spike TorchScript harmonic well (back-compat path)",
    )
    AtomisticModel(model.eval(), metadata, capabilities).save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="harmonic.pt")
    export(parser.parse_args().path)
