"""
Water
=====

A displaced water geometry (types H, H, O) in OpenMM units: nanometers and
kJ/mol. The harmonic well is centered on a TIP3P-like equilibrium. Forces from
the TorchScript backend are compared with :math:`F = -k(r - r_0)` and with a
central finite difference.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonic import SYSTEMS, analytic, evaluate_torch, finite_difference_forces  # noqa: E402

system = SYSTEMS["water"]
energy_a, forces_a = analytic(system)
energy_t, forces_t = evaluate_torch(system)
forces_fd = finite_difference_forces(
    lambda pos: evaluate_torch({**system, "positions": pos})[0],
    np.asarray(system["positions"], dtype=np.float64),
)

print(f"energy analytic {energy_a:.8e} kJ/mol")
print(f"energy torch    {energy_t:.8e} kJ/mol")
print("forces / kJ/mol/nm")
print(" atom      analytic            torch               FD")
for i, (fa, ft, fd) in enumerate(zip(forces_a, forces_t, forces_fd)):
    print(f" {i:4d}  {fa}  {ft}  {fd}")

fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=True)
labels = ["x", "y", "z"]
atoms = np.arange(len(system["types"]))
for c, ax in enumerate(axes):
    ax.plot(atoms, forces_a[:, c], "o", label="analytic")
    ax.plot(atoms, forces_t[:, c], "x", label="torch")
    ax.plot(atoms, forces_fd[:, c], "+", label="FD")
    ax.set_title(labels[c])
    ax.set_xlabel("atom")
    ax.set_xticks(atoms)
axes[0].set_ylabel("F / kJ mol$^{-1}$ nm$^{-1}$")
axes[0].legend()
fig.suptitle(system["title"])
fig.tight_layout()
