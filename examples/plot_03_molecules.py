"""
Methane and CO2
===============

Two further vacuum systems: tetrahedral methane (C + 4 H) and linear carbon
dioxide. The same well stiffness :math:`k = 1` kJ/mol/nm$^2$ is used, so the
force on each atom is independent of the others. That is a stand-in for a
conservative Metatomic energy, not a physical force field.
"""

import matplotlib.pyplot as plt
import numpy as np

from _harmonic import SYSTEMS, analytic, evaluate_core, evaluate_torch

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
for ax, name in zip(axes, ("methane", "co2")):
    system = SYSTEMS[name]
    energy_a, forces_a = analytic(system)
    energy_c, forces_c = evaluate_core(system)
    energy_t, forces_t = evaluate_torch(system)
    rms = float(np.sqrt(np.mean((forces_t - forces_a) ** 2)))
    print(
        f"{name}: E_core={energy_c:.6e}  E_torch={energy_t:.6e}  "
        f"ΔE={abs(energy_c - energy_a):.3e}  RMS ΔF={rms:.3e}"
    )
    del forces_c
    rest = np.asarray(system["rest"])
    pos = np.asarray(system["positions"])
    ax.scatter(rest[:, 0], rest[:, 1], c="C0", label="rest")
    ax.scatter(pos[:, 0], pos[:, 1], c="C1", label="displaced")
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        forces_t[:, 0],
        forces_t[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.5,
        color="C1",
        width=0.008,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{system['title']}\nRMS ΔF = {rms:.1e}")
    ax.set_xlabel("x / nm")
    ax.set_ylabel("y / nm")
    ax.legend(loc="upper right")
fig.tight_layout()
