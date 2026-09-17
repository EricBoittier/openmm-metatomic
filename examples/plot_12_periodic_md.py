"""
Periodic MD
===========

A cubic water box through native ``MetatomicForce``:

* **harmonic-nl** (core, and TorchScript ``pairlist.pt`` when present) — the
  independent-atom well plus a vesin pair list that does not change the
  energy. Langevin 300 K, PBC on.
* **PET-MAD-XS** — eight waters, the first condensed-phase run of the
  vendored extra-small checkpoint.

Traces are potential, kinetic, total energy, and instantaneous temperature.
"""

import matplotlib.pyplot as plt
import numpy as np

from _native import (
    make_system,
    pairlist_pt,
    petmad_path,
    preferred_platform,
    run_nve,
    run_nvt,
    water_box,
)

platform = preferred_platform()
print(f"platform {platform.getName()}")
n_mol = 8
types, positions, box = water_box(n_mol)
print(f"water box  molecules={n_mol}  atoms={len(types)}  L={box[0, 0]:.3f} nm")

jobs = [("harmonic-nl/core", "harmonic-nl", "core")]
nl_pt = pairlist_pt()
if nl_pt is not None:
    jobs.append(("harmonic-nl/torch", str(nl_pt), "torch"))
jobs.append(("PET-MAD-XS", str(petmad_path()), "torch"))

nve_steps = 40
nvt_steps = 30
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
dt = 0.5

print(f"{'model':<22} {'NVE drift':>12} {'NVT T':>8}  E0")
for title, path, backend in jobs:
    system = make_system(
        types, path, backend=backend, periodic=True, box_nm=box
    )
    nve = run_nve(system, positions, platform, nve_steps, box=box)
    system_nvt = make_system(
        types, path, backend=backend, periodic=True, box_nm=box
    )
    nvt = run_nvt(system_nvt, positions, platform, nvt_steps, box=box)
    drift = nve[-1, 2] - nve[0, 2]
    print(
        f"{title:<22} {drift:12.4f} {nvt[:, 3].mean():8.1f}  {nve[0, 0]:.4f}"
    )
    t_nve = np.arange(len(nve)) * dt
    t_nvt = np.arange(len(nvt)) * dt
    axes[0, 0].plot(t_nve, nve[:, 2] - nve[0, 2], label=title)
    axes[0, 1].plot(t_nve, nve[:, 3], label=title)
    axes[1, 0].plot(t_nvt, nvt[:, 0], label=title)
    axes[1, 1].plot(t_nvt, nvt[:, 3], label=title)

axes[0, 0].set_title("NVE total-energy drift")
axes[0, 0].set_ylabel(r"$\Delta E$ / kJ mol$^{-1}$")
axes[0, 1].set_title("NVE temperature")
axes[0, 1].set_ylabel("T / K")
axes[1, 0].set_title("NVT potential")
axes[1, 0].set_ylabel("E / kJ mol$^{-1}$")
axes[1, 1].set_title("NVT temperature")
axes[1, 1].set_ylabel("T / K")
for ax in axes.ravel():
    ax.set_xlabel("t / fs")
    ax.legend(fontsize=7)
fig.suptitle(f"{n_mol} waters, {platform.getName()} platform")
fig.tight_layout()
