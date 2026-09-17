"""
Periodic MD
===========

Cubic water boxes through native ``MetatomicForce``:

* **harmonic-nl** (core, and TorchScript ``pairlist.pt`` when present) —
  1,000 waters (3,000 atoms). The independent-atom well plus a vesin pair
  list that does not change the energy. Langevin 300 K, PBC on.
* **PET-MAD-XS** — 32 waters (96 atoms), a condensed-phase run of the
  vendored extra-small checkpoint. 96 waters is in the settings matrix
  (``plot_13``), not the conservation traces here.
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
    save_figure,
    water_box,
)

platform = preferred_platform()
print(f"platform {platform.getName()}")

HARMONIC_MOL = 1000
PETMAD_MOL = 32

h_types, h_pos, h_box = water_box(HARMONIC_MOL)
print(
    f"harmonic-nl box  molecules={HARMONIC_MOL}  atoms={len(h_types)}  "
    f"L={h_box[0, 0]:.3f} nm"
)
p_types, p_pos, p_box = water_box(PETMAD_MOL)
print(
    f"PET-MAD box  molecules={PETMAD_MOL}  atoms={len(p_types)}  "
    f"L={p_box[0, 0]:.3f} nm"
)

jobs = [
    ("harmonic-nl/core", "harmonic-nl", "core", h_types, h_pos, h_box, 40, 30),
]
nl_pt = pairlist_pt()
if nl_pt is not None:
    jobs.append(
        ("harmonic-nl/torch", str(nl_pt), "torch", h_types, h_pos, h_box, 40, 30)
    )
jobs.append(
    ("PET-MAD-XS", str(petmad_path()), "torch", p_types, p_pos, p_box, 20, 15)
)

fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
dt = 0.5

print(f"{'model':<22} {'N':>8} {'NVE drift':>12} {'NVT T':>8}  E0")
for title, path, backend, types, positions, box, nve_steps, nvt_steps in jobs:
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
        f"{title:<22} {len(types):8d} {drift:12.4e} {nvt[:, 3].mean():8.1f}  "
        f"{nve[0, 0]:.4f}"
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
fig.suptitle(
    f"{HARMONIC_MOL} waters harmonic-nl / {PETMAD_MOL} waters PET-MAD-XS, "
    f"{platform.getName()}"
)
fig.tight_layout()
save_figure(fig, __file__)
