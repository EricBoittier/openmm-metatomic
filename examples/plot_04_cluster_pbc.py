"""
Carbon cluster and periodic water
=================================

The core evaluator accepts any number of atoms of the supported types, and a
periodic cell. The harmonic well does not use pair lists, so switching on PBC
must not change the energy — it only changes how the ``metatomic::System`` is
built (non-zero cell, ``pbc = true``).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonic import SYSTEMS, analytic, evaluate_torch  # noqa: E402

cluster = SYSTEMS["carbon8"]
vacuum = SYSTEMS["water"]
pbc = SYSTEMS["water_pbc"]

e_c, f_c = analytic(cluster)
et_c, ft_c = evaluate_torch(cluster)
e_v, _ = analytic(vacuum)
e_p, _ = analytic(pbc)
et_v, _ = evaluate_torch(vacuum)
et_p, _ = evaluate_torch(pbc)

print(f"carbon8   N={len(cluster['types'])}  E_torch={et_c:.6e}  ΔE={abs(et_c - e_c):.3e}")
print(f"water     E_torch={et_v:.6e}")
print(f"water_pbc E_torch={et_p:.6e}  ΔE vs vacuum={abs(et_p - et_v):.3e}")
assert abs(et_p - et_v) < 1e-12
assert abs(et_c - e_c) < 1e-10

pos = np.asarray(cluster["positions"])
fig = plt.figure(figsize=(7.2, 3.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="k")
ax.quiver(
    pos[:, 0], pos[:, 1], pos[:, 2],
    ft_c[:, 0], ft_c[:, 1], ft_c[:, 2],
    length=0.4, normalize=False, color="C1",
)
ax.set_title("carbon8 forces")
ax.set_xlabel("x / nm")

ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(["water", "water_pbc", "carbon8"], [et_v, et_p, et_c], color=["C0", "C0", "C2"])
ax2.set_ylabel("E / kJ mol$^{-1}$")
ax2.set_title("PBC does not change the well")
fig.tight_layout()
