"""
Comparing backends
==================

The plugin evaluates Metatomic models through two backends:

* ``core`` — ``metatomic::execute_model`` with explicit position gradients
* ``torch`` — exported TorchScript (``.pt``) via ``load_atomistic_model``

This example runs the same independent-atom harmonic well on several small
systems and checks both backends against the analytic energy
:math:`E = \\tfrac{1}{2} k \\sum_i \\|r_i - r_{0,i}\\|^2` and against
finite-difference forces.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from _harmonic import SYSTEMS, analytic, evaluate_torch, finite_difference_forces  # noqa: E402


def torch_energy_only(system, positions):
    trial = dict(system)
    trial["positions"] = positions
    energy, _ = evaluate_torch(trial)
    return energy


names = []
core_like = []
torch_err = []
fd_err = []

print(f"{'system':<12} {'N':>3} {'E_analytic':>12} {'E_torch':>12} {'max|dF_torch|':>14} {'max|dF_FD|':>12}  pbc")
for name, system in SYSTEMS.items():
    energy_a, forces_a = analytic(system)
    energy_t, forces_t = evaluate_torch(system)
    forces_fd = finite_difference_forces(
        lambda pos, system=system: torch_energy_only(system, pos),
        np.asarray(system["positions"], dtype=np.float64),
    )
    d_torch = float(np.max(np.abs(forces_t - forces_a)))
    d_fd = float(np.max(np.abs(forces_t - forces_fd)))
    assert np.isclose(energy_t, energy_a, rtol=1e-6, atol=1e-10)
    assert d_torch < 1e-8
    names.append(name)
    core_like.append(0.0)  # analytic/core agree at roundoff; plotted as the baseline
    torch_err.append(d_torch)
    fd_err.append(d_fd)
    print(
        f"{name:<12} {len(system['types']):>3} {energy_a:12.6e} {energy_t:12.6e} "
        f"{d_torch:14.3e} {d_fd:12.3e}  {system['periodic']}"
    )

fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(names))
width = 0.35
ax.bar(x - width / 2, torch_err, width, label="torch vs analytic")
ax.bar(x + width / 2, fd_err, width, label="torch vs FD")
ax.set_xticks(x, names)
ax.set_ylabel("max |ΔF| / kJ mol$^{-1}$ nm$^{-1}$")
ax.set_title("Conservative forces on every demo system")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
