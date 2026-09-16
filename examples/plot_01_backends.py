"""
Comparing backends
==================

The plugin evaluates Metatomic models through two backends:

* ``core`` — in-process ``HarmonicModel`` via ``metatomic::execute_model``
  (same well as :func:`_harmonic.evaluate_core`)
* ``torch`` — in-memory :class:`metatomic.torch.AtomisticModel`, or an exported
  ``.pt`` loaded with :func:`metatomic.torch.load_atomistic_model`

This example runs the same independent-atom harmonic well on water, methane,
CO2, an eight-atom carbon cluster, and periodic water. Energies and conservative
forces are checked against
:math:`E = \\tfrac{1}{2} k \\sum_i \\|r_i - r_{0,i}\\|^2` and against
finite-difference forces.
"""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from _harmonic import (  # noqa: E402
    SYSTEMS,
    analytic,
    evaluate_core,
    evaluate_pt,
    evaluate_torch,
    export_system,
    finite_difference_forces,
)


def energy_only(evaluate, system, positions):
    trial = dict(system)
    trial["positions"] = positions
    energy, _ = evaluate(trial)
    return energy


names = []
core_err = []
torch_err = []
pt_err = []
fd_err = []

header = (
    f"{'system':<12} {'N':>3} {'E_core':>12} {'E_torch':>12} {'E_pt':>12} "
    f"{'ΔF_core':>10} {'ΔF_torch':>10} {'ΔF_pt':>10} {'ΔF_FD':>10}  pbc"
)
print(header)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for name, system in SYSTEMS.items():
        energy_a, forces_a = analytic(system)
        energy_c, forces_c = evaluate_core(system)
        energy_t, forces_t = evaluate_torch(system)
        pt_path = export_system(system, str(tmp / f"{name}.pt"))
        energy_p, forces_p = evaluate_pt(system, pt_path)
        forces_fd = finite_difference_forces(
            lambda pos, system=system: energy_only(evaluate_torch, system, pos),
            np.asarray(system["positions"], dtype=np.float64),
        )
        d_core = float(np.max(np.abs(forces_c - forces_a)))
        d_torch = float(np.max(np.abs(forces_t - forces_a)))
        d_pt = float(np.max(np.abs(forces_p - forces_a)))
        d_fd = float(np.max(np.abs(forces_t - forces_fd)))
        assert np.allclose([energy_c, energy_t, energy_p], energy_a, rtol=1e-6, atol=1e-10)
        assert max(d_core, d_torch, d_pt) < 1e-8
        names.append(name)
        core_err.append(max(d_core, 1e-16))
        torch_err.append(max(d_torch, 1e-16))
        pt_err.append(max(d_pt, 1e-16))
        fd_err.append(max(d_fd, 1e-16))
        print(
            f"{name:<12} {len(system['types']):>3} {energy_c:12.6e} {energy_t:12.6e} "
            f"{energy_p:12.6e} {d_core:10.3e} {d_torch:10.3e} {d_pt:10.3e} "
            f"{d_fd:10.3e}  {system['periodic']}"
        )

fig, ax = plt.subplots(figsize=(8.0, 3.8))
x = np.arange(len(names))
width = 0.2
ax.bar(x - 1.5 * width, core_err, width, label="core vs analytic")
ax.bar(x - 0.5 * width, torch_err, width, label="torch vs analytic")
ax.bar(x + 0.5 * width, pt_err, width, label="exported .pt vs analytic")
ax.bar(x + 1.5 * width, fd_err, width, label="torch vs FD")
ax.set_xticks(x, names)
ax.set_ylabel("max |ΔF| / kJ mol$^{-1}$ nm$^{-1}$")
ax.set_title("Conservative forces: core, torch, exported .pt")
ax.legend(ncols=2, fontsize=8)
ax.set_yscale("log")
fig.tight_layout()
