"""
SOAP-BPNN: core vs TorchScript
==============================

Metatrain's Behler–Parrinello architecture is SOAP plus a per-species MLP
(``soap_bpnn``, legacy path: orthogonal species, SiLU, no bias), not the
original ACSF G2/G4 functions. This example runs that stack in

* **core** — ``BpnnModel`` through ``metatomic::execute_model``
* **torch** — the matching ``SoapBpnn`` module, including an exported ``.pt``

on the same water, methane, CO2, carbon cube, and periodic water systems.
Energies must agree; forces are checked against finite differences.
"""

import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bpnn import (
    SYSTEMS,
    evaluate_numpy,
    evaluate_pt,
    evaluate_torch,
    export_bpnn,
    wrap,
)
from _harmonic import finite_difference_forces
from _openmm import repo_root


wrapper = wrap()
core_bin = repo_root() / "build" / "openmm-metatomic-bpnn"

print(
    f"{'system':<12} {'N':>3} {'E_np':>12} {'E_torch':>12} {'ΔE':>10} {'ΔF_FD':>10}  pbc"
)

names = []
dE = []
dF = []

with tempfile.TemporaryDirectory() as tmp:
    pt = export_bpnn(str(Path(tmp) / "soap-bpnn.pt"))
    if core_bin.is_file():
        result = subprocess.run([str(core_bin), pt], capture_output=True, text=True, check=False)
        print(result.stdout.rstrip())
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    else:
        print("C++ SOAP-BPNN spike not built; torch vs numpy only\n")

    for name, system in SYSTEMS.items():
        e_np = evaluate_numpy(system)
        e_t, f_t = evaluate_torch(system, wrapper)
        e_p, _ = evaluate_pt(system, pt)
        f_fd = finite_difference_forces(
            lambda pos, system=system: evaluate_numpy({**system, "positions": pos}),
            np.asarray(system["positions"], dtype=np.float64),
            h=1e-6,
        )
        delta_e = abs(e_t - e_np)
        delta_f = float(np.max(np.abs(f_t - f_fd)))
        assert delta_e < 1e-10
        assert abs(e_p - e_np) < 1e-10
        assert delta_f < 5e-5
        names.append(name)
        dE.append(max(delta_e, 1e-18))
        dF.append(max(delta_f, 1e-18))
        print(
            f"{name:<12} {len(system['types']):>3} {e_np:12.6e} {e_t:12.6e} "
            f"{delta_e:10.3e} {delta_f:10.3e}  {system['periodic']}"
        )

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
x = np.arange(len(names))
axes[0].bar(x, dE, color="C0")
axes[0].set_xticks(x, names)
axes[0].set_ylabel("|ΔE| / kJ mol$^{-1}$")
axes[0].set_title("Torch vs numpy SOAP-BPNN")
axes[0].set_yscale("log")
axes[1].bar(x, dF, color="C2")
axes[1].set_xticks(x, names)
axes[1].set_ylabel("max |ΔF| / kJ mol$^{-1}$ nm$^{-1}$")
axes[1].set_title("Autograd vs finite differences")
axes[1].set_yscale("log")
fig.tight_layout()
