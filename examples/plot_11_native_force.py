"""
Native MetatomicForce
=====================

The C++ plugin is a real OpenMM ``Force``. Import it, set atomic types
explicitly, and add it to a ``System``::

    from openmmmetatomic import MetatomicForce
    force = MetatomicForce("model.pt")
    force.setBackend("auto")
    force.setAtomicTypes([8, 1, 1])
    system.addForce(force)

This example runs NVE (energy drift) and NVT Langevin on vacuum water for
the built-in core harmonic well, an exported TorchScript copy, PET-MAD-XS,
and SOAP-BPNN when the ``.pt`` files are in ``build/``.
"""

import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit

from _native import (
    MetatomicForce,
    harmonic_pt,
    make_system,
    petmad_path,
    preferred_platform,
    run_nve,
    run_nvt,
    soap_pt,
)
from _openmm import openmm_ml_data
from _petmad import WATER_NM, WATER_NUMBERS

force_cls = MetatomicForce()
print(f"MetatomicForce {force_cls}")

platform = preferred_platform()
print(f"platform {platform.getName()}")
positions = WATER_NM
types = WATER_NUMBERS
nve_steps = 80
nvt_steps = 40

jobs = [("harmonic/core", "harmonic", "core")]
pt = harmonic_pt()
if pt is not None:
    jobs.append(("harmonic/torch", str(pt), "torch"))
jobs.append(("PET-MAD-XS", str(petmad_path()), "torch"))
soap = soap_pt()
if soap is not None:
    jobs.append(("SOAP-BPNN", str(soap), "torch"))

data = openmm_ml_data()
if data is not None and (data / "toluene" / "toluene.pdb").is_file():
    import openmm.app as app

    pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
    toluene_types = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    toluene_pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    system = make_system(toluene_types, str(petmad_path()), backend="torch")
    nvt = run_nvt(system, toluene_pos, platform, 20)
    print(
        f"{'toluene PET-MAD':<16} {'n/a':>12} {'n/a':>8} {nvt[:, 3].mean():8.1f}  "
        f"{nvt[0, 0]:.4f}"
    )
    traces.append(("toluene PET-MAD NVT", None, nvt))

traces = []
print(f"{'model':<16} {'NVE drift':>12} {'NVE T':>8} {'NVT T':>8}  E0")
for title, path, backend in jobs:
    system = make_system(types, path, backend=backend)
    nve = run_nve(system, positions, platform, nve_steps)
    system_nvt = make_system(types, path, backend=backend)
    nvt = run_nvt(system_nvt, positions, platform, nvt_steps)
    drift = nve[-1, 2] - nve[0, 2]
    print(
        f"{title:<16} {drift:12.4f} {nve[:, 3].mean():8.1f} {nvt[:, 3].mean():8.1f}  "
        f"{nve[0, 0]:.4f}"
    )
    traces.append((title, nve, nvt))

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
dt = 0.5
for title, nve, nvt in traces:
    if nve is not None:
        axes[0].plot(np.arange(len(nve)) * dt, nve[:, 2] - nve[0, 2], label=title)
    if nvt is not None:
        axes[1].plot(np.arange(len(nvt)) * dt, nvt[:, 3], label=title)
axes[0].set_xlabel("t / fs")
axes[0].set_ylabel(r"$\Delta E_\mathrm{tot}$ / kJ mol$^{-1}$")
axes[0].set_title(f"NVE, {nve_steps} x 0.5 fs")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("t / fs")
axes[1].set_ylabel("T / K")
axes[1].set_title(f"Langevin NVT, {nvt_steps} x 0.5 fs")
axes[1].legend(fontsize=8)
fig.tight_layout()
