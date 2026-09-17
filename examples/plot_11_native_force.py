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

This example runs NVE (energy drift) and NVT Langevin on a 3,000-atom
harmonic cloud (core, and TorchScript when ``harmonic-3000.pt`` is
exported), PET-MAD-XS and SOAP-BPNN on vacuum water, and PET-MAD-XS on
toluene.
"""

import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit

from _native import (
    MetatomicForce,
    ensure_harmonic_pt,
    harmonic_cloud,
    harmonic_pt,
    make_system,
    petmad_path,
    preferred_platform,
    run_nve,
    run_nvt,
    save_figure,
    soap_pt,
)
from _openmm import openmm_ml_data
from _petmad import WATER_NM, WATER_NUMBERS

force_cls = MetatomicForce()
print(f"MetatomicForce {force_cls}")

platform = preferred_platform()
print(f"platform {platform.getName()}")

N_HARMONIC = 3000
nve_steps = 80
nvt_steps = 40
ml_nve = 40
ml_nvt = 25

traces = []
print(f"{'model':<22} {'N':>8} {'NVE drift':>12} {'NVE T':>8} {'NVT T':>8}  E0")

cloud_types, cloud_pos = harmonic_cloud(N_HARMONIC)
harmonic_jobs = [("harmonic/core", "harmonic", "core")]
scaled = ensure_harmonic_pt(N_HARMONIC)
if scaled is not None:
    harmonic_jobs.append(("harmonic/torch", str(scaled), "torch"))
elif harmonic_pt() is not None and N_HARMONIC == 3:
    harmonic_jobs.append(("harmonic/torch", str(harmonic_pt()), "torch"))

for title, path, backend in harmonic_jobs:
    system = make_system(cloud_types, path, backend=backend)
    nve = run_nve(system, cloud_pos, platform, nve_steps)
    system_nvt = make_system(cloud_types, path, backend=backend)
    nvt = run_nvt(system_nvt, cloud_pos, platform, nvt_steps)
    drift = nve[-1, 2] - nve[0, 2]
    print(
        f"{title:<22} {len(cloud_types):8d} {drift:12.4e} "
        f"{nve[:, 3].mean():8.1f} {nvt[:, 3].mean():8.1f}  {nve[0, 0]:.4f}"
    )
    traces.append((title, nve, nvt))

ml_jobs = [("PET-MAD-XS", str(petmad_path()), "torch")]
soap = soap_pt()
if soap is not None:
    ml_jobs.append(("SOAP-BPNN", str(soap), "torch"))

for title, path, backend in ml_jobs:
    system = make_system(WATER_NUMBERS, path, backend=backend)
    nve = run_nve(system, WATER_NM, platform, ml_nve)
    system_nvt = make_system(WATER_NUMBERS, path, backend=backend)
    nvt = run_nvt(system_nvt, WATER_NM, platform, ml_nvt)
    drift = nve[-1, 2] - nve[0, 2]
    print(
        f"{title:<22} {len(WATER_NUMBERS):8d} {drift:12.4f} "
        f"{nve[:, 3].mean():8.1f} {nvt[:, 3].mean():8.1f}  {nve[0, 0]:.4f}"
    )
    traces.append((title, nve, nvt))

data = openmm_ml_data()
if data is not None and (data / "toluene" / "toluene.pdb").is_file():
    import openmm.app as app

    pdb = app.PDBFile(str(data / "toluene" / "toluene.pdb"))
    toluene_types = [atom.element.atomic_number for atom in pdb.topology.atoms()]
    toluene_pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    system = make_system(toluene_types, str(petmad_path()), backend="torch")
    nve = run_nve(system, toluene_pos, platform, ml_nve)
    system_nvt = make_system(toluene_types, str(petmad_path()), backend="torch")
    nvt = run_nvt(system_nvt, toluene_pos, platform, ml_nvt)
    drift = nve[-1, 2] - nve[0, 2]
    print(
        f"{'toluene PET-MAD':<22} {len(toluene_types):8d} {drift:12.4f} "
        f"{nve[:, 3].mean():8.1f} {nvt[:, 3].mean():8.1f}  {nve[0, 0]:.4f}"
    )
    traces.append(("toluene PET-MAD", nve, nvt))

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
dt = 0.5
for title, nve, nvt in traces:
    if nve is not None:
        axes[0].plot(np.arange(len(nve)) * dt, nve[:, 2] - nve[0, 2], label=title)
    if nvt is not None:
        axes[1].plot(np.arange(len(nvt)) * dt, nvt[:, 3], label=title)
axes[0].set_xlabel("t / fs")
axes[0].set_ylabel(r"$\Delta E_\mathrm{tot}$ / kJ mol$^{-1}$")
axes[0].set_title(f"NVE, harmonic {N_HARMONIC} atoms / ML molecules")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("t / fs")
axes[1].set_ylabel("T / K")
axes[1].set_title("Langevin NVT")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_figure(fig, __file__)
