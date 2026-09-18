"""
Complex mixed ML/MM: every MM term, a large ML region, and the sharp edges
==========================================================================

ACE-ALA-NME in explicit water is small as a solute and huge as a System.
Amber19 already puts bonds, angles, proper and improper periodic torsions, a
CMAP and PME on it; CHARMM36 adds NBFix and 1-4 custom bonds. This gallery
stacks NaCl, an injected Ryckaert–Bellemans torsion, link atoms on ALA,
``lambda_interpolate``, a reversed ML subset, every other water molecule as
ML (~1,000 atoms, discontiguous), and the native plugin against
``PythonForce`` — the attitude is to try to break the gather/scatter path.
"""

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from _openmm import (
    make_native_potential,
    make_potential,
    preferred_platforms,
    save_figure,
)
from _complex_mixed import (
    ethanol_paths,
    export_probe_model,
    format_inventory,
    inventory,
    ion_atoms,
    load_alanine_box,
    mixed,
    native_forces,
    peptide_atoms,
    residue_atoms,
    water_atoms,
    energy_forces,
)

platform = preferred_platforms()[0]
print(f"platform {platform.getName()}")

topology, positions, mm_system, _ = load_alanine_box("amber19", add_salt=True)
box = topology.getPeriodicBoxVectors()
info = inventory(mm_system, topology)
peptide = peptide_atoms(topology)
waters = water_atoms(topology)
ions = ion_atoms(topology)
print("amber19", format_inventory(info))
print(
    f"  peptide={len(peptide)}  waters={len(waters)}  ions={len(ions)} "
    f"{ions}  N={topology.getNumAtoms()}"
)

numbers = [atom.element.atomic_number for atom in topology.atoms()]
model_path = "/tmp/openmm-metatomic-complex-mixed.pt"
export_probe_model(model_path, sorted(set(numbers) | {11, 17}))
python = make_potential(model_path, device="cpu", uncertainty_threshold=None)
native = make_native_potential(model_path, device="cpu", uncertainty_threshold=None)

cases = {}


def compare(label, atoms, **kwargs):
    ref = mixed(python, topology, mm_system, atoms, **kwargs)
    nat = mixed(native, topology, mm_system, atoms, **kwargs)
    if isinstance(ref, dict):
        ref, nat = ref["system"], nat["system"]
    e_ref, f_ref = energy_forces(ref, positions, box)
    e_nat, f_nat = energy_forces(nat, positions, box)
    n = min(len(f_ref), len(f_nat))
    d = np.linalg.norm(f_nat[:n] - f_ref[:n], axis=1)
    print(
        f"{label:<28} ΔE {e_nat - e_ref: .3e} kJ/mol  "
        f"max|ΔF| {d.max():.3e}  RMS {np.sqrt((d ** 2).mean()):.3e}  "
        f"ML={len(atoms)}/{topology.getNumAtoms()}"
    )
    cases[label] = dict(dE=e_nat - e_ref, maxF=d.max(), n_ml=len(atoms), e=e_nat)
    return nat, e_nat, f_nat


compare("peptide ML, water+ions MM", peptide)
compare("ALA + link hydrogens", residue_atoms(topology, "ALA"), returnInfo=True)
compare("peptide, reversed order", list(reversed(peptide)))
compare("ions only as ML", ions)
mols = [waters[i : i + 3] for i in range(0, len(waters), 3)]
gapped = [atom for mol in mols[::2] for atom in mol]
compare("every other water as ML", gapped)

# Interpolation: λ=0 must be pure MM, λ=1 the mixed energy above.
interp = mixed(native, topology, mm_system, peptide, interpolate=True)
mm_e, _ = energy_forces(mm_system, positions, box)
lam, e_lam = [], []
for value in (0.0, 0.25, 0.5, 0.75, 1.0):
    energy, _ = energy_forces(
        interp, positions, box, parameters={"lambda_interpolate": value}
    )
    lam.append(value)
    e_lam.append(energy)
    print(f"lambda_interpolate={value:.2f}  E={energy:.4f} kJ/mol")
print(f"  MM energy {mm_e:.4f}  matches λ=0 {np.isclose(e_lam[0], mm_e)}")

try:
    ch_top, ch_pos, ch_sys, _ = load_alanine_box("charmm", add_salt=False)
    ch_info = inventory(ch_sys, ch_top)
    print("charmm36", format_inventory(ch_info))
    ch_peptide = peptide_atoms(ch_top)
    e_ref, f_ref = energy_forces(
        mixed(python, ch_top, ch_sys, ch_peptide), ch_pos, ch_top.getPeriodicBoxVectors()
    )
    e_nat, f_nat = energy_forces(
        mixed(native, ch_top, ch_sys, ch_peptide), ch_pos, ch_top.getPeriodicBoxVectors()
    )
    d = np.linalg.norm(f_nat - f_ref, axis=1)
    print(
        f"{'CHARMM peptide ML':<28} ΔE {e_nat - e_ref: .3e}  max|ΔF| {d.max():.3e}"
    )
    cases["CHARMM peptide ML"] = dict(
        dE=e_nat - e_ref, maxF=d.max(), n_ml=len(ch_peptide), e=e_nat
    )
except Exception as exc:
    print(f"charmm36 skipped: {exc}")

paths = ethanol_paths()
if paths is not None:
    pdb = app.PDBFile(str(paths[0]))
    ethanol_mm = app.ForceField(str(paths[1])).createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff
    )
    heavy = [atom.index for atom in pdb.topology.atoms() if atom.element.symbol != "H"]
    e_ref, f_ref = energy_forces(mixed(python, pdb.topology, ethanol_mm, heavy), pdb.positions)
    e_nat, f_nat = energy_forces(mixed(native, pdb.topology, ethanol_mm, heavy), pdb.positions)
    d = np.linalg.norm(f_nat - f_ref, axis=1)
    print(f"{'ethanol heavy as ML':<28} ΔE {e_nat - e_ref: .3e}  max|ΔF| {d.max():.3e}")
    cases["ethanol heavy as ML"] = dict(
        dE=e_nat - e_ref, maxF=d.max(), n_ml=len(heavy), e=e_nat
    )

# Short Langevin on the peptide-ML mixed system: ML forces must stay off MM.
# Isolate MetatomicForce on group 6 so getState(groups={6}) is not MM group 0.
mixed_nat = mixed(native, topology, mm_system, peptide, forceGroup=6)
ml = native_forces(mixed_nat)[0]
integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, 0.0005 * unit.picoseconds
)
context = mm.Context(mixed_nat, integrator, platform)
context.setPositions(positions)
context.setPeriodicBoxVectors(*box)
context.setVelocitiesToTemperature(300 * unit.kelvin, 11)
pe = []
for _ in range(12):
    pe.append(
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )
    integrator.step(1)
ml_forces = np.asarray(
    context.getState(getForces=True, groups={ml.getForceGroup()})
    .getForces(asNumpy=True)
    .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
)
xyz = np.asarray(
    context.getState(getPositions=True)
    .getPositions(asNumpy=True)
    .value_in_unit(unit.nanometer)
)
outside = [i for i in range(mixed_nat.getNumParticles()) if i not in set(ml.getParticles())]
leaked = float(np.linalg.norm(ml_forces[outside], axis=1).max())
print(f"Langevin 12 steps  leaked ML force on MM atoms {leaked:.3e} kJ/mol/nm")
del context

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
labels = list(cases)
axes[0].bar(range(len(labels)), [abs(cases[n]["dE"]) + 1e-16 for n in labels], color="C0")
axes[0].set_yscale("log")
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels([n.replace(" ", "\n") for n in labels], fontsize=7)
axes[0].set_ylabel("|ΔE| native vs PythonForce / kJ mol$^{-1}$")
axes[0].set_title("Energy agreement")

axes[1].bar(range(len(labels)), [cases[n]["n_ml"] for n in labels], color="C1")
axes[1].set_xticks(range(len(labels)))
axes[1].set_xticklabels([n.replace(" ", "\n") for n in labels], fontsize=7)
axes[1].set_ylabel("ML atoms")
axes[1].set_title("Subset size")

axes[2].plot(np.arange(len(pe)), pe, "o-", color="C2")
axes[2].set_xlabel("Langevin step")
axes[2].set_ylabel("E / kJ mol$^{-1}$")
axes[2].set_title(f"peptide-ML NVT, leak {leaked:.1e}")
fig.suptitle("Complex mixed: Amber19 ACE-ALA-NME + water + NaCl")
fig.tight_layout()
save_figure(fig, __file__)

fig2, ax = plt.subplots(figsize=(5.2, 5.2))
com = xyz[peptide].mean(axis=0)
w = xyz - com
ax.scatter(w[waters, 0], w[waters, 1], s=4, c="0.8", label="water")
if ions:
    ax.scatter(w[ions, 0], w[ions, 1], s=40, c="C0", marker="s", label="Na/Cl")
ax.scatter(w[peptide, 0], w[peptide, 1], s=28, c="C3", label="peptide (ML)")
ax.set_aspect("equal")
ax.set_xlabel("x / nm")
ax.set_ylabel("y / nm")
ax.set_title("After 12 Langevin steps")
ax.legend(fontsize=8)
fig2.tight_layout()
save_figure(fig2, __file__, suffix="_snapshot")
