"""
OpenMM-ML mixed systems with PET-MAD-XS
=======================================

``createMixedSystem`` keeps a conventional force field on the whole
Topology and replaces the *internal* energy of an ML subset with PET-MAD-XS.
Custom metatomic models are not classified as long-range, so a periodic
box needs ``mlLongRange=False`` (the 7.5 Å cutoff does not see ligand
images across a multi-nanometre solvent box).

Covered here:

* ``MLPotential.getSupportedEmbeddings()`` and mechanical embedding
* toluene in explicit water (Amber ``prm7`` / ``rst7``)
* ``lambda_interpolate`` between MM internals and ML internals
* short Langevin vs an MM-only control (PE/KE/total, snapshots, RDFs)
* link atoms: only the ALA residue of ACE-ALA-NME is ML, so ACE/NME bonds
  that cross the boundary are capped (``returnInfo=True``)
"""

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from _openmm import (
    energy_forces,
    make_potential,
    openmm_ml_data,
    preferred_platforms,
    save_figure,
)
from _petmad import ensure_model


def potential_energy(context) -> float:
    return (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )


def wrap_delta(delta, box):
    L = np.diag(box)
    return delta - L * np.round(delta / L)


def rdf_pairs(pos, idx_i, idx_j, box, rmax=1.0, bins=40, skip_self=True):
    hist = np.zeros(bins, dtype=np.float64)
    edges = np.linspace(0.0, rmax, bins + 1)
    n_i, n_j = len(idx_i), len(idx_j)
    for i in idx_i:
        d = wrap_delta(pos[idx_j] - pos[i], box)
        r = np.linalg.norm(d, axis=1)
        if skip_self:
            r = r[r > 1e-8]
        hist += np.histogram(r, bins=edges)[0]
    vol = float(np.linalg.det(box))
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    norm = n_i * n_j / vol
    g = hist / np.maximum(norm * shell, 1e-30)
    r_mid = 0.5 * (edges[1:] + edges[:-1])
    return r_mid, g


def langevin_trace(system, positions, box_vectors, platform, n_steps, dt_ps, seed):
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picoseconds
    )
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
    n = system.getNumParticles()
    n_cons = system.getNumConstraints()
    dof = max(1, 3 * n - n_cons)
    r = 8.314462618e-3
    rows, xyz = [], []
    box = None
    for i in range(n_steps + 1):
        state = context.getState(getEnergy=True, getPositions=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        pos = np.array(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometers),
            dtype=np.float64,
            copy=True,
        )
        rows.append((pe, ke, pe + ke, 2.0 * ke / (dof * r)))
        xyz.append(pos)
        if box is None:
            box = np.array(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometer
                ),
                dtype=np.float64,
            )
        if i < n_steps:
            integrator.step(1)
    del context
    return np.asarray(rows, dtype=np.float64), xyz, box


model_path = ensure_model()
platform = preferred_platforms()[0]
potential = make_potential(str(model_path), device="cpu")
print(f"model {model_path.name}  platform {platform.getName()}")
print(f"embeddings {potential.getSupportedEmbeddings()}")

data = openmm_ml_data()
labels, energies = [], []
lambdas, e_lam = [], []

if data is None or not (data / "toluene" / "toluene-explicit.prm7").is_file():
    print("openmm-ml test data not found; skipping mixed PET-MAD examples")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.text(0.5, 0.5, "openmm-ml test data missing", ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, __file__)
else:
    prmtop = app.AmberPrmtopFile(str(data / "toluene" / "toluene-explicit.prm7"))
    inpcrd = app.AmberInpcrdFile(str(data / "toluene" / "toluene-explicit.rst7"))
    ml_atoms = list(range(15))
    mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
    mixed = potential.createMixedSystem(
        prmtop.topology,
        mm_system,
        ml_atoms,
        interpolate=False,
        embedding="mechanical",
        mlLongRange=False,
    )
    python_forces = [f for f in mixed.getForces() if isinstance(f, mm.PythonForce)]
    print(
        f"toluene mixed  N={prmtop.topology.getNumAtoms()}  ML={len(ml_atoms)}  "
        f"PythonForce particles={list(python_forces[0].getParticles()) if python_forces else None}"
    )

    mm_ctx = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform)
    mix_ctx = mm.Context(mixed, mm.VerletIntegrator(0.001), platform)
    mm_ctx.setPositions(inpcrd.positions)
    mix_ctx.setPositions(inpcrd.positions)
    e_mm = potential_energy(mm_ctx)
    e_mix = potential_energy(mix_ctx)
    labels.extend(["toluene MM", "toluene mixed"])
    energies.extend([e_mm, e_mix])
    print(f"  E_MM={e_mm:.4f}  E_mixed={e_mix:.4f} kJ/mol")
    del mm_ctx, mix_ctx

    try:
        interp = potential.createMixedSystem(
            prmtop.topology,
            mm_system,
            ml_atoms,
            interpolate=True,
            embedding="mechanical",
            mlLongRange=False,
        )
        interp_ctx = mm.Context(interp, mm.VerletIntegrator(0.001), platform)
        interp_ctx.setPositions(inpcrd.positions)
        e_l1 = potential_energy(interp_ctx)
        interp_ctx.setParameter("lambda_interpolate", 0)
        e_l0 = potential_energy(interp_ctx)
        print(f"  λ=1 {e_l1:.4f}  λ=0 {e_l0:.4f}")
        labels.extend(["λ=1 (ML)", "λ=0 (MM)"])
        energies.extend([e_l1, e_l0])
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            interp_ctx.setParameter("lambda_interpolate", lam)
            lambdas.append(lam)
            e_lam.append(potential_energy(interp_ctx))
        del interp_ctx
    except Exception as exc:
        lambdas, e_lam = [], []
        print(f"  λ-interpolate skipped ({type(exc).__name__}: {exc})")

    potential = make_potential(
        str(model_path), device="cpu", uncertainty_threshold=None
    )
    mm_md = prmtop.createSystem(nonbondedMethod=app.PME, rigidWater=True)
    mixed_md = potential.createMixedSystem(
        prmtop.topology,
        mm_md,
        ml_atoms,
        interpolate=False,
        embedding="mechanical",
        mlLongRange=False,
    )
    n_steps = 16
    dt_ps = 0.0005
    mm_only = prmtop.createSystem(nonbondedMethod=app.PME, rigidWater=True)
    mm_trace, mm_xyz, box = langevin_trace(
        mm_only, inpcrd.positions, inpcrd.boxVectors, platform, n_steps, dt_ps, 3
    )
    mix_trace, mix_xyz, _ = langevin_trace(
        mixed_md, inpcrd.positions, inpcrd.boxVectors, platform, n_steps, dt_ps, 3
    )
    t_fs = np.arange(len(mix_trace)) * dt_ps * 1000.0
    p0, p1 = mix_xyz[0], mix_xyz[-1]
    dr = wrap_delta(p1 - p0, box)
    print(
        f"Langevin {n_steps} x {dt_ps*1000:.1f} fs  "
        f"mixed dPE={mix_trace[-1,0]-mix_trace[0,0]:.1f}  "
        f"dKE={mix_trace[-1,1]-mix_trace[0,1]:.1f}  "
        f"dE={mix_trace[-1,2]-mix_trace[0,2]:.1f}  "
        f"T0={mix_trace[0,3]:.0f} T1={mix_trace[-1,3]:.0f} K"
    )
    print(
        f"  MM-only control  dPE={mm_trace[-1,0]-mm_trace[0,0]:.1f}  "
        f"dKE={mm_trace[-1,1]-mm_trace[0,1]:.1f}  "
        f"dE={mm_trace[-1,2]-mm_trace[0,2]:.1f}"
    )
    print(
        f"  max|dr| toluene={np.linalg.norm(dr[ml_atoms], axis=1).max():.4f} nm  "
        f"all={np.linalg.norm(dr, axis=1).max():.4f} nm"
    )

    atoms = list(prmtop.topology.atoms())
    c_idx = [a.index for a in atoms[:15] if a.element.symbol == "C"]
    h_idx = [a.index for a in atoms[:15] if a.element.symbol == "H"]
    o_idx = [a.index for a in atoms if a.element.symbol == "O" and a.index >= 15]
    ch_pairs = [
        (i, j)
        for i in c_idx
        for j in h_idx
        if np.linalg.norm(p0[i] - p0[j]) < 0.15
    ]
    cc_pairs = [
        (i, j)
        for a, i in enumerate(c_idx)
        for j in c_idx[a + 1 :]
        if np.linalg.norm(p0[i] - p0[j]) < 0.16
    ]

    fig2, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    for trace, name, color in (
        (mm_trace, "MM-only", "C0"),
        (mix_trace, "PET-MAD mixed", "C1"),
    ):
        axes[0, 0].plot(t_fs, trace[:, 0] - trace[0, 0], color=color, label=f"{name} ΔPE")
        axes[0, 0].plot(
            t_fs, trace[:, 1] - trace[0, 1], color=color, ls="--", label=f"{name} ΔKE"
        )
        axes[0, 0].plot(
            t_fs, trace[:, 2] - trace[0, 2], color=color, ls=":", label=f"{name} ΔE"
        )
    axes[0, 0].set_xlabel("t / fs")
    axes[0, 0].set_ylabel("kJ mol$^{-1}$")
    axes[0, 0].set_title("Minimized rst7: PE rises as KE partitions")
    axes[0, 0].legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    com = p0[ml_atoms].mean(axis=0)

    def xy(p):
        w = wrap_delta(p - com, box)
        return w[:, 0], w[:, 1]

    near_o = [
        i
        for i in o_idx
        if np.linalg.norm(wrap_delta(p0[i] - com, box)) < 0.55
    ]
    ax.scatter(*xy(p0[near_o]), s=18, c="0.75", label="O$_w$ t=0")
    ax.scatter(*xy(p1[near_o]), s=18, c="C0", alpha=0.45, label="O$_w$ t=end")
    ax.plot(*xy(p0[c_idx]), "o", color="C1", ms=7, label="C t=0")
    ax.plot(*xy(p1[c_idx]), "s", color="C3", ms=6, label="C t=end")
    ax.plot(*xy(p0[h_idx]), ".", color="C1", ms=5)
    ax.plot(*xy(p1[h_idx]), ".", color="C3", ms=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x / nm")
    ax.set_ylabel("y / nm")
    ax.set_title("Toluene + water O < 0.55 nm")
    ax.legend(fontsize=7)

    r, g_mix0 = rdf_pairs(p0, c_idx, o_idx, box, rmax=1.0)
    _, g_mix = rdf_pairs(p1, c_idx, o_idx, box, rmax=1.0)
    _, g_mm = rdf_pairs(mm_xyz[-1], c_idx, o_idx, box, rmax=1.0)
    axes[1, 0].plot(r, g_mix0, color="0.5", label="mixed t=0")
    axes[1, 0].plot(r, g_mix, color="C1", label="mixed t=end")
    axes[1, 0].plot(r, g_mm, color="C0", ls="--", label="MM t=end")
    axes[1, 0].set_xlabel("r(C–O$_w$) / nm")
    axes[1, 0].set_ylabel("g(r)")
    axes[1, 0].set_title("Toluene carbon – water oxygen")
    axes[1, 0].legend(fontsize=7)

    def bond_lengths(pos, pairs):
        return np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs])

    ch0, ch1 = bond_lengths(p0, ch_pairs), bond_lengths(p1, ch_pairs)
    cc0, cc1 = bond_lengths(p0, cc_pairs), bond_lengths(p1, cc_pairs)
    print(
        f"  C–H mean {ch0.mean():.4f} → {ch1.mean():.4f} nm  "
        f"C–C mean {cc0.mean():.4f} → {cc1.mean():.4f} nm"
    )
    xs_ch = np.arange(len(ch0))
    xs_cc = np.arange(len(cc0)) + len(ch0) + 1
    axes[1, 1].plot(xs_ch, ch0 * 10, "o", color="0.5", label="C–H t=0")
    axes[1, 1].plot(xs_ch, ch1 * 10, "s", color="C1", label="C–H t=end")
    axes[1, 1].plot(xs_cc, cc0 * 10, "o", color="0.5", fillstyle="none", label="C–C t=0")
    axes[1, 1].plot(xs_cc, cc1 * 10, "s", color="C3", fillstyle="none", label="C–C t=end")
    axes[1, 1].axhline(1.09, color="0.7", ls=":", lw=0.8)
    axes[1, 1].axhline(1.40, color="0.7", ls="--", lw=0.8)
    axes[1, 1].set_xlabel("bond index")
    axes[1, 1].set_ylabel("r / Å")
    axes[1, 1].set_title("Toluene bonds (ML region)")
    axes[1, 1].legend(fontsize=7)
    fig2.suptitle("Toluene-in-water Langevin: geometry is stable")
    fig2.tight_layout()
    save_figure(fig2, __file__, suffix="_langevin")

    ala_path = data / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb"
    if ala_path.is_file():
        ala_pdb = app.PDBFile(str(ala_path))
        ala_atoms = [
            atom.index
            for atom in ala_pdb.topology.atoms()
            if atom.residue.name == "ALA"
        ]
        print(
            f"alanine-dipeptide  N={ala_pdb.topology.getNumAtoms()}  "
            f"ML ALA={ala_atoms}"
        )
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        ala_mm = forcefield.createSystem(
            ala_pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
        info = potential.createMixedSystem(
            ala_pdb.topology,
            ala_mm,
            ala_atoms,
            interpolate=False,
            embedding="mechanical",
            mlLongRange=False,
            returnInfo=True,
        )
        n_old = ala_pdb.topology.getNumAtoms()
        n_new = info["topology"].getNumAtoms()
        link = n_new - n_old
        print(
            f"  returnInfo  atoms {n_old} → {n_new}  link hydrogens={link}  "
            f"oldToNew[:5]={info['oldToNew'][:5]}"
        )
        vsites = sum(
            1
            for i in range(info["system"].getNumParticles())
            if info["system"].isVirtualSite(i)
        )
        print(f"  virtual sites={vsites}")
        padded = list(ala_pdb.getPositions())
        padded.extend([mm.Vec3(0, 0, 0) * unit.nanometer] * link)
        ala_ctx = mm.Context(
            info["system"], mm.VerletIntegrator(0.001), platform
        )
        box = ala_pdb.topology.getPeriodicBoxVectors()
        if box is not None:
            ala_ctx.setPeriodicBoxVectors(*box)
        ala_ctx.setPositions(padded)
        ala_ctx.computeVirtualSites()
        e_link, _ = energy_forces(ala_ctx)
        print(f"  mixed energy with link atoms  E={e_link:.4f} kJ/mol")

    fig, axes = plt.subplots(
        1, 2 if e_lam else 1, figsize=(9.2 if e_lam else 6.4, 3.6)
    )
    if not e_lam:
        axes = [axes]
    colors = [
        "C0"
        if "MM" in name and "mixed" not in name and "λ" not in name
        else "C1"
        for name in labels
    ]
    axes[0].bar(labels, energies, color=colors)
    axes[0].set_ylabel("E / kJ mol$^{-1}$")
    axes[0].set_title("PET-MAD-XS mixed internals")
    axes[0].tick_params(axis="x", rotation=25)
    if e_lam:
        axes[1].plot(lambdas, e_lam, "o-", color="C2")
        axes[1].set_xlabel(r"$\lambda_\mathrm{interpolate}$")
        axes[1].set_ylabel("E / kJ mol$^{-1}$")
        axes[1].set_title("MM (0) → ML internals (1)")
    fig.tight_layout()
    save_figure(fig, __file__)
