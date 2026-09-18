OpenMM-ML and PET-MAD-XS
========================

The **Python** entry point for an exported Metatomic model is OpenMM-ML,
not the native ``MetatomicForce`` plugin::

    from openmmml import MLPotential
    potential = MLPotential(
        "metatomic",
        modelPath="models/pet-mad-xs-v1.5.0.pt",
        device="cpu",
    )
    system = potential.createSystem(topology)

PET-MAD-XS (``lab-cosmo/upet``, extra-small v1.5.0) is the gallery's real
MLIP: cutoff 7.5 Å, CPU and CUDA, conservative energy plus
``non_conservative_force``, ``non_conservative_stress``, and
``energy_uncertainty``. The gallery walks the OpenMM-ML surface with that
checkpoint.

Constructor
-----------

+-------------------------+--------------------------------------------------+
| Argument                | PET-MAD-XS                                       |
+=========================+==================================================+
| ``modelPath``           | vendored ``.pt``                                 |
+-------------------------+--------------------------------------------------+
| ``device``              | ``"cpu"`` or ``"cuda"``                          |
+-------------------------+--------------------------------------------------+
| ``checkConsistency``    | metatomic runtime checks (slow)                  |
+-------------------------+--------------------------------------------------+
| ``non_conservative``    | ``False`` (autograd), ``"forces"``, ``"stress"``,|
|                         | or ``True`` (both). OpenMM ``PythonForce`` still |
|                         | returns only energy and forces.                  |
+-------------------------+--------------------------------------------------+
| ``uncertainty_threshold`` | per-atom energy uncertainty in eV (default     |
|                         | 0.1). ``None`` disables the warning.             |
+-------------------------+--------------------------------------------------+
| ``variants``            | output-name → variant string; unused on this     |
|                         | checkpoint (no named energy variants)            |
+-------------------------+--------------------------------------------------+

``createSystem`` / ``createMixedSystem`` also accept ``pbc`` (length-3
booleans), and ``charge`` / ``spinMultiplicity`` / ``info`` if a model
requests those extra inputs. PET-MAD-XS does not.

Pure ML
-------

* ``plot_05_petmad.py`` — energy match vs ``load_atomistic_model``,
  short NVE, toluene vacuum.
* ``plot_14_petmad_openmm_ml.py`` — conservative vs non-conservative NVE,
  force scatter, ``energy_uncertainty``, Langevin, MonteCarlo NPT on 32
  waters. The barostat finite-differences the **energy** (a strain tensor
  is applied when all three directions are periodic). Non-conservative
  stress cannot drive it.

Mixed ML/MM
-----------

Mechanical embedding is the default. Custom metatomic models must set
``mlLongRange`` on a periodic box: PET-MAD's cutoff is local, so
``mlLongRange=False`` — MM still computes ligand–image nonbonded terms.

* ``plot_07_mlmm.py`` — the same API on a harmonic well (controlled ΔE).
* ``plot_15_petmad_mixed.py`` — toluene in explicit water with PET-MAD-XS
  internals, ``lambda_interpolate``, short Langevin vs an MM-only control
  (PE/KE/total, snapshots, C–O$_w$ RDF), and link atoms on ACE-ALA-NME
  (only ALA is ML; ``returnInfo=True``). A minimized ``rst7`` plus fresh
  300 K velocities moves energy from KE into PE; that is not a geometry
  blow-up (the MM-only box does the same).

::

    mixed = potential.createMixedSystem(
        topology, mm_system, ml_atoms,
        embedding="mechanical",
        mlLongRange=False,
        interpolate=False,
    )
    print(potential.getSupportedEmbeddings())  # includes "mechanical"

Native plugin
-------------

Once the SWIG wrappers are built, the same ``.pt`` can go through
``MetatomicForce`` instead of ``PythonForce``, either directly (gallery
``plot_11``) or as an OpenMM-ML backend::

    from openmmml import MLPotential
    import openmmmetatomic

    openmmmetatomic.register()          # entry point does this when installed
    potential = MLPotential(
        "metatomic-native",
        modelPath="model.pt",
        device="cpu",
        non_conservative="forces",      # same keywords as "metatomic"
        variants={"energy": "pbe"},
        uncertainty_threshold=0.1,
        backend="auto",                 # extra: "torch" or "core"
    )
    mixed = potential.createMixedSystem(topology, mm_system, ml_atoms)

``MechanicalEmbedding`` only calls ``potential.addForces(...)``, so mixed
systems, ``lambda_interpolate`` and link atoms behave exactly as they do with
``MLPotential("metatomic")``. Two differences worth knowing:

* the native backend answers ``getMLLongRange()`` with ``False`` instead of
  ``None``, so mechanical embedding already knows the ML part is local and
  rejects an explicit ``mlLongRange`` argument;
* a non-conservative *stress* is requested and validated, but OpenMM has no
  virial path, so it cannot drive an integrator. NPT works through a
  ``MonteCarloBarostat``, which finite-differences the energy.

``plot_16_native_vs_pythonforce.py`` runs both backends on identical Systems.
On PET-MAD-XS the mixed toluene-in-water box (15 ML atoms of 6,495) comes out
the same (415 ms either way) — the model forward dominates — while toluene in
vacuum favours the native path a little: 12.6 ms vs 11.1 ms (1.13x), or
6.6 ms vs 5.4 ms (1.23x) with the direct force head. Energies agree to
~1e-3 kJ/mol and forces to ~1e-2 kJ/mol/nm, the level expected from a float32
model evaluated with two different neighbor-list builds.

``plot_17_complex_mixed.py`` (and ``tests/python/test_complex_mixed.py``) is the
attempt to break that: Amber19 ACE-ALA-NME in explicit water with NaCl (bonds, angles,
proper and improper torsions, CMAP, PME, rigid water, H-mass repartitioning,
plus an injected Ryckaert–Bellemans torsion), CHARMM36 on the same peptide
(NBFix ``CustomNonbondedForce`` and 1-4 ``CustomBondForce``), ALA-only link
atoms, ``lambda_interpolate`` at 0/½/1, a reversed ML subset, every other
water molecule as ML (~1,000 discontiguous atoms), two native forces in one
System, and a short Langevin run that asserts the ML force group is silent
on MM atoms.
