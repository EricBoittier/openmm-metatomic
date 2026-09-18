Examples
========

The gallery covers these layers, in ``plot_NN_*.py`` order:

1. **M0 spike** — conservative harmonic well
   :math:`E = \tfrac{1}{2} k \sum_i \|r_i - r_{0,i}\|^2` on water, methane, CO2,
   an eight-atom carbon cube, and periodic water. **core**
   (``metatomic::execute_model``) and **torch** (``.pt`` /
   ``load_atomistic_model``). ``plot_01`` through ``plot_04``.
2. **PET-MAD** (``plot_05``) — the vendored extra-small v1.5.0 TorchScript
   file from ``lab-cosmo/upet``, run through OpenMM-ML (pure ML water,
   toluene, and toluene-in-water ML/MM in ``plot_06``/``plot_07``).
3. **OpenMM-ML setups and timings** (``plot_08``) — ``createSystem``,
   ``createMixedSystem`` (including λ interpolation), and wall-clock cost of
   setup vs evaluation vs short Langevin MD on CPU, Reference, Torch CUDA,
   and the C++ spike.
4. **SOAP-BPNN** (``plot_09``) — metatrain ``soap_bpnn`` SOAP (torch-spex
   Laplacian eigenstates, sphericart spherical harmonics, power spectrum)
   plus a per-species SiLU MLP, implemented in metatomic-core C++ and as a
   matching TorchScript module.
5. **Scaling with atom count** (``plot_10``) — the same ``PythonForce`` path
   as ``plot_08``, swept from 3 to 60,000 atoms, with and without a
   ``vesin`` neighbor list, to see where ``createSystem`` vs. per-step
   evaluation cost actually dominates.
6. **Native ``MetatomicForce``** (``plot_11``–``plot_13``) — the SWIG wrapper
   around the C++ plugin. Vacuum NVE/NVT on a 3,000-atom harmonic cloud plus
   PET-MAD-XS / SOAP-BPNN / toluene, a 1,000-water harmonic-nl box and a
   32-water PET-MAD box, and a settings matrix at 3k–60k (harmonic) /
   256–5,000 waters (harmonic-nl) / 32 and 96 waters (PET-MAD). Matching
   C++ drivers: ``openmm-metatomic-run-md`` and
   ``openmm-metatomic-bench-settings``. Timing is a hot Context (warmup,
   median ``getState``, then ``Integrator.step(N)``).
7. **OpenMM-ML features on PET-MAD-XS** (``plot_14``–``plot_15``) —
   ``MLPotential("metatomic")`` constructor knobs (device, non-conservative
   forces, uncertainty), Langevin NVT and MonteCarlo NPT, then mixed
   toluene-in-water with mechanical embedding, ``lambda_interpolate``, and
   link atoms on ACE-ALA-NME. See also ``docs/src/openmm_ml.rst``.
8. **Native vs PythonForce** (``plot_16``) — the same PET-MAD-XS Systems built
   through ``MLPotential("metatomic")`` and ``MLPotential("metatomic-native")``,
   compared on energy, forces, ``getState`` cost and ms/step: toluene in
   vacuum, toluene in explicit water through ``createMixedSystem`` (so the
   native force uses its particle subset), and the direct force head.
