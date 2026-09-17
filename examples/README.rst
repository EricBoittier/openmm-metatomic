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
   around the C++ plugin. Vacuum NVE/NVT (harmonic, PET-MAD-XS, SOAP-BPNN,
   toluene), a periodic water box, and a settings matrix (backend, device,
   consistency, neighbor list, platform). Matching C++ drivers:
   ``openmm-metatomic-run-md`` and ``openmm-metatomic-bench-settings``.
