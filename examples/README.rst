Examples
========

The gallery covers three layers:

1. **M0 spike** — conservative harmonic well
   :math:`E = \tfrac{1}{2} k \sum_i \|r_i - r_{0,i}\|^2` on water, methane, CO2,
   an eight-atom carbon cube, and periodic water. **core**
   (``metatomic::execute_model``) and **torch** (``.pt`` /
   ``load_atomistic_model``).
2. **SOAP-BPNN** — metatrain ``soap_bpnn`` SOAP (torch-spex Laplacian
   eigenstates, sphericart spherical harmonics, power spectrum) plus a
   per-species SiLU MLP, implemented in metatomic-core C++ and as a matching
   TorchScript module.
3. **PET-MAD** — the vendored extra-small v1.5.0 TorchScript file from
   ``lab-cosmo/upet``, run through OpenMM-ML (pure ML water, toluene, and
   toluene-in-water ML/MM).
4. **OpenMM-ML setups and timings** — ``createSystem``, ``createMixedSystem``
   (including λ interpolation), and wall-clock cost of setup vs evaluation vs
   short Langevin MD on CPU, Reference, Torch CUDA, and the C++ spike.
