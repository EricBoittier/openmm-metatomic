# Roadmap

Native OpenMM plugin for Metatomic. Public API stays small; `MetatomicForce`
holds serializable configuration, not live model objects.

Evaluation has two backends:

- **core** — `metatomic::load_model` / `execute_model`, `System` from DLPack,
  explicit position gradients. Torch is a runtime plugin, not a required
  link-time dependency.
- **torch** — `metatomic_torch::load_atomistic_model` for exported TorchScript
  `.pt` models (backwards compatible with today's Metatomic exports).

`setBackend("auto")` picks `torch` for `.pt`/`.pth` and `core` otherwise.

## Milestones

| Milestone | Deliverable |
| --- | --- |
| M0 | Standalone C++ spike: core harmonic + TorchScript `.pt`, energy and forces vs FD |
| M1 | CPU `MetatomicForce`, full systems, energy and conservative forces (builds and validated up to 500k atoms, both backends, through a real `Context`; periodic systems and pair lists now work too, see M2. Python entry point: SWIG `openmmmetatomic.MetatomicForce`) |
| M2 | Periodic systems and validated pair lists (done: both backends build pair lists via `vesin` — the core backend's `System::add_pairs`/`System::pairs` round-trip and the torch backend's `add_neighbor_list` both use the same shared, engine-side neighbor search, replacing the old O(N²) fallback and the previous "core backend does not implement pair lists" hard error; validated on periodic and non-periodic systems for both backends, `test-pairlist` in `ctest`. Verlet skin/caching across steps is done too, see Phase 3) |
| M3 | CUDA execution with measured transfer overhead (done: nothing in the plugin needed changing — `setDevice("cuda")` moves the model and `CustomCPPForceImpl` hands over host buffers on every platform. What was missing was platform plugins: `sim::loadPlatformPlugins()` now registers them, so the drivers see CPU and CUDA instead of Reference alone. `test-cuda` checks cpu/cuda and Reference/CPU/CUDA agreement including the subset path; `openmm-metatomic-bench-cuda` measures it. PET-MAD-XS on an RTX 4060 Ti: 2.8x at 96 atoms, 8.6x at 288. Our host↔device copies are 23–49 µs, ≤0.2% of an evaluation) |
| M4 | Zero-copy or low-copy CUDA via DLPack wrapping of OpenMM buffers (**no-go on the M3 evidence**: the host path costs ≤1% of a realistic evaluation, and the CUDA platform is no slower than Reference for the same CUDA-resident model. Revisit only for a model whose GPU forward is ~1 ms at ≥10k atoms; see CHANGELOG.md) |
| M5 | OpenMM-ML entry-point adapter (done: `MLPotential("metatomic-native")` from `openmmmetatomic.openmmml`, registered by an `openmmml.potentials` entry point or `openmmmetatomic.register()`; `getMLLongRange()` is `False`, so mechanical embedding rejects an explicit `mlLongRange`) |
| M6 | ML/MM subsets (done: `MetatomicForce::setParticles` plus gather/scatter in `computeForce`; mixed systems, `lambda_interpolate` and link atoms come from OpenMM-ML, which only calls `addForces`. Verified against the PythonForce backend on PET-MAD-XS toluene-in-water, `python-openmmml-native` and `plot_16`) |
| M7 | Stress, non-conservative forces, requested inputs (done except the virial: `setNonConservative`, `setVariant`, `setUncertaintyThreshold`, `setCharge`, `setSpinMultiplicity`, `test-features` in `ctest`. A non-conservative stress is requested and validated but cannot drive an integrator — OpenMM has no virial path — so NPT stays on a `MonteCarloBarostat`. Measured ~1.23x on PET-MAD-XS toluene from the direct force head, see CHANGELOG.md) |
| M8 | Packaging, documentation, upstream proposals |

Go/no-go at M3, answered: the host path is safe and cheap, so neither a
TorchForce-compatible wrapper nor a CUDA-specific kernel is worth maintaining.
`CustomCPPForceImpl` copies positions out and forces back for us, the copies
are tens of microseconds, and the model forward is three to four orders of
magnitude larger. Phase 2 closes with M4 declined rather than implemented.

## Phases

0. Technical spike: load an in-process `BaseModel` and a TorchScript `.pt`,
   query capabilities, build a `metatomic::System` from fixed arrays, evaluate
   energy, read position gradients, compare to analytic / finite differences.
1. Minimal OpenMM force plugin via `CustomCPPForceImpl`.
2. Eliminate avoidable transfers (DLPack, stream sharing). Measured and
   declined: the transfers are 23–49 µs against an 12–129 ms evaluation. See
   M3/M4.
3. Pair-list strategy (vesin, skin, caching). Done: each request keeps a
   `NeighborList` alive across `compute()` calls and vesin caches its topology
   behind a Verlet skin (`OPENMM_METATOMIC_NEIGHBOR_SKIN`, default 0.05 nm).
   Cached and fresh runs agree exactly (`checkSkinCache` in `test-pairlist`).
   Worth 24% per step on a cheap model, 1–2% on PET-MAD-XS, where the model
   forward dominates.
4. OpenMM-ML adapter as an external entry point. Done, see M5.
5. ML/MM subsets (scatter/gather; isolated internal energy unless the model
   supports environment information). Done, see M6.
6. Variants, non-conservative forces, virial, charge/spin, uncertainty reporter.
   Done except the virial, which OpenMM cannot consume; see M7.
7. XML serialization (model path, not bytes) and conda-forge packaging.
