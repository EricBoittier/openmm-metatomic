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
| M2 | Periodic systems and validated pair lists (done: both backends build pair lists via `vesin` — the core backend's `System::add_pairs`/`System::pairs` round-trip and the torch backend's `add_neighbor_list` both use the same shared, engine-side neighbor search, replacing the old O(N²) fallback and the previous "core backend does not implement pair lists" hard error; validated on periodic and non-periodic systems for both backends, `test-pairlist` in `ctest`. Not done: Verlet-list skin/caching across steps — `vesin`'s pairs are rebuilt from scratch on every `computeForce` call) |
| M3 | CUDA execution with measured transfer overhead (`CustomCPPForceImpl` host path) |
| M4 | Zero-copy or low-copy CUDA via DLPack wrapping of OpenMM buffers |
| M5 | OpenMM-ML entry-point adapter |
| M6 | ML/MM subsets |
| M7 | Stress, non-conservative forces, requested inputs (measured ~1.8x speedup on PET-MAD-XS via the upstream openmm-ml PR's direct force head, see CHANGELOG.md) |
| M8 | Packaging, documentation, upstream proposals |

Go/no-go at M3: if OpenMM CUDA interfaces cannot support safe low-copy
interoperation, compare a TorchForce-compatible wrapper against maintaining a
CUDA-specific kernel. The CPU plugin, TorchScript loader, and OpenMM-ML adapter
remain useful either way.

## Phases

0. Technical spike: load an in-process `BaseModel` and a TorchScript `.pt`,
   query capabilities, build a `metatomic::System` from fixed arrays, evaluate
   energy, read position gradients, compare to analytic / finite differences.
1. Minimal OpenMM force plugin via `CustomCPPForceImpl`.
2. Eliminate avoidable transfers (DLPack, stream sharing).
3. Pair-list strategy (vesin, skin, caching). Vesin is wired and pair lists
   are validated for both backends; skin/caching across steps is not done.
4. OpenMM-ML adapter as an external entry point.
5. ML/MM subsets (scatter/gather; isolated internal energy unless the model
   supports environment information).
6. Variants, non-conservative forces, virial, charge/spin, uncertainty reporter.
7. XML serialization (model path, not bytes) and conda-forge packaging.
