# Changelog

All notable changes to openmm-metatomic are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- SWIG Python wrappers for `MetatomicForce` (`from openmmmetatomic import
  MetatomicForce`), a C++ NVE/NVT driver (`openmm-metatomic-run-md`), and
  a settings-matrix bench (`openmm-metatomic-bench-settings`). Gallery
  `plot_11`–`plot_13` run vacuum and periodic MD plus the settings sweep
  (figures under `examples/plots/`). Timing is a hot `Context`: warmup
  evals and steps, median `getState`, then `Integrator::step(N)` with no
  per-step energy query. Systems are 3k–60k atoms (harmonic), 256–5,000
  waters (harmonic-nl), and 32 / 96 waters (PET-MAD-XS). Set
  `OPENMM_METATOMIC_NEIGHBOR_LIST=naive` to force the O(N²) pair-list
  fallback when vesin is compiled in.
- Gallery `plot_14` / `plot_15` and `docs/src/openmm_ml.rst` walk OpenMM-ML
  features on PET-MAD-XS: `non_conservative` forces vs autograd NVE,
  `energy_uncertainty`, MonteCarlo NPT, mixed toluene-in-water
  (`mlLongRange=False`, mechanical embedding, `lambda_interpolate`), and
  link atoms on ACE-ALA-NME (`returnInfo=True`). The mixed Langevin figure
  shows PE/KE/total against MM-only plus toluene snapshots and C–O$_w$
  RDFs: the PE rise from a minimized `rst7` is equipartition, not a
  blow-up.
- **M2: pair-list support for both backends.** `CMakeLists.txt` now detects
  and links `vesin` (`pip install vesin`; falls back to the O(N^2) loop with
  a `-- vesin not found` status message if missing) via a new
  `OPENMM_METATOMIC_VESIN` option (on by default). This activates two
  things at once:
  - The torch backend's neighbor-list construction (`addNeighborList` in
    `MetatomicEvaluator.cpp`) had a real `vesin`-backed code path since M0,
    guarded by `OPENMM_METATOMIC_USE_VESIN` — but nothing ever defined that
    macro, so every torch model requesting a neighbor list (PET-MAD-XS
    included) has been running the O(N^2)/periodic-image fallback the whole
    time, untested against the real thing until now.
  - The core backend previously hard-error'd on any model that called
    `requested_pair_lists()` ("core backend does not yet implement pair
    lists"). It now builds and attaches pair lists the same way, via
    `metatomic::System::add_pairs`/`System::pairs` (the C++, non-torch
    equivalent of the torch backend's `add_neighbor_list`).
  - Both paths share one new backend-agnostic helper,
    `computeNeighborPairs()` (vesin or the naive fallback, returning flat
    `{first_atom, second_atom, cell_shift_a/b/c}` samples and pair vectors),
    that the torch and core evaluators each wrap in their own
    metatensor(_torch) `TensorBlock` construction. `HarmonicModel.h` gained
    a `NeighborHarmonicModel` (core, exposed as the built-in model
    `"harmonic-nl"`) and `spike/export_pairlist_torch.py` exports a
    matching TorchScript twin: both add a pair-list contribution multiplied
    by zero to the same independent-atom harmonic well, so they share its
    analytic solution — this isolates "did the pair list round-trip
    correctly" from "is the physics right". Validated periodic and
    non-periodic, both backends, in the new `spike/test_pairlist.cpp`
    (`test-pairlist` in `ctest`).
  - Not done: Verlet-list skin/caching across steps (`vesin`'s pairs are
    rebuilt from scratch on every `computeForce` call, same as before this
    change) — see ROADMAP.md Phase 3.
- `spike/bench_scaling.cpp` (`openmm-metatomic-bench-scaling`, built whenever
  `OPENMM_DIR` is set): the first timing that actually exercises
  `MetatomicForce` through a real `OpenMM::Context` (`CustomCPPForceImpl` ->
  `getState`), for both backends, against the same model called directly
  with no OpenMM in the loop. Every row is verified against the analytic
  harmonic-well energy before it is timed (`verifyCore`/`verifyTorch`), and
  each reported number is a median/mean/stddev over 50 timed evals (15
  warmup) — see the timing baseline below for repeated full-process runs on
  top of that. `spike/export_scaling_models.py` exports the matching
  per-atom-count TorchScript models. Registered as a small smoke test
  (`bench-scaling-smoke`, N=3,30) in `ctest`. `runFairly()` primes every
  case for a given N in round-robin before timing any of them, so no case
  gets an unearned memory-warmth advantage over another — see "Fixed" and
  the revised timing baseline below for why that matters.
- GitHub Pages and a Docs workflow that builds the Sphinx gallery
  (<https://ericboittier.github.io/openmm-metatomic/>).
- SOAP-BPNN twin of metatrain `soap_bpnn` (legacy path): torch-spex Laplacian
  eigenstates + sphericart spherical harmonics + `SoapPowerSpectrum` contraction,
  then a per-species SiLU MLP. Core C++ uses sphericart and the dumped spline.
  A tiny metatrain checkpoint (`models/soap-bpnn-tiny.ckpt`) is vendored so the
  official `SoapBpnn` can execute; `ensure_soap_checkpoint()` will fetch from
  Hugging Face when `OPENMM_METATOMIC_SOAP_HF` is set. Checked against torch-spex,
  loaded-checkpoint `SoapBpnn.forward`, and a matching TorchScript export.
  `./build/openmm-metatomic-bpnn build/soap-bpnn.pt`
- Local timing baseline for the current end-user path
  (`MLPotential("metatomic")` → `PythonForce`) and the C++ `execute_model`
  spike. Re-run with `python3.14 examples/plot_08_timings.py` and
  `./build/openmm-metatomic-spike --bench 80 build/harmonic.pt`.
- `examples/plot_10_scaling.py`: end-to-end timing as a function of atom
  count (3 to 60k atoms), on the `PythonForce` path, with and without a
  `vesin` neighbor list. See the timing baseline below.

### Fixed

- `bench_scaling.cpp` timed each of its four cases one after another with
  its own internal warmup, so whichever case ran second for a given atom
  count inherited an unearned memory-warmth advantage (first large
  allocation at a given size pays page-fault/malloc-arena-growth cost that
  later ones don't) — this produced physically impossible results
  (`MetatomicForce`/`Context` measuring *faster* than the direct call it
  wraps). Fixed by priming every case in round-robin before timing any of
  them. See the revised timing baseline below for the corrected numbers
  and what they actually show.
- `MetatomicEvaluator.cpp`'s core and torch evaluators each converted
  between `OpenMM::Vec3` and flat `double` arrays with a manual
  element-by-element loop, for both positions and forces, on every
  `computeForce` call. `Vec3` is exactly `{double data[3]}`
  (`static_assert`ed at the two call sites), so both conversions are now a
  single `memcpy`/range-construction; `MetatomicForceImpl::computeForce`
  also moves its result instead of copying it. A real, modest
  optimization, found while chasing the benchmark bug above — but not
  itself the explanation for the original (wrong) numbers.
- `CMakeLists.txt`'s TorchScript-backend detection only looked for a
  standalone top-level `metatomic_torch` Python package. metatomic-torch
  >=0.2 installs into the `metatomic.torch` namespace instead (no
  importable `metatomic_torch` module, despite the `metatomic_torch`
  dist-info); both probes (`openmm_metatomic_detect_prefixes` and
  `openmm_metatomic_import_torch`) now fall back to `import metatomic.torch`
  when the standalone import fails.
- The `openmm-metatomic-spike` C++ binary (and `OpenMMMetatomic` itself, M1)
  failed to compile against `metatomic-core`'s vendored C++ headers:
  `TensorBlock::release()` was private, and `TensorMap`/`Labels` had no
  `unsafe_from_ptr`/`unsafe_view_from_ptr` — APIs `metatomic-core`'s
  `system.hpp`/`model.hpp` call unconditionally. Root cause: the version
  floors were wrong on both sides.
  - `metatomic-core/CMakeLists.txt` required `find_package(metatensor 0.3 ...)`,
    a version metatensor has never released (upstream's newest is 0.2.5, and
    the vendored `metatensor` checkout here is at `metatensor-core` 0.2.0,
    76 commits behind `upstream/main` on the same, otherwise-unmodified
    branch). `find_package`'s version check also rejects a *lower* patch
    within the same minor (0.2.0 does not satisfy a request for 0.2.4
    either), so the un-patched 0.2.4 floor in `openmm-metatomic`'s own
    `CMakeLists.txt` was equally unsatisfiable here.
  - Fix: built `metatensor-core` v0.2.5 (an isolated `git worktree` off the
    `metatensor-core-v0.2.5` tag, not the shared vendored checkout other
    projects in this workspace build against) and confirmed its
    `metatensor.hpp` has the APIs `metatomic-core` needs. Lowered both
    version floors to the real, verified minimum: `metatomic-core/CMakeLists.txt`'s
    `REQUIRED_METATENSOR_VERSION` from `"0.3"` to `"0.2.5"`, and
    `openmm-metatomic/CMakeLists.txt`'s `find_package(metatensor ...)` from
    `0.2.4` to `0.2.5`. Verified the tightened floor still rejects the
    broken 0.2.0 config (a plain `0.2` floor would have silently let it
    back in) and that both `openmm-metatomic-spike` and `OpenMMMetatomic`
    build and `ctest` passes against a real 0.2.5 install.
  - This did not touch the shared `/metatensor` checkout other repos in
    this workspace (`metatrain`, `featomic`, `lammps`, `gromacs`, ...)
    build against — fast-forwarding that branch to pick up 0.2.5 is a
    separate, wider-blast-radius decision this session didn't make.
  - Build recipe (until a real metatensor-core >=0.2.5 lands in the shared
    checkout or its Python package): build `metatensor-core` from the
    `metatensor-core-v0.2.5` tag with plain CMake+cargo, install it
    somewhere, then configure `metatomic-core` and `openmm-metatomic` with
    `-Dmetatensor_DIR=<that install>/lib/cmake/metatensor` so it takes
    priority over the (older) `metatensor` Python package's config.

### Known issues

- `vesin.metatomic` warns that it was only tested against `metatomic.torch`
  0.1.3-0.2, and the installed version here is 0.2.0.dev1125 — ran fine in
  practice for this session's benchmarks, but worth re-checking once vesin
  publishes a compatible release.
- OpenMM's CUDA platform fails to load on this machine
  (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) — the pip-installed `openmm`
  wheel's CUDA kernels predate the driver/toolkit here. Unrelated to the
  plugin; CPU/Reference timings below are unaffected.

### Timing baseline: scaling with atom count (2026-09-17)

Recorded on a second machine (Linux 6.8.0-124-generic, AMD Ryzen Threadripper
PRO 5945WX 12-core, 125 GiB RAM, NVIDIA RTX 4070 Ti SUPER — GPU present but
OpenMM's CUDA platform doesn't load here, see Known issues). Python 3.12,
OpenMM 8.6.0.dev-0224ca2, Torch 2.13.0+cu126, metatomic.torch
0.2.0.dev1125+git.fe7ef0a, vesin (CPU neighbor lists). CPU platform.
`examples/plot_10_scaling.py`: independent-atom harmonic well, water
molecules tiled on a 6 Å cubic lattice (`vesin`'s 4 Å cutoff stays sparse),
10 energy+force evals after 1 warmup (5 for N > 3000), 10 Langevin steps for
N <= 3000.

| Atoms | createSystem | Context | eval, no NL | eval, vesin NL | direct TorchScript* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 17.6 ms | 1.1 ms | 0.696 ms | 1.281 ms | 14.8 ms |
| 30 | 13.0 ms | 0.8 ms | 0.723 ms | 1.243 ms | 14.8 ms |
| 300 | 13.9 ms | 0.9 ms | 0.750 ms | 1.373 ms | 14.8 ms |
| 3,000 | 18.2 ms | 0.9 ms | 0.879 ms | 1.988 ms | 15.0 ms |
| 15,000 | 39.3 ms | 2.1 ms | 1.802 ms | 5.707 ms | 17.4 ms |
| 60,000 | 118.4 ms | 4.9 ms | 4.634 ms | 17.253 ms | 19.2 ms |

\* `direct_energy_forces` reloads the TorchScript model on every call (same
caveat as the plot_08_timings.py baseline), so this column tracks load cost,
not eval cost; it is not comparable to the other two.

`PythonForce` eval without a neighbor list scales sub-linearly up to ~3k
atoms (fixed per-call Python/autograd overhead dominates) and then
approaches linear (0.88 ms at 3k -> 4.6 ms at 60k, a 5.2x cost for a 20x atom
count). The `vesin` neighbor list adds real but bounded overhead (roughly
2-4x the no-NL eval time across the range) rather than blowing up, which is
the expected behavior for a cell-list-based neighbor search at this density.
`createSystem` is the surprise: it grows faster than the eval cost (6.7x
from 3k to 60k atoms for a 20x atom count increase), so for large systems
the one-time `MLPotential.createSystem()` call, not the per-step cost, is
where wall-clock time concentrates for short runs.

### Timing baseline: native MetatomicForce vs. TorchScript (2026-09-17, revised)

First timing of the actual M1 deliverable: `MetatomicForce` evaluated
through a real `OpenMM::Context`, not the standalone spike or the
`PythonForce` fallback. Same machine as the scaling baseline above
(Threadripper PRO 5945WX, 12c/24t). `spike/bench_scaling.cpp`, Reference
platform, independent-atom harmonic well (`E = 0.5 sum ||r||^2`, k=1,
r0=0 — same model both backends evaluate, core built-in vs. an exported
`.pt` of the identical math). Energies checked against the analytic
solution before timing. Every number is the median of 3 independent
process launches (fresh interpreter/thread-pool state each time).

**This supersedes the numbers first posted under this heading**, which had
a real bug: the four cases (core/torch x direct/`Context`) were timed one
after another, each with its own internal warmup. The first large
allocation at a given atom count pays a page-fault/malloc-arena-growth cost
that later allocations of the same size don't, so whichever case happened
to run *second* for a given N inherited an unearned advantage from the
case that ran first. That's why the original table showed `Context`
*beating* a direct call at 15k+ atoms — a physical impossibility for code
that calls the direct path internally. Two independent fixes went in:

- **The benchmark**: `runFairly()` now primes every case for a given N
  (one round-robin pass through all of them) *before* timing any of them,
  so no case gets a memory-warmth advantage over another.
- **The plugin, while investigating**: `CoreEvaluatorImpl::compute` and the
  torch evaluator each did an O(N) manual element-by-element loop to
  convert between `OpenMM::Vec3` and flat `double` arrays for positions and
  forces. `Vec3` is exactly `{double data[3]}` (`static_assert`ed to catch
  a future change), so both directions are now a single `memcpy` /
  range-construction instead of N indexed writes, and
  `MetatomicForceImpl::computeForce` moves its result instead of copying
  it. This was a real, if modest, optimization independent of the
  benchmark bug — but it is **not** what caused the original numbers to
  look wrong; the ordering bug alone accounts for that.

#### Corrected numbers (median of 3 runs; default libtorch threading, 12 threads)

| Atoms | core, direct `execute_model` | core, `MetatomicForce`/`Context` | torch, direct forward+backward | torch, `MetatomicForce`/`Context` |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.0204 ms | 0.0222 ms | 0.2063 ms | 0.2157 ms |
| 30 | 0.0216 ms | 0.0235 ms | 0.2071 ms | 0.2175 ms |
| 300 | 0.0287 ms | 0.0331 ms | 0.2135 ms | 0.2225 ms |
| 3,000 | 0.0884 ms | 0.1095 ms | 0.2446 ms | 0.2744 ms |
| 15,000 | 0.567\* ms | 0.796\* ms | 1.021 ms | 1.174 ms |
| 60,000 | 1.255 ms | 1.676 ms | 1.897 ms | 2.480 ms |
| 200,000 | 5.095 ms | 7.508 ms | 3.680 ms | 6.787 ms |
| 500,000 | 15.690 ms | 23.280 ms | 3.641 ms | 12.977 ms |

\* the 15k row is the one place residual run-to-run variance survived the
fix: medians across 3 runs were 0.334, 0.567, 0.632 ms (core direct) and
0.437, 0.771, 0.796 ms (core+`Context`) — the ratio between them stayed
~1.3-1.4x every time, only the absolute scale moved. Not chased further.

**Findings, corrected.**

- **`MetatomicForce`/`Context` overhead is modest and roughly constant**:
  ~1.1x at small N, ~1.3-1.4x at large N, for *both* backends. The earlier
  "core pays a uniquely growing per-atom Context tax" claim does not hold
  up — that was the ordering artifact, not a backend asymmetry. There may
  still be a small real per-atom `Context` cost (the ratio does drift from
  1.1x to 1.35x), but it's nowhere near the 2x this session first reported.
- **metatomic-core wins convincingly up to tens of thousands of atoms**
  (3-10x faster than TorchScript below 15k atoms) but **TorchScript
  overtakes it for real, reproducibly, somewhere between 60k and 200k
  atoms** — not at 60k as first (wrongly) reported. By 500k atoms
  TorchScript's default 12-thread parallelism wins by more than 4x direct
  (3.6 ms vs. 15.7 ms) and nearly 2x through `Context` (13.0 ms vs.
  23.3 ms). metatomic-core's `execute_model` is single-threaded throughout;
  this is a threading effect, not a per-eval efficiency win — see the
  1-thread comparison below.
- **libtorch pinned to 1 thread does not close the gap at 500k atoms**:
  direct eval goes from 3.6 ms (12 threads) to 12.9 ms (1 thread) — still
  faster than core's 15.7 ms, meaning TorchScript's batched tensor ops
  beat metatomic-core's per-atom Rust loop at this size even single
  threaded, not only because of parallelism. At 60k atoms, though, 1
  thread is enough to lose to core again (2.87 ms vs. 1.26 ms direct) — so
  the crossover point itself is threading-dependent, somewhere between 60k
  and 200k on 12 threads, and higher again on 1.
- Net for the M3 go/no-go in ROADMAP.md: metatomic-core is the right
  default for anything from single molecules up to typical ML/MM QM
  regions (tens of thousands of atoms); TorchScript's batched execution
  becomes the better bet only once systems get very large by MetatomicForce
  standards. A foundation model's own per-atom cost will usually dominate
  either backend's plumbing well before this crossover matters — this
  measures relative overhead of the two evaluation paths, not which model
  to pick.

### Timing baseline: MetatomicForce settings matrix (2026-09-17)

Native plugin through `OpenMM::Context` (`openmm-metatomic-bench-settings`),
this machine (Nobara / Ryzen 9 5900X). OpenMM C++ install exposes Reference
only (the pip wheel's CPU platform is a different `libOpenMM`). Median of
**three process launches**. Each case uses **one Context**: 8 warmup
evals + 10 warmup steps, then the median of 11 `getState(Energy|Forces)`,
then `Integrator::step(20)` with no per-step `getState`. 0.5 fs. Harmonic
rest is the origin. CUDA is `setDevice("cuda")` on the TorchScript model
with host-copy `CustomCPPForceImpl` (M3), not OpenMM's CUDA platform.
Figures: `examples/plots/plot_11_native_force.png`,
`plot_12_periodic_md.png`, `plot_13_settings.png`.

An earlier pass of this table used 3-atom / 8-water systems and a **new
Context for eval vs NVE vs NVT**. That mixed JIT/load into "eval" and
counted `getState` inside the MD loop, so Torch/PET-MAD eval looked
several times slower than a step. Those rows are superseded.

| Case | N | eval / ms | NVE ms/step | NVT ms/step | NVE drift kJ/mol |
| --- | ---: | ---: | ---: | ---: | ---: |
| harmonic core | 3000 | 0.839 | 0.860 | 0.973 | 0 |
| harmonic torch | 3000 | 0.342 | 0.367 | 0.468 | 0 |
| harmonic core | 15000 | 3.36 | 3.43 | 3.99 | 0 |
| harmonic torch | 15000 | 1.02 | 1.58 | 2.35 | 0 |
| harmonic core | 60000 | 15.4 | 15.4 | 17.6 | 0 |
| harmonic torch | 60000 | 2.94 | 3.58 | 6.08 | 0 |
| harmonic-nl core, vesin | 768 | 1.21 | 1.31 | 1.29 | 0 |
| harmonic-nl core, naive O(N²) | 768 | 1014 | 982 | 975 | 0 |
| harmonic-nl torch, vesin | 768 | 1.29 | 1.47 | 1.69 | 0 |
| harmonic-nl torch, naive O(N²) | 768 | 968 | 971 | 974 | 0 |
| harmonic-nl core, vesin | 3000 | 3.09 | 2.99 | 3.00 | 0 |
| harmonic-nl torch, vesin | 3000 | 3.59 | 3.91 | 3.68 | 0 |
| harmonic-nl core, vesin | 15000 | 10.5 | 10.7 | 11.8 | 0 |
| harmonic-nl torch, vesin | 15000 | 12.5 | 14.7 | 16.7 | 0 |
| SOAP-BPNN 32 waters, vesin | 96 | 3.92 | 4.27 | 4.94 | 0.018 |
| PET-MAD-XS 32 waters, cpu | 96 | 47.1 | 47.1 | 47.4 | 1.38 |
| PET-MAD-XS 32 waters, cuda | 96 | 17.2 | 20.1 | 12.7 | 1.39 |
| PET-MAD-XS 96 waters, cpu | 288 | 98.1 | 107 | 114 | 4.08 |
| PET-MAD-XS 96 waters, cuda | 288 | 16.9 | 16.6 | 34.2 | 4.10 |

On a hot Context, eval and NVE ms/step agree to tens of percent. vesin is
~800× faster than the naive pair list at 768 atoms (256 waters); the 2.7×
ratio measured on 24 atoms was a toy-system artifact. TorchScript's batched
well is already faster than metatomic-core at 3k atoms through
`MetatomicForce` (0.34 vs 0.84 ms) and about 5× faster at 60k. PET-MAD-XS
CUDA through the host-copy path **beats CPU** once the box is 32–96 waters
(~17 ms vs 47–98 ms); the 3-atom water result that had CUDA slower was
dominated by launch overhead. NVE drift on the harmonic well is numerical
zero at 3k–60k atoms; PET-MAD-XS drifts ~1.4 kJ/mol over 20 × 0.5 fs on 32
waters and ~4 kJ/mol on 96 waters.

Separate `openmm-metatomic-run-md` runs, same machine, same hot-Context
method:

| System | NVE drift | NVE ms/step | notes |
| --- | ---: | ---: | --- |
| harmonic cloud, 15k atoms, 50 × 0.5 fs | 9e-7 kJ/mol | 3.69 | core |
| harmonic cloud, 60k atoms, 30 × 0.5 fs | 1.6e-6 kJ/mol | 4.15 | torch |
| harmonic-nl 1000-water box, 40 × 0.5 fs | −7e-7 kJ/mol | 3.13 | core, vesin, PBC, 3000 atoms |
| SOAP-BPNN 32-water box, 20 × 0.5 fs | 0.018 kJ/mol | 4.85 | torch, 96 atoms |
| PET-MAD-XS 32-water box, 20 × 0.5 fs | 1.38 kJ/mol | 39.6 | torch/cpu, 96 atoms |
| PET-MAD-XS 96-water box, 12 × 0.5 fs | 5.48 kJ/mol | 111 | torch/cpu, 288 atoms |

Gallery `plot_11` is a 3,000-atom harmonic NVE (core and torch overlap)
plus PET-MAD-XS / SOAP-BPNN / toluene vacuum traces.
`plot_12` is 1,000 waters harmonic-nl and 32 waters PET-MAD-XS.

### Timing baseline: vs. the upstream openmm-ml PR (2026-09-17)

[metatensor/openmm-ml#1](https://github.com/metatensor/openmm-ml/pull/1)
("Add `MLPotential("metatomic")` backend for exported models") is a
materially newer `metatomicpotential.py` than the copy on this fork's
`main` used by every `PythonForce` number above: it adds NPT/virial
support, `non_conservative`/`variants`/`uncertainty_threshold`, and CUDA
neighbor lists. Compared with an isolated `git worktree` of the PR branch
(swapped in for the editable-install finder's module mapping, then swapped
back — no change to the shared `.venv`), same machine, same session:

- **No regression or improvement on the existing conservative-forces path**:
  the harmonic-well scaling numbers (`plot_10_scaling.py`, N=3 to 60k) match
  the `main` numbers above to within run-to-run noise, both with and
  without the `vesin` neighbor list. The PR adds capability, not speed, for
  the case every other numbers in this file already measured.
- **`non_conservative` forces are a real, substantial speedup on a real
  model**: PET-MAD-XS water (`WATER_NM`, single molecule, CPU), 40 evals
  after 5 warmup, 3 repeated process launches, `MLPotential("metatomic",
  ..., non_conservative="forces")` vs. the default conservative
  (`energy.backward()`) path:

  | Mode | eval (median) |
  | --- | ---: |
  | conservative (autograd) | 7.66-8.09 ms |
  | non\_conservative (direct force head) | 4.24-4.40 ms |

  PET-MAD-XS predicts forces directly as a second network head; skipping
  `pos.requires_grad_(True)` + `energy.backward()` for models that support
  this is close to a 1.8x speedup. `OpenMMMetatomic`'s own torch backend
  (`MetatomicEvaluator.cpp`) doesn't have an equivalent yet — that's M7
  ("non-conservative forces") in ROADMAP.md, not implemented.
- **The native plugin still wins even on a real model**: the same
  PET-MAD-XS water system through native `MetatomicForce`/`Context`
  (conservative forces, `spike/`-style standalone C++, `setDevice("cpu")`
  to match) evaluates in **5.64-5.77 ms** across 3 runs — faster than
  openmm-ml's conservative `PythonForce` path (7.66-8.09 ms) by
  ~1.3-1.4x, energy checked to match the Python path exactly
  (-1475.74 kJ/mol both ways). That gap is much smaller than on the
  harmonic well (where a real model's own cost dominates and the thin
  C++-vs-Python wrapper overhead is a smaller fraction of the total), but
  it doesn't close — the native path is strictly faster here too, just not
  non-conservative yet, so a non-conservative-capable native evaluator
  would likely beat both PythonForce modes above.

### Timing baseline (2026-09-16)

Recorded on this machine, not GitHub Actions. OpenMM 8.6.1 from pip has no CUDA
platform; GPU numbers are Torch CUDA only. The native `MetatomicForce` Python
API is not the path under test yet.

**Host.** Linux 7.2.4 (Nobara / Fedora 44), AMD Ryzen 9 5900X (12c/24t), 62 GiB
RAM, NVIDIA GeForce RTX 4060 Ti (16 GiB) + RTX 2070 (8 GiB), driver 595.99.02.

**Software.** Python 3.14.7, OpenMM 8.6.1.dev-b399af4 (platforms: Reference,
CPU, OpenCL), OpenMM-ML from `/home/ericb/metawork/openmm-ml`, Torch
2.13.0+cu130, metatensor 0.2.4, metatomic-torch 0.2.0.dev1167+git.4c6c588,
PET-MAD-XS v1.5.0 TorchScript (`models/pet-mad-xs-v1.5.0.pt`).

**Method.** Gallery script `examples/plot_08_timings.py`: 25 energy+force evals
after 2 warmup calls (8 evals / 1 warmup for PET-MAD), 30 Langevin steps
(`LangevinMiddleIntegrator`, 1 fs, 300 K) except PET-MAD (12 steps). Medians
from `time.perf_counter`. Direct Torch `eval` reloads the Atomistic model on
every call, so it is not comparable to an OpenMM `Context` that keeps the
`PythonForce` live. C++ spike: 80 `execute_model` evals after one warmup, in
process, metatomic-core `HarmonicModel`.

#### OpenMM-ML `PythonForce` (harmonic well unless noted)

| Case | Platform | createSystem | Context | first eval | eval (median) | MD |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| water vacuum ML (3 atoms) | CPU | 15.5 ms | 0.71 ms | 20.9 ms | 0.69 ms | 30 steps 25.3 ms (0.84 ms/step) |
| water vacuum ML | Reference | 12.7 ms | 0.67 ms | 1.79 ms | 0.50 ms | 30 steps 14.6 ms (0.49 ms/step) |
| water + vesin NL | CPU | 13.5 ms | 0.57 ms | 3.80 ms | 1.18 ms | 30 steps 43.3 ms (1.44 ms/step) |
| toluene vacuum ML (15 atoms) | CPU | 13.6 ms | 0.71 ms | 3.52 ms | 0.74 ms | 30 steps 28.8 ms (0.96 ms/step) |
| toluene ML/MM (6495 atoms, ML = 15) | CPU | 212 ms | 6.91 ms | 17.0 ms | 3.57 ms | skipped (`OpenMMException`) |
| PET-MAD-XS water | CPU | 146 ms | 0.63 ms | 399 ms | 19.9 ms | 12 steps 523 ms (43.6 ms/step) |
| water `createSystem(checkConsistency=True)` | — | 13.3 ms | — | — | — | — |

#### Direct TorchScript (no OpenMM)

| Case | Device | load | eval (median) |
| --- | --- | ---: | ---: |
| water vacuum harmonic | cpu | 13.8 ms | 13.8 ms |
| water vacuum harmonic | cuda | 11.9 ms | 14.4 ms |
| toluene vacuum harmonic | cpu | 12.7 ms | 13.7 ms |
| PET-MAD-XS water | cpu | — | 23.1 ms (min 9.45) |
| PET-MAD-XS water | cuda | — | 9.90 ms (min 9.49) |

PET-MAD first OpenMM eval (399 ms) includes model load inside `PythonForce`.
Later evals are ~20 ms on CPU. CUDA PET-MAD is a direct `evaluate_exported`
call, not an OpenMM platform.

#### C++ spike, metatomic-core (`--bench 80`)

| System | Atoms | ms/eval |
| --- | ---: | ---: |
| water | 3 | 0.153 |
| methane | 5 | 0.155 |
| co2 | 3 | 0.159 |
| carbon8 | 8 | 0.184 |
| water_pbc | 3 | 0.161 |

Core and TorchScript backends both match the analytic harmonic well and
finite-difference forces. On this well the C++ `execute_model` loop is
faster than Torch, not slower: ~0.15 ms/eval vs ~0.42 ms for a `.pt` loaded
once (consistency off) and ~0.69 ms through OpenMM `PythonForce`. The
~14 ms “direct Torch” numbers above reload the Atomistic model on every
call; they are not a backend comparison. Gallery `evaluate_core` is the
numpy analytic (4 µs), not `execute_model`. PET-MAD has no core path.
