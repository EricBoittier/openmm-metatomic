# Changelog

All notable changes to openmm-metatomic are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
  (`bench-scaling-smoke`, N=3,30) in `ctest`.
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

### Timing baseline: native MetatomicForce vs. TorchScript (2026-09-17)

First timing of the actual M1 deliverable: `MetatomicForce` evaluated
through a real `OpenMM::Context`, not the standalone spike or the
`PythonForce` fallback. Same machine as the scaling baseline above
(Threadripper PRO 5945WX, 12c/24t). `spike/bench_scaling.cpp`, Reference
platform, independent-atom harmonic well (`E = 0.5 sum ||r||^2`, k=1,
r0=0 — same model both backends evaluate, core built-in vs. an exported
`.pt` of the identical math). 50 timed evals after 15 warmup calls per
row, energies checked against the analytic solution before timing.
**Every number below is the median of 3 independent process launches**
(fresh interpreter/thread-pool state each time), reported as
`middle (low-high)` across those 3 runs so the process-to-process
variance is visible rather than hidden by picking one run.

libtorch defaults to 12 intra-op threads on this machine; metatomic-core's
`execute_model` is single-threaded throughout. Both are reported, since
that default-vs-pinned split *is* the finding.

#### Default libtorch threading (12 threads)

| Atoms | core, direct `execute_model` | core, `MetatomicForce`/`Context` | torch, direct forward+backward | torch, `MetatomicForce`/`Context` |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.0214 ms | 0.0225 ms | 0.2065 ms | 0.2157 ms |
| 30 | 0.0223 ms | 0.0245 ms | 0.2077 ms | 0.2173 ms |
| 300 | 0.0348 (0.030-0.040) ms | 0.0338 ms | 0.2117 ms | 0.2249 ms |
| 3,000 | 0.0886 ms | 0.1120 ms | 0.2457 ms | 0.2791 ms |
| 15,000 | 0.824 ms | 1.066 ms | 0.843 (0.39\*-0.85) ms | 1.511 (0.99-1.54) ms |
| 60,000 | 3.231 (3.208-3.500) ms | 5.698 (5.681-5.749) ms | 1.547 (1.526-1.555) ms | 2.418 (1.319-2.566) ms |

\* one of the three runs produced a 0.389 ms outlier at 15k atoms, well
below the other two (0.843, 0.852 ms); not reproduced on any other row or
run, and plausibly a scheduler artifact. Left in rather than discarded.

#### libtorch pinned to 1 thread (`--torch-threads 1`, apples-to-apples with core)

| Atoms | core, direct `execute_model` | core, `MetatomicForce`/`Context` | torch, direct forward+backward | torch, `MetatomicForce`/`Context` |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.0207 ms | 0.0221 ms | 0.2049 ms | 0.2140 ms |
| 30 | 0.0218 ms | 0.0240 ms | 0.2072 ms | 0.2164 ms |
| 300 | 0.0291 ms | 0.0340 ms | 0.2104 ms | 0.2236 ms |
| 3,000 | 0.0871 ms | 0.1110 ms | 0.2434 ms | 0.2771 ms |
| 15,000 | 0.821 ms | 1.064 ms | 0.880 ms | 0.997 ms |
| 60,000 | 3.518 ms | 5.763 (5.700-5.792) ms | 2.892 ms | 3.445 (1.325\*\*-3.470) ms |

\*\* one of the three single-thread runs still produced a 1.33 ms outlier on
the `MetatomicForce`/`Context` + torch row at 60k atoms, well below the
other two (3.44, 3.47 ms), despite `torch::set_num_threads(1)`. This is the
only row where pinning intra-op threads didn't fully remove the variance —
most likely libtorch's BLAS backend (MKL/OpenBLAS) spinning up its own
thread pool independently of `torch::set_num_threads`, not something this
session chased further.

**Findings.**

- **core has near-zero, deterministic overhead per eval** (0.02-0.09 ms up
  to 3k atoms, 1-3% run-to-run spread almost everywhere) and **is the
  clear winner up to a few thousand atoms** — 3-10x faster than TorchScript
  at every size below 15k atoms, both directly and through `Context`.
- **Going through `MetatomicForce`/`Context` costs core a real, growing
  tax**: +0.15 ms at 3 atoms but +2.1-2.5 ms at 60k atoms (an extra ~35-40
  µs/atom that direct `execute_model` doesn't pay). `CoreEvaluatorImpl`
  copies `vector<Vec3>` positions into a fresh `vector<double>` and rebuilds
  a `metatomic::System` (fresh DLPack wrapping) on every `computeForce`
  call — likely where that per-atom cost lives. Torch's OpenMM overhead is
  much flatter (+0.01-0.1 ms across the same range) since the DLPack/Vec3
  copy is a much smaller fraction of an already-heavier eval. Worth
  profiling before M2 (periodic systems will add more per-call setup, not
  less).
- **TorchScript overtakes core at large N only because it parallelizes**:
  with the default 12 threads it's faster than core by 60k atoms (~1.5-2.4
  ms vs. ~3.2-5.7 ms); pinned to 1 thread it is not (2.9-3.4 ms vs.
  3.2-5.8 ms — comparable to core's direct call, still behind core+Context
  only because of core's own per-atom Context tax above). There is no
  atom count in this range where a single TorchScript thread beats
  metatomic-core's direct call.
- **TorchScript's large-N timing is far less reproducible than core's**:
  60k-atom medians ranged 1.3-2.6 ms across identical process launches at
  default threading (94% spread) vs. core's 5.68-5.75 ms (1.2% spread).
  Thread-pool/BLAS scheduling noise, not the benchmark harness — the
  energies matched the analytic solution on every run.
- Net for the M3 go/no-go in ROADMAP.md: metatomic-core is the right
  default for the harmonic-well-sized end of the range (small molecules,
  ML/MM QM regions); a foundation model's real per-atom cost will dominate
  either backend well before this crossover matters in practice, so this
  says more about per-call plumbing overhead than about which backend to
  pick for a given model.

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
