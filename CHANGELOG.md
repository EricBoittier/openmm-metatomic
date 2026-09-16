# Changelog

All notable changes to openmm-metatomic are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- GitHub Pages and a Docs workflow that builds the Sphinx gallery
  (<https://ericboittier.github.io/openmm-metatomic/>).
- SOAP-BPNN twin of metatrain `soap_bpnn` (legacy path): torch-spex Laplacian
  eigenstates + sphericart spherical harmonics + `SoapPowerSpectrum` contraction,
  then a per-species SiLU MLP. Core C++ uses sphericart and the dumped spline;
  checked against torch-spex, metatrain, and a matching TorchScript export.
  `./build/openmm-metatomic-bpnn build/soap-bpnn.pt`

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
