# OpenMM-Metatomic

[![Tests](https://github.com/EricBoittier/openmm-metatomic/actions/workflows/tests.yml/badge.svg)](https://github.com/EricBoittier/openmm-metatomic/actions/workflows/tests.yml)
[![Docs](https://github.com/EricBoittier/openmm-metatomic/actions/workflows/docs.yml/badge.svg)](https://ericboittier.github.io/openmm-metatomic/)

Native [OpenMM](https://openmm.org) plugin for [Metatomic](https://docs.metatensor.org/metatomic/) models.

`MetatomicForce` stores serializable configuration. The model is loaded inside
the OpenMM process when a `Context` is created — this is **not** the OpenMM-ML
`PythonForce` bridge.

Docs: <https://ericboittier.github.io/openmm-metatomic/>

```python
from openmm import System
from openmmmetatomic import MetatomicForce

force = MetatomicForce("model.pt")          # TorchScript export
force.setBackend("auto")                    # auto | torch | core
force.setDevice("cuda")
force.setAtomicTypes([1, 1, 8])             # never inferred from masses
force.setExtensionsDirectory("./extensions")  # TorchScript .so deps, or core plugins

system = System()
system.addForce(force)
```

The core backend also accepts the built-in names `"harmonic"` and
`"harmonic-nl"` (in-process test models). A third-party core plugin is a
shared library (`.so` / `.dylib` / `.dll`) loaded from
`setExtensionsDirectory`; `modelPath` is then the name `load_model` should
ask that plugin for.

`.mta` is **not** a model file. It is a serialized
[`metatomic.torch.System`](https://docs.metatensor.org/metatomic/)
(positions, types, cell). See `tests/python/test_mta_system.py`, or
[metatomic-core's system round-trip](https://github.com/metatensor/metatomic/blob/main/metatomic-core/tests/system.cpp).

## Backends

| Backend | Selected when | Loads |
| --- | --- | --- |
| `torch` | `setBackend("torch")`, or `auto` on `.pt` / `.pth` | `metatomic_torch::load_atomistic_model` |
| `core` | `setBackend("core")`, or `auto` on anything else | `metatomic::load_plugin` + `load_model` / `execute_model` |

Torch is optional at compile time (`OPENMM_METATOMIC_TORCH`).
`MetatomicForceImpl` subclasses OpenMM `CustomCPPForceImpl`, so the same
evaluation path runs on Reference, CPU, CUDA, OpenCL, and HIP.

Requires [metatensor-core 0.2.5+](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.2.5).

## OpenMM-ML

The plugin is also an OpenMM-ML backend. Mixed ML/MM, `lambda_interpolate`,
and link atoms come from OpenMM-ML; this package only supplies `addForces`.

```python
from openmmml import MLPotential
import openmmmetatomic; openmmmetatomic.register()  # or the installed entry point

potential = MLPotential("metatomic-native", modelPath="model.pt")
mixed = potential.createMixedSystem(topology, mm_system, ml_atoms)
```

A non-conservative *stress* is requested and validated but cannot drive an
integrator (OpenMM has no virial path). NPT uses a `MonteCarloBarostat`.

## Examples

Sphinx gallery in `examples/`. PET-MAD-XS is vendored as
`models/pet-mad-xs-v1.5.0.pt` (~20 MB) from
[lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet).

```bash
python -m pip install -r docs/requirements.txt
make -C docs html          # docs/_build/html/index.html
```

## Build

```bash
cmake -S . -B build \
  -DOPENMM_DIR=$PREFIX \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DOPENMM_METATOMIC_TORCH=ON
cmake --build build
```

`OPENMM_DIR` is an OpenMM install with headers and `libOpenMM`. SWIG wrappers
(`from openmmmetatomic import MetatomicForce`) build when `swig` is on `PATH`.

```bash
PYTHONPATH=$PWD/build/python OPENMM_PLUGIN_DIR=$PWD/build \
  python -c "from openmmmetatomic import MetatomicForce"
```

C++ drivers (`openmm-metatomic-run-md`, `openmm-metatomic-bench-settings`,
`openmm-metatomic-bench-cuda`) load OpenMM platform plugins from `OPENMM_DIR`.
Point `OPENMM_METATOMIC_PLUGINS_DIR` or `--plugins-dir` elsewhere if needed;
otherwise they only see `Reference`.

## Tests and coverage

```bash
# after a plugin build
ctest --test-dir build --output-on-failure

# Python only (plugin tests skip if openmmmetatomic is not importable)
python -m pip install -r tests/requirements.txt
python -m pytest tests/python --cov=_openmm --cov-report=term-missing
```

`ctest` runs the C++ spikes (`spike-core`, `test-pairlist`, `test-features`,
`test-cuda`, …) and the Python suites in `tests/python/`
(serialization, native vs `PythonForce`, complex mixed ML/MM).

## Status

Milestones 0–8 are done (M8 without conda-forge). Pair lists are cached
behind a Verlet skin (`OPENMM_METATOMIC_NEIGHBOR_SKIN`, default 0.05 nm).
CUDA evaluation `copy_()`s into persistent device tensors (M4).

See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md) for the
feature list and timing baselines.
