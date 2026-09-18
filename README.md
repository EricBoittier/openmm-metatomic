# OpenMM-Metatomic

Native [OpenMM](https://openmm.org) plugin for [Metatomic](https://docs.metatensor.org/metatomic/) models.

```python
from openmm import System
from openmmmetatomic import MetatomicForce

force = MetatomicForce("model.pt")          # TorchScript, still supported
force = MetatomicForce("model.mta")         # metatomic-core plugin model
force.setBackend("auto")                    # auto | torch | core
force.setDevice("cuda")
force.setExtensionsDirectory("./extensions")
force.setCheckConsistency(False)
force.setAtomicTypes([1, 1, 8])             # explicit types; never inferred from masses

system = System()
system.addForce(force)
```

This is **not** the OpenMM-ML `PythonForce` bridge. `MetatomicForce` stores
serializable configuration. The model is loaded inside the OpenMM plugin process
when a `Context` is created.

## Backends

The evaluator talks to **metatomic-core** (`<metatomic.hpp>`) and keeps a
**TorchScript** path for exported `.pt` models.

| Backend | How it is selected | Loader |
| --- | --- | --- |
| `core` | `setBackend("core")`, or `auto` for non-`.pt` paths | `metatomic::load_plugin` / `load_model` → `execute_model` |
| `torch` | `setBackend("torch")`, or `auto` for `.pt`/`.pth` | `metatomic_torch::load_atomistic_model` |

Torch is an optional compile-time dependency (`OPENMM_METATOMIC_TORCH`). Existing
exported Atomistic models keep working; new engines can sit on the C API
(DLPack systems, explicit `Gradients::Positions`) without linking LibTorch.

Requires [metatensor-core 0.2.5+](https://github.com/metatensor/metatensor/releases/tag/metatensor-core-v0.2.5)
and the `metatomic-core` branch of metatomic.

`MetatomicForceImpl` subclasses OpenMM `CustomCPPForceImpl`, so the same
evaluation path runs on Reference, CPU, CUDA, OpenCL, and HIP. The first CUDA
path copies through host memory.

## Proof of concept (M0)

```bash
cmake -S . -B build -DOPENMM_METATOMIC_TORCH=ON
cmake --build build
./build/openmm-metatomic-spike            # in-process HarmonicModel via metatomic-core
./build/openmm-metatomic-spike harmonic.pt  # TorchScript .pt via metatomic-torch
./build/openmm-metatomic-bpnn soap-bpnn.pt  # SOAP-BPNN core vs TorchScript
```

The spike builds a `metatomic::System` from DLPack arrays, calls
`execute_model` with position gradients, and checks energy/forces against the
analytic harmonic well and finite differences on water, methane, CO2, an
eight-atom carbon cluster, and periodic water. The `.pt` path uses the same
numbers through `load_atomistic_model`.

## Examples

A Sphinx gallery lives in `examples/`. Built HTML is at
<https://ericboittier.github.io/openmm-metatomic/>.

```bash
python3.14 -m pip install -r docs/requirements.txt
make -C docs html
# output: docs/_build/html/index.html
```
PET-MAD-XS is a first-class example (`examples/plot_05_petmad.py`). The
repo vendors `models/pet-mad-xs-v1.5.0.pt` (~20 MB) from
[lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet). See
[models/README.md](models/README.md).

## Build

```bash
cmake -S . -B build \
  -DOPENMM_DIR=$PREFIX \
  -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build build
```

`OPENMM_DIR` should point at an OpenMM install that provides headers and
`libOpenMM`. Metatomic and metatensor are found via `CMAKE_PREFIX_PATH`.

SWIG wrappers (`from openmmmetatomic import MetatomicForce`) build when `swig`
is on `PATH` (`pip install swig`). After `cmake --build`, the module is in
`build/python`:

```bash
PYTHONPATH=$PWD/build/python OPENMM_PLUGIN_DIR=$PWD/build \
  python -c "from openmmmetatomic import MetatomicForce"
# or: cmake --build build --target PythonInstall
```

## Status

Milestones 0–3 and 5–7 are done, M4 is declined on measurement.
`MetatomicForce` runs through a real
`OpenMM::Context` (energy, conservative forces, periodic systems, vesin pair
lists, both backends) and now covers the rest of what
`MLPotential("metatomic")` does: particle subsets, per-axis PBC,
non-conservative forces, output variants, energy uncertainty, and charge/spin
inputs. The plugin also *is* an OpenMM-ML backend:

```python
from openmmml import MLPotential
import openmmmetatomic; openmmmetatomic.register()  # or the entry point

potential = MLPotential("metatomic-native", modelPath="model.pt")
mixed = potential.createMixedSystem(topology, mm_system, ml_atoms)
```

so mixed ML/MM systems, `lambda_interpolate` and link atoms come from
OpenMM-ML unchanged. A non-conservative *stress* is requested and validated but
cannot drive an integrator (OpenMM has no virial path); NPT goes through a
`MonteCarloBarostat`.

`setDevice("cuda")` runs the model on the GPU on any OpenMM platform —
`CustomCPPForceImpl` exchanges host buffers either way. PET-MAD-XS on an
RTX 4060 Ti is 2.8x faster than CPU at 96 atoms and 8.6x at 288, while the
host↔device copies cost 23–49 µs, so M4 (zero-copy DLPack) is declined rather
than implemented.

Pair lists are cached across steps: every neighbor-list request keeps its
`vesin` list alive behind a Verlet skin (`OPENMM_METATOMIC_NEIGHBOR_SKIN`, nm,
default 0.05, 0 to disable). Cached and fresh runs agree exactly. That is 24%
per step on a model cheap enough for the list to matter and 1–2% on
PET-MAD-XS, whose forward dominates everything else.

Python bindings (`from openmmmetatomic import MetatomicForce`) and NVE/NVT
drivers live in the plugin: `openmm-metatomic-run-md`,
`openmm-metatomic-bench-settings` and `openmm-metatomic-bench-cuda`. The
drivers register OpenMM's platform plugins from the build's `OPENMM_DIR`;
point `OPENMM_METATOMIC_PLUGINS_DIR` or `--plugins-dir` elsewhere if your
OpenMM lives somewhere else, or they will only find `Reference`. Building
needs `metatensor-core` >=0.2.5;
see "Fixed" in [CHANGELOG.md](CHANGELOG.md#unreleased) for the build recipe if
the vendored `metatensor` checkout is older. See [ROADMAP.md](ROADMAP.md).
Timing baselines (2026-09-16, 2026-09-17, including scaling with atom count,
native `MetatomicForce` vs. TorchScript, and the settings matrix) are in
[CHANGELOG.md](CHANGELOG.md).
