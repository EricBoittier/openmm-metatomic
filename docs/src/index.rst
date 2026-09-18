OpenMM-Metatomic
================

Native OpenMM plugin for exported Metatomic models. ``MetatomicForce`` stores
serializable configuration and evaluates energy plus conservative forces inside
the OpenMM process.

Backends
--------

* **core** — `metatomic-core <https://github.com/metatensor/metatomic>`_ C++ API
  (``load_model``, ``execute_model``, DLPack ``System``, explicit position
  gradients).
* **torch** — TorchScript ``.pt`` models via ``load_atomistic_model``, for
  existing exported Atomistic models.

``setBackend("auto")`` selects torch for ``.pt`` / ``.pth`` and core otherwise.

Devices and platforms
---------------------

``setDevice("cuda")`` moves the model to the GPU. The OpenMM platform is a
separate choice: ``CustomCPPForceImpl`` still hands the force host positions
and takes host forces back. The evaluator then ``copy_()`` into persistent
device tensors (positions, cell, forces, neighbor lists) instead of
``clone()`` on every step. PET-MAD-XS on an RTX 4060 Ti is 2.8x faster on CUDA
than on CPU at 96 atoms and 8.6x at 288. A wrap of OpenMM's CUDA ``posq`` would
mean a CUDA ``ForceImpl``; it is only worth it if ``getPositions`` beats the
model forward at ≥10k atoms. ``openmm-metatomic-bench-cuda`` reproduces the
timings (including a 1024-water box), and ``test-cuda`` checks that both
devices and all three platforms agree, including eight consecutive steps on
reused buffers.

Pair lists
----------

Both backends get their neighbors from `vesin
<https://github.com/Luthaf/vesin>`_, and each of the model's requests keeps its
list alive between evaluations so that vesin can cache the topology behind a
Verlet skin: it builds candidates out to ``cutoff + skin`` and reuses them until
an atom moves more than ``skin / 2``, tracking the cell too, so a
``MonteCarloBarostat`` is safe. Set ``OPENMM_METATOMIC_NEIGHBOR_SKIN`` (nm,
default 0.05) to tune it, or 0 to rebuild from scratch every step;
``OPENMM_METATOMIC_NEIGHBOR_LIST=naive`` falls back to the O(N²) loop. Caching
is worth about a quarter of the per-step cost for a model cheap enough that the
list matters, and 1–2% for PET-MAD-XS, where the forward dominates. Either way
the answer is identical, which ``test-pairlist`` checks over a run with several
rebuilds.

The C++ drivers register OpenMM's platform plugins from the ``OPENMM_DIR`` they
were built against; set ``OPENMM_METATOMIC_PLUGINS_DIR`` or pass
``--plugins-dir`` for an OpenMM installed elsewhere. Without them only
``Reference`` is available, since that is the one platform linked into
``libOpenMM``.

The gallery includes **PET-MAD-XS** (`lab-cosmo/upet`_), a **SOAP-BPNN**
twin of metatrain ``soap_bpnn`` (torch-spex SOAP + per-species SiLU MLP)
run in both metatomic-core C++ and TorchScript, **scaling** with atom
count, native ``MetatomicForce`` MD, and an OpenMM-ML feature tour of the
same PET-MAD-XS checkpoint (NVT/NPT, non-conservative forces, mixed
embedding, λ interpolation, link atoms).

.. _lab-cosmo/upet: https://huggingface.co/lab-cosmo/upet

Built HTML is at https://ericboittier.github.io/openmm-metatomic/ .

.. toctree::
   :maxdepth: 2

   install
   openmm_ml
   examples/index

