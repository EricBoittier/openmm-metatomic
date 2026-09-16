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

The gallery includes **PET-MAD-XS** (`lab-cosmo/upet`_): the Hugging Face
checkpoint is downloaded and converted to TorchScript, then run through
OpenMM-ML as a pure-ML water molecule and as the ML region of toluene in
explicit solvent.

.. _lab-cosmo/upet: https://huggingface.co/lab-cosmo/upet

.. toctree::
   :maxdepth: 2

   examples/index
