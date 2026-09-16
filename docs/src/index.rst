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

The gallery includes **PET-MAD-XS** (`lab-cosmo/upet`_) and a **SOAP-BPNN**
twin of metatrain ``soap_bpnn`` (SOAP power spectrum + per-species SiLU MLP)
run in both metatomic-core C++ and TorchScript.

.. _lab-cosmo/upet: https://huggingface.co/lab-cosmo/upet

Built HTML is at https://ericboittier.github.io/openmm-metatomic/ .

.. toctree::
   :maxdepth: 2

   examples/index

