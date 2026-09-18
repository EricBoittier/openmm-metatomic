Install
=======

There is no conda-forge package yet. Build the plugin, put the SWIG module on
``PYTHONPATH``, and point OpenMM at the plugin library.

Build
-----

Needs OpenMM headers and ``libOpenMM``, metatensor-core ≥ 0.2.5, metatomic, and
(for TorchScript models) a LibTorch that matches the ``metatomic.torch`` wheel.
``swig`` on ``PATH`` builds the Python module.

::

    cmake -S . -B build \
      -DOPENMM_DIR=$PREFIX \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DOPENMM_METATOMIC_TORCH=ON
    cmake --build build

The plugin library is ``build/libOpenMMMetatomic.so`` (or ``.dylib`` /
``.dll``). The Python package is ``build/python/openmmmetatomic``.

Run from the build tree
-----------------------

::

    export PYTHONPATH=$PWD/build/python
    export OPENMM_PLUGIN_DIR=$PWD/build
    python -c "from openmmmetatomic import MetatomicForce"

``OPENMM_PLUGIN_DIR`` is how OpenMM finds ``libOpenMMMetatomic``. Without it,
``from openmmmetatomic import MetatomicForce`` still imports, but a ``Context``
cannot construct the force.

``cmake --build build --target PythonInstall`` copies the module into the
active environment. The plugin library still has to be on
``OPENMM_PLUGIN_DIR`` (or OpenMM's plugin directory).

OpenMM-ML backend
-----------------

``MLPotential("metatomic-native")`` is registered in two ways:

* **installed package** — the ``openmmml.potentials`` entry point in
  ``python/setup.py`` registers it on import of ``openmmml``;
* **checkout** — call ``openmmmetatomic.register()`` once before constructing
  the potential.

::

    from openmmml import MLPotential
    import openmmmetatomic

    openmmmetatomic.register()
    potential = MLPotential("metatomic-native", modelPath="model.pt", device="cuda")
    system = potential.createSystem(topology)
    mixed = potential.createMixedSystem(topology, mm_system, ml_atoms)

The keywords match ``MLPotential("metatomic")``. ``backend`` is extra
(``"auto"``, ``"torch"``, or ``"core"``). See :doc:`openmm_ml`.

C++ drivers
-----------

``openmm-metatomic-run-md``, ``openmm-metatomic-bench-settings`` and
``openmm-metatomic-bench-cuda`` load OpenMM platform plugins from
``$OPENMM_DIR/lib/plugins``. Override with ``OPENMM_METATOMIC_PLUGINS_DIR`` or
``--plugins-dir``. Without that they only see ``Reference``.
