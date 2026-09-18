"""OpenMM-ML backend backed by the native ``MetatomicForce``.

``MLPotential("metatomic-native")`` builds the same Systems as
``MLPotential("metatomic")``, but the model is evaluated by the C++ plugin
instead of an ``openmm.PythonForce``. Everything OpenMM-ML layers on top --
mechanical embedding, ``lambda_interpolate``, link atoms -- only ever calls
``addForces``, so it comes along unchanged.

Register the backend with :func:`register`, or let the ``openmmml.potentials``
entry point do it when this package is installed.
"""

import warnings

from .openmmmetatomic import MetatomicForce

BACKEND_NAME = "metatomic-native"

_VALID_NC = (True, False, None, "forces", "stress")
_NC_MODES = {True: "both", "forces": "forces", "stress": "stress"}
_VARIANT_OUTPUTS = (
    "energy",
    "energy_uncertainty",
    "non_conservative_force",
    "non_conservative_stress",
)


def _resolve_variants(variants):
    """Spread a bare ``{"energy": v}`` over every output, as OpenMM-ML does."""
    variants = dict(variants or {})
    if "non_conservative_forces" in variants:
        warnings.warn(
            "variant name 'non_conservative_forces' is deprecated, please use "
            "'non_conservative_force' instead",
            stacklevel=3,
        )
        if "non_conservative_force" in variants:
            raise ValueError(
                "you can not specify both 'non_conservative_force' and "
                "'non_conservative_forces' in `variants`"
            )
        variants["non_conservative_force"] = variants.pop("non_conservative_forces")
    default = variants.get("energy")
    return {
        key: variants.get(key, default)
        for key in _VARIANT_OUTPUTS
        if variants.get(key, default) is not None
    }


def _resolve_pbc(args, topology, system):
    periodic = (
        topology.getPeriodicBoxVectors() is not None
        or system.usesPeriodicBoundaryConditions()
    )
    user_pbc = args.get("pbc")
    if user_pbc is None:
        return [periodic] * 3
    flags = [bool(x) for x in user_pbc]
    if len(flags) != 3:
        raise ValueError("pbc must be a length-3 sequence of booleans")
    return flags


def _spin_value(args, info, default=1.0):
    for name in ("spinMultiplicity", "spin_multiplicity", "multiplicity"):
        if name in args:
            return args[name]
    for name in ("spin_multiplicity", "spinMultiplicity", "multiplicity", "spin"):
        if name in info:
            return info[name]
    return default


class MetatomicNativePotentialImplFactory:
    """Factory for :class:`MetatomicNativePotentialImpl`."""

    def createImpl(
        self,
        name,
        modelPath,
        device=None,
        extensionsDirectory=None,
        checkConsistency=False,
        non_conservative=False,
        variants=None,
        uncertainty_threshold=0.1,
        backend="auto",
        **args,
    ):
        return MetatomicNativePotentialImpl(
            name,
            modelPath,
            device,
            extensionsDirectory,
            checkConsistency,
            non_conservative,
            variants,
            uncertainty_threshold,
            backend,
        )


class MetatomicNativePotentialImpl:
    """Evaluate an exported metatomic model through the native OpenMM plugin.

    The keyword arguments match ``MLPotential("metatomic")``, plus ``backend``
    to choose between the TorchScript (``"torch"``) and metatomic-core
    (``"core"``) evaluators.

    >>> potential = MLPotential(
    ...     "metatomic-native",
    ...     modelPath="exported-model.pt",
    ...     non_conservative="forces",
    ...     uncertainty_threshold=0.1,
    ... )
    >>> system = potential.createSystem(topology)
    """

    def __init__(
        self,
        name,
        modelPath,
        device=None,
        extensionsDirectory=None,
        checkConsistency=False,
        non_conservative=False,
        variants=None,
        uncertainty_threshold=0.1,
        backend="auto",
    ):
        if non_conservative not in _VALID_NC:
            raise ValueError(
                f"non_conservative must be one of {list(_VALID_NC)}, "
                f"got {non_conservative!r}"
            )
        self.name = name
        self.modelPath = modelPath
        self.device = device
        self.extensionsDirectory = extensionsDirectory
        self.checkConsistency = checkConsistency
        self.non_conservative = non_conservative
        self.variants = variants
        self.uncertainty_threshold = uncertainty_threshold
        self.backend = backend

    def addForces(self, topology, system, atoms, forceGroup, **args):
        if any(atom.element is None for atom in topology.atoms()):
            raise ValueError("All atoms in the Topology must have elements defined.")
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = [int(i) for i in atoms]
            includedAtoms = [includedAtoms[i] for i in atoms]

        force = MetatomicForce(self.modelPath)
        force.setAtomicTypes([atom.element.atomic_number for atom in includedAtoms])
        if atoms is not None:
            force.setParticles(atoms)
        if self.device is not None:
            force.setDevice(str(self.device))
        if self.extensionsDirectory is not None:
            force.setExtensionsDirectory(str(self.extensionsDirectory))
        force.setCheckConsistency(bool(self.checkConsistency))
        force.setBackend(self.backend)
        force.setNonConservative(_NC_MODES.get(self.non_conservative, ""))
        for output, variant in _resolve_variants(self.variants).items():
            force.setVariant(output, variant)
        if self.uncertainty_threshold is not None:
            force.setUncertaintyThreshold(float(self.uncertainty_threshold))

        info = dict(args.get("info") or {})
        force.setCharge(float(args.get("charge", info.get("charge", 0.0))))
        force.setSpinMultiplicity(float(_spin_value(args, info)))
        force.setPeriodicDirections(*_resolve_pbc(args, topology, system))
        force.setForceGroup(forceGroup)
        system.addForce(force)

    def getSupportedEmbeddings(self):
        return []

    def getMLLongRange(self):
        # Metatomic models are short-ranged: everything they see comes from the
        # neighbor lists they request.
        return False


def register():
    """Register ``MLPotential("metatomic-native")``.

    The ``openmmml.potentials`` entry point does this automatically for an
    installed package; call it by hand when running from a checkout.
    """
    from openmmml.mlpotential import MLPotential

    MLPotential.registerImplFactory(BACKEND_NAME, MetatomicNativePotentialImplFactory())
    return BACKEND_NAME
