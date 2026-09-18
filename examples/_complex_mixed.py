"""Builders for a complex mixed ML/MM System.

The point is to give the native plugin every conventional force-field term
OpenMM-ML will actually leave in a mixed System, plus the subset / order /
link-atom / interpolation combinations that have historically been where
gather/scatter and mechanical embedding go wrong.

Amber19 on ACE-ALA-NME in explicit water already has bonds, angles, proper and
improper periodic torsions, a CMAP, PME, rigid-water constraints and a
CMMotionRemover. CHARMM36 adds NBFix (``CustomNonbondedForce``) and 1-4 LJ
(``CustomBondForce``). Ions, an injected Ryckaert–Bellemans torsion, and a
large water-as-ML region sit on top of that.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from _openmm import mixed_system, openmm_ml_data

# Cheap enough that 2,000+ ML atoms still belong in ctest; pair lists are
# requested so the cache and vesin path run, then multiplied by zero so native
# and PythonForce still have to agree exactly on the harmonic well.
CUTOFF_A = 4.0
K_KJ = 1.0


class HarmonicProbe(torch.nn.Module):
    """Independent-atom well ``E = k Σ|r|²`` plus a zeroed neighbor-list term.

    The well has no reference geometry, so one checkpoint covers a 22-atom
    peptide subset and a 2,000-atom water box. Pair lists are requested (and
    then multiplied by zero) so vesin still runs; charge and spin shift the
    energy when the caller actually sets them.
    """

    def __init__(self, k: float = K_KJ):
        super().__init__()
        self.k = float(k)
        self._nl = NeighborListOptions(CUTOFF_A, False, True)
        self.charge_coeff = 10.0
        self.spin_coeff = 100.0

    def requested_neighbor_lists(self):
        return [self._nl]

    def requested_inputs(self):
        return {
            "charge": ModelOutput(unit="e", sample_kind="system"),
            "spin_multiplicity": ModelOutput(unit="", sample_kind="system"),
        }

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        del selected_atoms, outputs
        energy = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype)
        for i, system in enumerate(systems):
            neighbors = system.get_neighbor_list(self._nl)
            charge = system.get_data("charge").block().values.reshape(())
            spin = system.get_data("spin_multiplicity").block().values.reshape(())
            energy[i] = (
                neighbors.values.sum() * 0.0
                + self.k * (system.positions**2).sum()
                + self.charge_coeff * charge
                + self.spin_coeff * (spin - 1.0)
            )
        block = TensorBlock(
            values=energy,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


def export_probe_model(path: str, atomic_types: Sequence[int]) -> str:
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="kJ/mol", sample_kind="system")},
        atomic_types=sorted(set(int(z) for z in atomic_types)),
        interaction_range=CUTOFF_A,
        length_unit="Angstrom",
        supported_devices=["cpu"],
        dtype="float64",
    )
    AtomisticModel(
        HarmonicProbe().eval(),
        ModelMetadata(name="gallery-complex-mixed-harmonic"),
        capabilities,
    ).save(path)
    return path


def alanine_path():
    data = openmm_ml_data()
    if data is None:
        return None
    path = data / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb"
    return path if path.is_file() else None


def ethanol_paths():
    data = openmm_ml_data()
    if data is None:
        return None
    pdb = data / "ethanol" / "ethanol.pdb"
    xml = data / "ethanol" / "ethanol.xml"
    if pdb.is_file() and xml.is_file():
        return pdb, xml
    return None


def inventory(system: mm.System, topology: Optional[app.Topology] = None) -> dict:
    """Count every force, constraint, and (if given) residue the System holds."""
    bonded = None
    if topology is not None:
        bonded = [set() for _ in range(system.getNumParticles())]
        for bond in topology.bonds():
            bonded[bond.atom1.index].add(bond.atom2.index)
            bonded[bond.atom2.index].add(bond.atom1.index)

    counts = Counter()
    details = {}
    for force in system.getForces():
        name = type(force).__name__
        counts[name] += 1
        if isinstance(force, mm.HarmonicBondForce):
            details["bonds"] = force.getNumBonds()
        elif isinstance(force, mm.HarmonicAngleForce):
            details["angles"] = force.getNumAngles()
        elif isinstance(force, mm.PeriodicTorsionForce):
            details["periodic_torsions"] = force.getNumTorsions()
            proper = improper = 0
            if bonded is not None:
                for i in range(force.getNumTorsions()):
                    a, b, c, d, *_ = force.getTorsionParameters(i)
                    if b in bonded[a] and c in bonded[b] and d in bonded[c]:
                        proper += 1
                    else:
                        improper += 1
            details["proper_torsions"] = proper
            details["improper_torsions"] = improper
        elif isinstance(force, mm.RBTorsionForce):
            details["rb_torsions"] = force.getNumTorsions()
        elif isinstance(force, mm.CMAPTorsionForce):
            details["cmaps"] = force.getNumTorsions()
            details["cmap_maps"] = force.getNumMaps()
        elif isinstance(force, mm.NonbondedForce):
            details["nonbonded_method"] = force.getNonbondedMethod()
            details["exceptions"] = force.getNumExceptions()
        elif isinstance(force, mm.CustomNonbondedForce):
            details["custom_nonbonded"] = force.getNumExclusions()
        elif isinstance(force, mm.CustomBondForce):
            details["custom_bonds"] = force.getNumBonds()
        elif isinstance(force, mm.CustomExternalForce):
            details["custom_external"] = force.getNumParticles()
        elif isinstance(force, mm.MonteCarloBarostat):
            details["barostat"] = True

    residues = defaultdict(list)
    elements = Counter()
    if topology is not None:
        for atom in topology.atoms():
            residues[atom.residue.name].append(atom.index)
            if atom.element is not None:
                elements[atom.element.symbol] += 1

    return {
        "particles": system.getNumParticles(),
        "constraints": system.getNumConstraints(),
        "forces": dict(counts),
        "details": details,
        "residues": {name: (idx[0], idx[-1], len(idx)) for name, idx in residues.items()},
        "elements": dict(elements),
    }


def require_terms(info: dict, required: Sequence[str], label: str):
    missing = [name for name in required if name not in info["forces"]]
    if missing:
        raise AssertionError(f"{label} is missing {missing}; inventory={info['forces']}")


def peptide_atoms(topology: app.Topology) -> List[int]:
    return [atom.index for atom in topology.atoms() if atom.residue.chain.index == 0]


def residue_atoms(topology: app.Topology, name: str) -> List[int]:
    return [atom.index for atom in topology.atoms() if atom.residue.name == name]


def water_atoms(topology: app.Topology) -> List[int]:
    return [atom.index for atom in topology.atoms() if atom.residue.name == "HOH"]


def ion_atoms(topology: app.Topology) -> List[int]:
    return [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name in ("NA", "CL", "Na+", "Cl-", "SOD", "CLA")
        or (atom.element is not None and atom.element.symbol in ("Na", "Cl"))
    ]


def replace_waters_with_nacl(modeller: app.Modeller, n_pairs: int = 2):
    """This OpenMM has no ``Modeller.addIons``; swap solvent molecules for Na/Cl."""
    waters = [residue for residue in modeller.topology.residues() if residue.name == "HOH"]
    if len(waters) < 2 * n_pairs:
        raise RuntimeError("not enough water to place salt")
    chosen = waters[: 2 * n_pairs]
    positions = modeller.getPositions()
    sites = []
    for residue in chosen:
        oxygen = next(atom for atom in residue.atoms() if atom.element.symbol == "O")
        sites.append(positions[oxygen.index])
    modeller.delete(chosen)
    extra = app.Topology()
    chain = extra.addChain()
    extra_pos = []
    for i, pos in enumerate(sites):
        if i < n_pairs:
            residue = extra.addResidue("NA", chain)
            extra.addAtom("NA", app.element.sodium, residue)
        else:
            residue = extra.addResidue("CL", chain)
            extra.addAtom("CL", app.element.chlorine, residue)
        extra_pos.append(pos)
    modeller.add(extra, extra_pos)


def load_alanine_box(ff_family: str = "amber19", add_salt: bool = True):
    """ACE-ALA-NME in TIP3P-FB water, optionally with NaCl, plus extra MM terms."""
    path = alanine_path()
    if path is None:
        raise FileNotFoundError("alanine-dipeptide test PDB not found")
    pdb = app.PDBFile(str(path))
    if ff_family == "amber19":
        forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    elif ff_family == "charmm":
        forcefield = app.ForceField("charmm36_2024.xml", "charmm36_2024/water.xml")
    else:
        raise ValueError(ff_family)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    if add_salt:
        replace_waters_with_nacl(modeller)
    topology = modeller.topology
    positions = modeller.getPositions()
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5 * unit.amu,
    )
    inject_extra_mm(system, topology)
    return topology, positions, system, forcefield


def inject_extra_mm(system: mm.System, topology: app.Topology):
    """Terms Amber/CHARMM will not put on this peptide, attached where they belong.

    A Ryckaert–Bellemans torsion on the backbone (so ``removeBonds`` has to
    classify it), and a custom external well on the first water oxygen, which
    lives entirely in the MM region for the peptide-as-ML cases.
    """
    peptide = peptide_atoms(topology)
    if len(peptide) >= 15:
        rb = mm.RBTorsionForce()
        # ACE C, ALA N, ALA CA, ALA C — a real phi-like quartet.
        rb.addTorsion(peptide[4], peptide[6], peptide[8], peptide[14], 0.4, 0.1, -0.2, 0.0, 0.0, 0.0)
        system.addForce(rb)
    waters = water_atoms(topology)
    if waters:
        ext = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        ext.addGlobalParameter("k", 0.5)
        ext.addPerParticleParameter("x0")
        ext.addPerParticleParameter("y0")
        ext.addPerParticleParameter("z0")
        # Pin the first solvent oxygen so a mixed evaluation still feels it.
        ext.addParticle(waters[0], [1.6, 1.6, 1.6])
        system.addForce(ext)
    return system


def rest_for_atoms(positions, atoms: Sequence[int]):
    xyz = np.array(positions.value_in_unit(unit.angstrom), dtype=np.float64)
    return xyz[list(atoms)]


def energy_forces(system, positions, box=None, platform_name="Reference", parameters=None):
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001),
        mm.Platform.getPlatformByName(platform_name),
    )
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    n = system.getNumParticles()
    pos = list(positions)
    if len(pos) < n:
        pos = pos + [mm.Vec3(0, 0, 0) * unit.nanometer] * (n - len(pos))
    context.setPositions(pos)
    if any(system.isVirtualSite(i) for i in range(n)):
        context.computeVirtualSites()
    if parameters:
        for name, value in parameters.items():
            context.setParameter(name, value)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    del context
    return energy, forces


def mixed(potential, topology, system, atoms, **kwargs):
    return mixed_system(potential, topology, system, atoms, **kwargs)


def native_forces(system):
    import openmmmetatomic

    return [
        openmmmetatomic.MetatomicForce.cast(force)
        for force in system.getForces()
        if openmmmetatomic.MetatomicForce.isinstance(force)
    ]


def format_inventory(info: dict) -> str:
    details = info["details"]
    parts = [
        f"N={info['particles']}",
        f"constraints={info['constraints']}",
        f"bonds={details.get('bonds', 0)}",
        f"angles={details.get('angles', 0)}",
        f"proper={details.get('proper_torsions', 0)}",
        f"improper={details.get('improper_torsions', 0)}",
        f"cmap={details.get('cmaps', 0)}",
        f"RB={details.get('rb_torsions', 0)}",
        f"NBFix={details.get('custom_nonbonded', 0)}",
        f"custom-1-4={details.get('custom_bonds', 0)}",
        f"external={details.get('custom_external', 0)}",
    ]
    return " ".join(parts)
