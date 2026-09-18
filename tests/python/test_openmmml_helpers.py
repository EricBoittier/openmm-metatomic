"""Unit-test the native OpenMM-ML adapter without a built plugin."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python" / "openmmmetatomic"


def _load_openmmml():
    pkg_name = "openmmmetatomic_src"
    if pkg_name in sys.modules:
        return sys.modules[f"{pkg_name}.openmmml"]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(SRC)]
    sys.modules[pkg_name] = pkg

    swig = types.ModuleType(f"{pkg_name}.openmmmetatomic")

    class MetatomicForce:
        def __init__(self, path):
            self.path = path
            self.calls = []

        def __getattr__(self, name):
            def _call(*args, **kwargs):
                self.calls.append((name, args, kwargs))

            return _call

    swig.MetatomicForce = MetatomicForce
    sys.modules[f"{pkg_name}.openmmmetatomic"] = swig

    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.openmmml",
        SRC / "openmmml.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


openmmml = _load_openmmml()


def test_resolve_variants_spreads_energy_default():
    resolved = openmmml._resolve_variants({"energy": "pbe"})
    assert resolved["energy"] == "pbe"
    assert resolved["energy_uncertainty"] == "pbe"
    assert resolved["non_conservative_force"] == "pbe"


def test_resolve_variants_renames_deprecated_forces():
    with pytest.warns(UserWarning, match="deprecated"):
        resolved = openmmml._resolve_variants({"non_conservative_forces": "direct"})
    assert resolved["non_conservative_force"] == "direct"
    assert "non_conservative_forces" not in resolved


def test_resolve_variants_rejects_both_force_names():
    with pytest.warns(UserWarning, match="deprecated"):
        with pytest.raises(ValueError, match="you can not specify both"):
            openmmml._resolve_variants(
                {"non_conservative_force": "a", "non_conservative_forces": "b"}
            )


def test_spin_value_prefers_args():
    assert openmmml._spin_value({"spin_multiplicity": 3}, {"spin": 1}) == 3
    assert openmmml._spin_value({}, {"multiplicity": 2}) == 2
    assert openmmml._spin_value({}, {}) == 1.0


def test_resolve_pbc_from_user_and_topology():
    class Topology:
        def getPeriodicBoxVectors(self):
            return (1, 0, 0)

    class System:
        def usesPeriodicBoundaryConditions(self):
            return False

    assert openmmml._resolve_pbc({}, Topology(), System()) == [True, True, True]
    assert openmmml._resolve_pbc(
        {"pbc": (True, False, True)}, Topology(), System()
    ) == [True, False, True]
    with pytest.raises(ValueError, match="length-3"):
        openmmml._resolve_pbc({"pbc": (True, False)}, Topology(), System())


def test_impl_rejects_bad_non_conservative():
    with pytest.raises(ValueError, match="non_conservative"):
        openmmml.MetatomicNativePotentialImpl(
            "metatomic-native", "model.pt", non_conservative="nope"
        )


def test_impl_reports_short_range_mechanical_embedding():
    impl = openmmml.MetatomicNativePotentialImpl("metatomic-native", "model.pt")
    assert impl.getMLLongRange() is False
    assert impl.getSupportedEmbeddings() == []
