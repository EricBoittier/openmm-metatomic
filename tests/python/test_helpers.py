"""Gallery helpers that do not need the native plugin."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

pytest.importorskip("torch")
pytest.importorskip("metatomic.torch")

from _openmm import _example_script_path, mixed_system, save_figure  # noqa: E402


class _Impl:
    def __init__(self, long_range):
        self._long_range = long_range

    def getMLLongRange(self):
        return self._long_range


class _Potential:
    def __init__(self, long_range):
        self._impl = _Impl(long_range)
        self.kwargs = None

    def createMixedSystem(self, *args, **kwargs):
        self.kwargs = kwargs
        return "mixed"


def test_mixed_system_sets_ml_long_range_when_unknown():
    potential = _Potential(None)
    assert mixed_system(potential, "top", "sys", [0]) == "mixed"
    assert potential.kwargs["mlLongRange"] is False


def test_mixed_system_does_not_override_explicit_flag():
    potential = _Potential(None)
    mixed_system(potential, "top", "sys", [0], mlLongRange=True)
    assert potential.kwargs["mlLongRange"] is True


def test_mixed_system_skips_flag_when_backend_knows():
    potential = _Potential(False)
    mixed_system(potential, "top", "sys", [0])
    assert "mlLongRange" not in potential.kwargs


def test_save_figure_uses_script_path(tmp_path: Path):
    script = tmp_path / "plot_14_petmad_openmm_ml.py"
    script.write_text("# gallery stub\n")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    dest = save_figure(fig, script)
    plt.close(fig)
    assert dest == tmp_path / "plots" / "plot_14_petmad_openmm_ml.png"
    assert dest.is_file()


def test_example_script_path_without_file():
    path = _example_script_path()
    assert path.suffix == ".py"
