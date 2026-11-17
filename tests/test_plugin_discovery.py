"""Which copy of the plugin library gets loaded must not depend on the cwd.

The prologue SWIG puts at the top of ``openmmmetatomic.openmmmetatomic`` looks
for ``libOpenMMMetatomic.so`` in a list of directories. It used to include
``Path(os.environ.get("OPENMM_METATOMIC_ROOT", "")) / "build"``, which is the
relative ``build`` whenever that variable is unset, so importing the package
from a directory holding an unrelated ``build/`` loaded that library instead.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("openmm")
import openmmmetatomic  # noqa: E402  (checks the package imports at all)

LOADED = textwrap.dedent(
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import openmmmetatomic
    print("WARNINGS", len(caught))
    for line in open("/proc/self/maps"):
        if "libOpenMMMetatomic.so" in line:
            print("LIBRARY", line.split()[-1])
            break
    """
)


def _loaded_library(cwd):
    done = subprocess.run(
        [sys.executable, "-c", LOADED],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=os.environ,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    library, warnings_seen = None, None
    for line in done.stdout.splitlines():
        if line.startswith("LIBRARY"):
            library = line.split(None, 1)[1]
        elif line.startswith("WARNINGS"):
            warnings_seen = int(line.split()[1])
    return library, warnings_seen


@pytest.mark.skipif(
    not os.path.exists("/proc/self/maps"), reason="needs /proc"
)
def test_the_same_library_loads_from_any_directory(tmp_path):
    # A decoy laid out exactly like the build tree the old candidate list
    # picked up from the cwd. It has to be a loadable copy of the real
    # library: an invalid file raises inside loadPluginsFromDirectory and so
    # never gets mapped, which is why a junk decoy does not reproduce this.
    real, _ = _loaded_library(tmp_path.parent)
    assert real is not None, "could not find the real library to copy"
    decoy = tmp_path / "build"
    decoy.mkdir()
    shutil.copy2(real, decoy / "libOpenMMMetatomic.so")

    from_tmp, warned_tmp = _loaded_library(tmp_path)
    from_root, warned_root = _loaded_library(tmp_path.parent)

    assert from_tmp is not None, "the plugin library was never loaded"
    assert from_tmp == from_root, (
        "the loaded library depends on the working directory: "
        f"{from_tmp} next to a decoy build/, {from_root} without one"
    )
    assert str(decoy) not in from_tmp, "the decoy in ./build was loaded"
    assert warned_tmp == 0, "importing the package warned"
    assert warned_root == 0


def test_an_unset_root_contributes_no_relative_path():
    from pathlib import Path

    from openmmmetatomic.openmmmetatomic import _openmmmetatomic_plugin_candidates

    env = dict(os.environ)
    env.pop("OPENMM_METATOMIC_ROOT", None)
    env.pop("OPENMM_PLUGIN_DIR", None)
    saved = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        candidates = list(_openmmmetatomic_plugin_candidates())
    finally:
        os.environ.clear()
        os.environ.update(saved)

    assert candidates, "no candidate directories at all"
    for candidate in candidates:
        assert Path(candidate).is_absolute(), f"{candidate} is relative"
