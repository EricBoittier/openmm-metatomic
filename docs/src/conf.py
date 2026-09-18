import os
import sys
from datetime import datetime

from sphinx_gallery.sorting import FileNameSortKey

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.join(ROOT, "build", "python"))
os.environ.setdefault("OPENMM_METATOMIC_ROOT", ROOT)
os.environ.setdefault("OPENMM_PLUGIN_DIR", os.path.join(ROOT, "build"))
_mtt = (
    "/home/ericb/metawork/atomistic-cookbook/examples/"
    "metatomic-hourglass/.hourglass-env/bin/mtt"
)
if os.path.isfile(_mtt):
    os.environ.setdefault("OPENMM_METATOMIC_MTT", _mtt)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SPHINX_GALLERY_RUNNING", "1")

project = "openmm-metatomic"
author = "Eric D. Boittier"
copyright = f"{datetime.now().year}, Stanford University and the Authors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "OpenMM-Metatomic"
html_copy_source = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

def _native_plugin_importable() -> bool:
    built = os.path.join(ROOT, "build", "python")
    if built not in sys.path:
        sys.path.insert(0, built)
    try:
        import openmmmetatomic  # noqa: F401
    except ImportError:
        return False
    return True


_ignore = r"^_|README"
if not _native_plugin_importable():
    # Docs CI does not compile the plugin; those examples stay in the tree
    # and run locally once ``build/python`` exists.
    _ignore += r"|plot_11_|plot_12_|plot_13_|plot_16_|plot_17_"

sphinx_gallery_conf = {
    "examples_dirs": os.path.join(ROOT, "examples"),
    "gallery_dirs": os.path.join(ROOT, "docs", "src", "examples"),
    "filename_pattern": r"plot_",
    "ignore_pattern": _ignore,
    "copyfile_regex": (
        r"_harmonic\.py|_openmm\.py|_petmad\.py|_bpnn\.py|_native\.py|"
        r"_complex_mixed\.py"
    ),
    "within_subsection_order": FileNameSortKey,
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "backreferences_dir": None,
    "download_all_examples": False,
    "plot_gallery": True,
    "min_reported_time": 1,
}
