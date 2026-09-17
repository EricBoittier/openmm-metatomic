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

sphinx_gallery_conf = {
    "examples_dirs": os.path.join(ROOT, "examples"),
    "gallery_dirs": os.path.join(ROOT, "docs", "src", "examples"),
    "filename_pattern": r"plot_",
    "ignore_pattern": r"^_|README",
    "copyfile_regex": r"_harmonic\.py|_openmm\.py|_petmad\.py|_bpnn\.py|_native\.py",
    "within_subsection_order": FileNameSortKey,
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "backreferences_dir": None,
    "download_all_examples": False,
    "plot_gallery": True,
    "min_reported_time": 1,
}
