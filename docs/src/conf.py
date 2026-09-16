import os
from datetime import datetime

from sphinx_gallery.sorting import FileNameSortKey

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

templates_path = ["_templates"]
exclude_patterns = ["_build"]

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
    "within_subsection_order": FileNameSortKey,
    "remove_config_comments": True,
    "matplotlib_animations": False,
    "reset_modules_order": "both",
    "backreferences_dir": None,
    "download_all_examples": False,
    "plot_gallery": True,
}

os.environ.setdefault("MPLBACKEND", "Agg")
