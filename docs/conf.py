"""Sphinx configuration.

The docs are a handful of hand-written pages plus a Sphinx-Gallery build over the
example scripts. Sphinx-Gallery turns each script into an HTML page with downloadable
``.py`` and ``.ipynb``; whether it also *executes* the script is controlled by the
``DOCS_RUN_EXAMPLES`` environment variable, read in the per-repo block below (see
``building.rst``). When examples run, ``abort_on_example_error`` makes a broken example a
build failure rather than a page with a traceback on it.

Every page can state a model in LaTeX (MathJax), cite its sources from
``references.bib`` (sphinxcontrib-bibtex, ``:footcite:``), and plot with matplotlib.

This file is shared by several repositories and is identical in each except for the
fenced per-repo block. Change anything outside the fence everywhere at once, never in
one repo.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- per-repo settings: the only lines that differ between repos ------------------
project = "regression-records"
release = importlib.metadata.version(project)
# The design notes and reference pages are Markdown.
EXTRA_EXTENSIONS: list[str] = ["myst_parser"]
# The Markdown pages link to their own headings ("[Paths](#paths)"). MyST only generates
# those anchors when asked, and under -W a missing one fails the build.
myst_heading_anchors = 3
# Directories importable by the docs build, relative to the repo root, beyond src/.
EXTRA_SYS_PATHS: list[str] = []
# Every example here is self-contained, so the gallery executes by default.
RUN_EXAMPLES = os.environ.get("DOCS_RUN_EXAMPLES", "1") == "1"
EXAMPLES_DIRS: str | list[str] = "../examples"
GALLERY_DIRS: str | list[str] = "auto_examples"
# Which scripts execute when examples run. The sphinx-gallery default, "/plot_",
# would skip every script here.
FILENAME_PATTERN = r".*\.py"
# Flat: every page imports its helpers from examples/_shared/, which sits beside it,
# and the tests run each page as a script from that directory.
SUBSECTION_ORDER: list[str] | None = None
# ---- end per-repo settings ---------------------------------------------------------

sys.path.insert(0, str(ROOT / "src"))
for extra_path in EXTRA_SYS_PATHS:
    sys.path.insert(0, str(ROOT / extra_path))

author = "Ronald MacEachern"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    *EXTRA_EXTENSIONS,
]

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = project

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Attributes sections become :ivar: fields; as directives they collide with the
# class attributes autodoc already lists under undoc-members.
napoleon_use_ivar = True
# Objects are re-exported from the package root, so a bare :class:`X` reference has two
# valid targets. That is not an error worth failing the build for.
suppress_warnings = ["ref.python"]

# One bibliography for the whole site. Examples cite with :footcite: and close with
# .. footbibliography::, so each page carries only the references it uses. The file
# exists even when nothing cites it: bibtex warns on a missing file, and -W fails.
bibtex_bibfiles = ["references.bib"]

# Intersphinx fetches the inventories over the network, and under -W a failed fetch is a
# failed build. DOCS_INTERSPHINX=0 builds offline, with numpy/Python names left unlinked.
intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
    }
    if os.environ.get("DOCS_INTERSPHINX", "1") == "1"
    else {}
)
intersphinx_timeout = 30


# The matplotlib reset is named by string so the config stays picklable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

sphinx_gallery_conf = {
    "examples_dirs": EXAMPLES_DIRS,
    "gallery_dirs": GALLERY_DIRS,
    "filename_pattern": FILENAME_PATTERN,
    # Files starting with an underscore are shared helpers, not pages. The
    # sphinx-gallery default ignores only __init__.py.
    "ignore_pattern": r"__init__\.py|(^|[\\/])_[^\\/]*\.py$",
    "plot_gallery": RUN_EXAMPLES,
    "subsection_order": SUBSECTION_ORDER,
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("_gallery_style.house_style",),
    "download_all_examples": False,
    "abort_on_example_error": True,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}
