"""Sphinx configuration for pypolca."""

import sys
from pathlib import Path

# -- Project info -----------------------------------------------------------

project = "pypolca"
copyright = "2025, Marc-André Chénier"
author = "Marc-André Chénier"
release = "0.4.0"

# -- Path setup -------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- General ----------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "breathe",
    "sphinx_design",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["pypolca._core"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Breathe ----------------------------------------------------------------

breathe_projects = {"pypolca": "../_doxygen/xml/"}
breathe_default_project = "pypolca"
breathe_domain_by_extension = {"h": "cpp"}

# -- HTML output ------------------------------------------------------------

html_theme = "furo"
html_title = "pypolca"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Epilogue (injected into every page) ------------------------------------

rst_epilog = """
.. _poLCA: https://github.com/dlinzer/poLCA
.. _Eigen: https://eigen.tuxfamily.org/
.. _pybind11: https://pybind11.readthedocs.io/
"""
