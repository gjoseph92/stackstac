# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import importlib.metadata
import datetime

DISTRIBUTION_METADATA = importlib.metadata.metadata("stackstac")

author = DISTRIBUTION_METADATA["Author"]
project = DISTRIBUTION_METADATA["Name"]
version = DISTRIBUTION_METADATA["Version"]
copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "jupyter_sphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
napoleon_use_param = True
# set_type_checking_flag = True
default_role = "py:obj"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_theme_options = {
    "globaltoc_maxdepth": -1,
}
html_context = {
    "display_github": True,
    "github_user": "gjoseph92",
    "github_repo": "stackstac",
}
# If False, source links to Bitbucket/Github/GitLab are shown
html_copy_source = False
# Don't suffix .ipynbs with .txt
html_sourcelink_suffix = ""

# Gotta fight fire with fire. nbsphinx injects their CSS directly into the page...
# https://github.com/spatialaudio/nbsphinx/blob/master/src/nbsphinx.py#L437-L683
# so we just inject after it to undo their settings that look bad in dark mode.
nbsphinx_prolog = f"""
.. raw:: html

    <style>
    @media (prefers-color-scheme: dark) {{
        /* input area */
        div.nbinput.container div.input_area {{
            border: unset;
            border-radius: unset;
        }}
    }}

    </style>

{{% set docname = env.doc2path(env.docname, base=False) %}}
.. note::
    You can run this notebook interactively: |Binder|, or view & download the original
    `on GitHub <https://github.com/gjoseph92/stackstac/blob/v{version}/docs/{{{{
        "../" + docname if docname.startswith("examples") else docname
    }}}}>`_.

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/gjoseph92/stackstac/v{version}?urlpath=lab/tree/docs/{{{{
        "../" + docname if docname.startswith("examples") else docname
    }}}}
"""


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

intersphinx_mapping = {
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pystac": ("https://pystac.readthedocs.io/en/latest/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "ipyleaflet": ("https://ipyleaflet.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
