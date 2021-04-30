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

project = "stackstac"
copyright = "2021, Gabe Joseph"
author = "Gabe Joseph"


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

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=False) %}
.. note::
    You can run this notebook interactively here: |Binder|, or view & download the original `on Github <https://github.com/gjoseph92/stackstac/blob/main/docs/{{ docname }}>`_.

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/gjoseph92/stackstac/main?urlpath=lab/tree/docs/{{ docname }}
"""  # noqa: E501


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "insipid"

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


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = {
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pystac": ("https://pystac.readthedocs.io/en/latest/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
}
