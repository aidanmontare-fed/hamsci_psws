# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hamsci_psws'
copyright = '2022, Nathaniel A. Frissell and Kristina V. Collins'
author = 'Nathaniel A. Frissell and Kristina V. Collins'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    # 'sphinx.ext.doctest'
]

autodoc_default_options = {
    'undoc-members': True, # set this to false if you do not want functions without docstrings to appear in the docs
    'show-inheritance': True,
}

# don't seem to need this to get desired functionality
# autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# Nothing static yet
# html_static_path = ['_static']

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

import inspect
import sys
from os.path import relpath, dirname
import hamsci_psws

# from numpy, with modifications
# retrieved 2022-08-31

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None
    lineno = None

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        # Ignore re-exports as their source files are not within the hamsci_psws repo
        module = inspect.getmodule(obj)
        if module is not None and not module.__name__.startswith("hamsci_psws"):
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        fn = relpath(fn, start=dirname(hamsci_psws.__file__))

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in hamsci_psws.__version__:
        return "https://github.com/HamSCI/hamsci_psws/blob/main/hamsci_psws/%s%s" % (
           fn, linespec)
    else:
        return "https://github.com/HamSCI/hamsci_psws/blob/v%s/hamsci_psws/%s%s" % (
           hamsci_psws.__version__, fn, linespec)
