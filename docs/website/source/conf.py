# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Chess Bot'
copyright = '2024, Mark Zimmerman'
author = 'Mark Zimmerman'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinxcontrib.bibtex',
    ]
bibtex_bibfiles = ['refs.bib']


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    "nosidebar": "false",
    "description": "Machine Learning applied to chess so that I don't have to play anymore.",
    "github_button":True,
    'github_user': 'mzimm003',
    'github_repo': 'Chess',
    "extra_nav_links":{
        "Mark Zimmerman's Portfolio":"https://mzimm003.github.io",
        },
    "fixed_sidebar":True,
    "sidebar_width":"15%",
    "page_width":"90%"
}
html_css_files = [
    'custom.css',
]
numfig = True