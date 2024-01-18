# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

from configparser import ConfigParser

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

if True:
    from changelog import generate_changelog

project = 'ESORM - ElasticSearch ORM'
# noinspection PyShadowingBuiltins
copyright = '2023, Adam Wallner'
author = 'Adam Wallner'

# Get version from setup.cfg
config = ConfigParser()
config.read('../setup.cfg')

# Generate changelog
changelog = generate_changelog('wallneradam/esorm')

version = config['metadata']['version']
release = config['metadata']['version']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme',
    'sphinx_rtd_dark_mode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_logo = '_static/img/esorm.svg'
html_favicon = '_static/img/favicon.ico'

source_suffix = ['.rst']

# user starts in light mode
default_dark_mode = False
