# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import re
import sys

import tomllib

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

if True:
    from changelog import generate_changelog

project = 'ESORM - ElasticSearch ORM'
# noinspection PyShadowingBuiltins
copyright = '2023, Adam Wallner'
author = 'Adam Wallner'

# Get version from pyproject.toml
with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), 'rb') as f:
    pyproject = tomllib.load(f)

# Generate changelog
generate_changelog('wallneradam/esorm', os.path.join(os.path.dirname(__file__), 'changelog.md'))

version = pyproject['project']['version']
release = pyproject['project']['version']

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

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
# These are included by introduction.md, advanced_usage.md and query_builder_usage.md
exclude_patterns += ['README.md', 'advanced.md', 'query_builder.md']

# user starts in light mode
default_dark_mode = False

# pdflatex stops at the emojis of the headings, xelatex just leaves them out (they are not in the font)
latex_engine = 'xelatex'
# Sphinx would use xindy with xelatex, makeindex is enough
latex_use_xindy = False


# -- Markdown preprocessing --------------------------------------------------
# The Markdown files are written for GitHub: the link targets are HTML anchors (<a id="..."></a>), and
# the README links to the files in docs/. Sphinx knows neither, so when the files are included, the
# anchors are converted to MyST targets, and the links between the files to target links, which MyST
# resolves across the documents. The badges are left out, the PDF can't show SVG images. The image paths
# are relative to the repository root, the static files are in the root of the built documentation.

_HTML_ANCHOR = re.compile(r'^<a id="([^"]+)"></a>[ \t]*$', re.MULTILINE)
_FILE_LINK = re.compile(r'\]\((?:docs/|\.\./)?(?:README|advanced|query_builder)\.md(#[^)\s]+)\)')
_ROOT_STATIC = re.compile(r'(src|href)="docs/_static/')
_BADGE = re.compile(r'^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)[ \t]*\n', re.MULTILINE)


def _prepare_markdown(_app, relative_path, _parent_docname, content):
    if relative_path.suffix == '.md':
        text = _HTML_ANCHOR.sub(r'(\1)=', content[0])
        text = _FILE_LINK.sub(r'](\1)', text)
        text = _ROOT_STATIC.sub(r'\1="_static/', text)
        content[0] = _BADGE.sub('', text)


def setup(app):
    app.connect('include-read', _prepare_markdown)
