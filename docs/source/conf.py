import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Points directly to your src folder

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Analyse des scripts de la series FRIENDS'
copyright = '2026, Charaf Edine BELGHITI JOUHRI, Ali ICHELMANN, Elouan PAUNA, Ismael MEFTOUH, Matthieu BETOUS, Nassr-Eddine BEKKAR, Virakyuth SAY, Oscar PUENTE GONZALEZ, Leo DUBOS, Nolawi GEBREKIRSTOS, Yoakin HAILSELASSIEA'
author = 'Charaf Edine BELGHITI JOUHRI, Ali ICHELMANN, Elouan PAUNA, Ismael MEFTOUH, Matthieu BETOUS, Nassr-Eddine BEKKAR, Virakyuth SAY, Oscar PUENTE GONZALEZ, Leo DUBOS, Nolawi GEBREKIRSTOS, Yoakin HAILSELASSIEA'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

latex_documents = [
    ('index', 'analysedesscriptsdelaseriesfriends.tex', 'Analyse des scripts de la series FRIENDS',
     'Charaf Edine BELGHITI JOUHRI \\and Ali ICHELMANN \\and Elouan PAUNA \\and Ismael MEFTOUH \\and Matthieu BETOUS \\and Nassr\\sphinxhyphen{}Eddine BEKKAR \\and Virakyuth SAY \\and Oscar PUENTE GONZALEZ \\and Leo DUBOS \\and Nolawi GEBREKIRSTOS \\and Yoakin HAILSELASSIEA', 'manual'),
]