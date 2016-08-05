#!/usr/bin/env python
"""
Soprano - A library for cracking crystals!
by Simone Sturniolo et al.
Copyright whatever whatever (TODO)

v 0.5
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# The next line is removed because it causes issues in interpreting 
# the package_data line, unfortunately
# from __future__ import unicode_literals

from setuptools import setup

setup(name='Soprano',
      version='0.1',
      packages=['soprano'],
      # For data files. Example: 'soprano': ['data/*.json']
      package_data={'soprano': ['data/*.json']},
      # For scripts - just put the paths
      scripts=['scripts/phylogen.py', 'scripts/vasp2cell.py'],
      )
