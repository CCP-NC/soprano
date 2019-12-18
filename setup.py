#!/usr/bin/env python
"""
Soprano - A library for cracking crystals!
by Simone Sturniolo
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# The next line is removed because it causes issues in interpreting
# the package_data line, unfortunately
# from __future__ import unicode_literals

from setuptools import setup, find_packages
from soprano import __version__

long_description = """
Soprano is a Python library developed and maintained by the CCP for NMR
Crystallography as a tool to help scientists working with crystallography and
simulations to generate, manipulate, run calculations on and analyse large
data sets of crystal structures, with a particular attention to the output of
ab-initio random structure searching,
or AIRSS. (https://www.mtg.msm.cam.ac.uk/Codes/AIRSS)

It provides a number of functionalities to help automate many common tasks
in computational crystallography."""

if __name__ == '__main__':

    setup(name='Soprano',
          version=__version__,
          description='A Python library to crack crystals',
          long_description=long_description,
          url='https://ccp-nc.github.io/soprano/',
          author='Simone Sturniolo',
          author_email='simone.sturniolo@stfc.ac.uk',
          license='LGPL',
          classifiers=[
              # How mature is this project? Common values are
              #   3 - Alpha
              #   4 - Beta
              #   5 - Production/Stable
              'Development Status :: 4 - Beta',

              # Indicate who your project is intended for
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Chemistry',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Scientific/Engineering :: Information Analysis',

              # Pick your license as you wish (should match "license" above)
              'License :: OSI Approved :: GNU Library or Lesser General Public'
              ' License (LGPL)',

              # Specify the Python versions you support here. In particular,
              # ensure that you indicate whether you support Python 2, Python 3
              # or both.
              'Programming Language :: Python :: 3',
          ],
          keywords=['crystallography', 'ccpnc', 'computational chemistry'],
          packages=find_packages(),
          # For data files. Example: 'soprano': ['data/*.json']
          package_data={'soprano': ['data/*.json']},
          entry_points={
              'console_scripts': ['vasp2cell = '
                                  'soprano.scripts.vasp2cell:__main__',
                                  'soprano_submitter = '
                                  'soprano.hpc.submitter:__main__',
                                  'phylogen = '
                                  'soprano.scripts.phylogen:__main__',
                                  'magresaverage = '
                                  'soprano.scripts.msaverage:__main__']
          },
          use_2to3=False,
          convert_2to3_doctests=[],
          # Requirements
          install_requires=[
              'numpy',
              'scipy',
              'ase'
          ],
          python_requires='>=3.1.*'
          )
