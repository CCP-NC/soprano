#!/usr/bin/env python
"""
Test code for the collection Generators
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
from ase import io, Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
from soprano.collection.generate import airssGen, linspaceGen
import unittest
import numpy as np


_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")

class TestGenerate(unittest.TestCase):

    def test_airss(self):

        to_gen = 10

        # Load the Al.cell file
        agen = airssGen(os.path.join(_TESTDATA_DIR, 'Al.cell'), n=to_gen)
        try:
            acoll = AtomsCollection(agen)
        except RuntimeError as e:
            if 'Buildcell' in str(e):
                # Then we just don't have the program
                print('WARNING - The AIRSS generator could not be tested as no '
                      'AIRSS installation has been found on this system.')
                return
            else:
                raise e

        # Some basic checks
        self.assertEqual(acoll.length, to_gen)
        self.assertTrue(all([chem == 'Al8'
                             for chem in acoll.all.get_chemical_formula()]))

    def test_linspace(self):

        a1 = Atoms('CO', [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]])
        a2 = Atoms('CO', [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]])

        lgen = linspaceGen(a1, a2, steps=5, periodic=False)
        lcoll = AtomsCollection(lgen)

        self.assertTrue(np.all(np.isclose(lcoll.all.get_positions()[:,1,1],
                                          np.linspace(0.2, 0.8, 5)
                                          )
        ))
        
        # With periodicity
        lgen = linspaceGen(a1, a2, steps=5, periodic=True)
        lcoll = AtomsCollection(lgen)

        self.assertTrue(np.all(np.isclose(lcoll.all.get_positions()[:,1,1],
                                          np.linspace(0.2, -0.2, 5)
                                          )
        ))

if __name__ == '__main__':
    unittest.main()