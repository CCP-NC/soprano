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
from soprano.collection.generate import airssGen
import unittest
import numpy as np


_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")

class TestGenerate(unittest.TestCase):

    def test_airss(self):

        to_gen = 10

        # Load the Al.cell file
        agen = airssGen(os.path.join(_TESTDATA_DIR, 'Al.cell'), n=to_gen)
        acoll = AtomsCollection(agen)

        # Some basic checks
        self.assertEqual(acoll.length, to_gen)
        self.assertTrue(all([chem == 'Al8'
                             for chem in acoll.all.get_chemical_formula()]))

if __name__ == '__main__':
    unittest.main()