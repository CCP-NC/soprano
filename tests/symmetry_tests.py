#!/usr/bin/env python
"""
Test code for the utils.py functions
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from ase import Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.properties.symmetry import *
import unittest
import numpy as np


class TestSymmetry(unittest.TestCase):

    def test_symdataset(self):

        # Create atoms objects according to a given symmetry group, check
        # that it works
        pos = np.zeros((6, 3))
        pos[0] = [0, 0.1, 0.2]
        pos[1] = [0, 0.3, 0.8]
        pos[2] = [0.4, 0.2, 0.6]
        pos[3:] = -pos[:3]

        symmA = Atoms(['C']*6, positions=pos, cell=[5]*3, pbc=[True]*3)

        symdata = SymmetryDataset.get(symmA)

        self.assertTrue(symdata['international'] == 'P-1')

if __name__ == '__main__':
    unittest.main()
