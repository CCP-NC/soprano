#!/usr/bin/env python
"""
Test code for the AtomsCollection class
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
from ase import Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
from soprano.analyse.phylogen import PhylogenCluster
from soprano.analyse.phylogen.genes import Gene
import unittest
import numpy as np

class TestPhylogen(unittest.TestCase):

    def test_instantiate(self):

        c1 = AtomsCollection([Atoms('C')])
        g1 = Gene('latt_abc', 1.0, {})
        p1 = PhylogenCluster(c1, [g1])

if __name__ == '__main__':
    unittest.main()