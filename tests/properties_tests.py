#!/usr/bin/env python
"""
Test code for the AtomsProperty class and its children
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
from soprano.properties import AtomsProperty
import unittest
import numpy as np


class TestPropertyLoad(unittest.TestCase):

    def test_dummyprop(self):

        # Define a dummy derived property and test it

        class DummyProperty(AtomsProperty):

            default_name = 'dummy'
            default_params = {'mul': 2.0}

            @staticmethod
            def extract(s, mul):
                return s.positions.shape[0]*mul

        # Now two atoms objects to test it on
        a1 = Atoms('C')
        a2 = Atoms('CC')

        dummyDoubled = DummyProperty(name='doubledummy', mul=4)

        self.assertEqual(DummyProperty.get(a1), 2.0)
        self.assertEqual(dummyDoubled(a2), 8.0)

        # Also check that improper parameters are rejected
        self.assertRaises(ValueError, DummyProperty, wrong='this is wrong')

        # Test behaviour on a collection instead
        c1 = AtomsCollection([a1, a2])
        c2 = AtomsCollection([a2, a1])

        DummyProperty.get(c1, store_array=True)
        dummyDoubled(c2, store_array=True)
        self.assertTrue(np.all(c1.get_array('dummy') == [2, 4]))
        self.assertTrue(np.all(c2.get_array('doubledummy') == [8, 4]))

    def test_basicprop(self):

        from soprano.utils import cart2abc
        from soprano.properties.basic import LatticeCart, LatticeABC, CalcEnergy

        cell = np.array([[1, 0, 0], [0, 2, 0], [0, 0.5, 3.0]])
        a = Atoms(cell=cell)
        linLatt = LatticeCart(shape=(9,))
        degLatt = LatticeABC(deg=True)

        ans = cell.reshape((9,))
        self.assertTrue(np.all(linLatt(a) == ans))
        ans = cart2abc(cell)
        ans[1, :] *= 180.0/np.pi
        self.assertTrue(np.all(degLatt(a) == ans))

if __name__ == '__main__':
    unittest.main()
