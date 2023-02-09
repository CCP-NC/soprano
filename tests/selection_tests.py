#!/usr/bin/env python
"""
Test code for AtomSelection
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import unittest
import numpy as np

from ase import Atoms
from ase.io import read


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestSelection(unittest.TestCase):
    def test_basic(self):

        from soprano.selection import AtomSelection

        # Create an Atoms object
        a = Atoms("HHH")

        # Try a valid selection
        s1 = AtomSelection(a, [0, 2])

        # Try an invalid one
        self.assertRaises(ValueError, AtomSelection, a, [0, 3])

        # Check validation
        self.assertTrue(s1.validate(a))

        # Now make a subset
        a_s = s1.subset(a)
        self.assertTrue(len(a_s) == 2)

    def test_operators(self):

        from soprano.selection import AtomSelection

        # Create an Atoms object
        a1 = Atoms("HHH")
        a2 = Atoms("CC")

        s1 = AtomSelection(a1, [0, 2])
        s2 = AtomSelection(a1, [0, 1])

        self.assertTrue(set((s1 + s2).indices) == set([0, 1, 2]))
        self.assertTrue(set((s1 - s2).indices) == set([2]))
        self.assertTrue(set((s1 * s2).indices) == set([0]))

        s3 = AtomSelection(a2, [0])

        # Different systems
        with self.assertRaises(ValueError):
            s1 + s3

    def test_selectors(self):

        from soprano.selection import AtomSelection

        # Multiple tests for various methods
        a = Atoms(
            "HCHC", positions=[[i] * 3 for i in range(4)], cell=[4] * 3, pbc=[True] * 3
        )

        # Element test
        s1 = AtomSelection.from_element(a, "C")

        self.assertTrue(set(s1.indices) == set([1, 3]))

        # Box test
        s1 = AtomSelection.from_box(a, [1.5] * 3, [4.5] * 3, periodic=True)
        s2 = AtomSelection.from_box(a, [1.5] * 3, [4.5] * 3, periodic=False)
        s3 = AtomSelection.from_box(
            a, [0.375] * 3, [1.125] * 3, periodic=True, scaled=True
        )

        self.assertTrue(set(s1.indices) == set([0, 2, 3]))
        self.assertTrue(set(s2.indices) == set([2, 3]))
        self.assertTrue(set(s3.indices) == set([0, 2, 3]))

        # Sphere test

        s1 = AtomSelection.from_sphere(a, [0.5] * 3, 3, periodic=True)
        s2 = AtomSelection.from_sphere(a, [0.5] * 3, 3, periodic=False)

        self.assertTrue(set(s1.indices) == set([0, 1, 2, 3]))
        self.assertTrue(set(s2.indices) == set([0, 1, 2]))

        # String test
        # add cif-like labels:
        a.set_array('labels', np.array(['H1a', 'C1', 'H1b', 'C1']))
        s1 = AtomSelection.from_selection_string(a, "H")
        s2 = AtomSelection.from_selection_string(a, "C")
        s3 = AtomSelection.from_selection_string(a, "C.1")
        s4 = AtomSelection.from_selection_string(a, "C.1-2")
        s5 = AtomSelection.from_selection_string(a, "C.1-2,H.2")
        s6 = AtomSelection.from_selection_string(a, "C.1,C.2")
        s7 = AtomSelection.from_selection_string(a, "C1,H1a")
        s8 = AtomSelection.from_selection_string(a, "C,H")

        self.assertTrue(set(s1.indices) == set([0, 2]))
        self.assertTrue(set(s2.indices) == set([1, 3]))
        self.assertTrue(set(s3.indices) == set([1]))
        self.assertTrue(set(s4.indices) == set([1, 3]))
        self.assertTrue(set(s5.indices) == set([1, 2, 3]))
        self.assertTrue(set(s6.indices) == set([1, 3]))
        self.assertTrue(set(s7.indices) == set([0, 1, 3]))
        self.assertTrue(set(s8.indices) == set([0, 1, 2, 3]))

        # Test invalid string
        self.assertRaises(ValueError, AtomSelection.from_selection_string, a, "C1-3")
        self.assertRaises(ValueError, AtomSelection.from_selection_string, a, "C1.3")

        # Unique atoms test
        a = read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        Z = 4 # for this molecular crystal
        s1 = AtomSelection.unique(a)
        print(len(s1.indices))
        self.assertEqual(len(s1)*Z, len(a))

    def test_arrays(self):

        from soprano.selection import AtomSelection

        a = Atoms(
            "HCHC", positions=[[i] * 3 for i in range(4)], cell=[4] * 3, pbc=[True] * 3
        )

        s = AtomSelection.from_element(a, "C")
        s.set_array("testarr", [1, 2])

        self.assertTrue(all(s.subset(a).get_array("testarr") == [1, 2]))

        # Test that arrays are reordered
        a.set_array("testarr", np.array([1, 2, 3, 4]))

        s = AtomSelection(a, [2, 0])
        a2 = s.subset(a)

        self.assertTrue((a2.get_array("testarr") == np.array([3, 1])).all())

        # Cell indices test!
        s = AtomSelection(a, [0, 3])
        s.set_array("cell_indices", [[0, 0, 0], [-1, 0, 0]])
        a2 = s.subset(a, True)

        self.assertTrue(np.allclose(a2.get_positions()[-1], [-1, 3, 3]))

    def test_mapsel(self):

        from soprano.selection import AtomSelection
        from soprano.collection import AtomsCollection

        el_list = 'HHHCCHCHCH'
        coll = AtomsCollection([Atoms(el) for el in el_list])

        h_sel = coll.all.map(AtomSelection.from_element, element="H")

        self.assertTrue(
            all(
                [
                    len(s) == 1 if el_list[i] == "H" else len(s) == 0
                    for i, s in enumerate(h_sel)
                ]
            )
        )

    def test_iterate(self):

        from soprano.selection import AtomSelection

        a = Atoms(
            "HCHC", positions=[[i] * 3 for i in range(4)], cell=[4] * 3, pbc=[True] * 3
        )

        sAll = AtomSelection.all(a)

        # Slicing?
        self.assertTrue((sAll[:2].indices == [0, 1]).all())

        # Iterating?
        for i, s in enumerate(sAll):
            self.assertEqual(i, s.indices[0])


if __name__ == "__main__":
    unittest.main()
