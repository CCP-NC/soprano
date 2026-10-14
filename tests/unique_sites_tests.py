#!/usr/bin/env python
"""
Tests for UniqueSites handling of CIF labels versus symmetry reduction.
"""


import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)


def _nacl():
    from ase.build import bulk

    return bulk("NaCl", "rocksalt", a=5.64, cubic=True)


def _match_warnings(w):
    return [x for x in w if "don't match the CIF labels" in str(x.message)]


class TestUniqueSites(unittest.TestCase):
    def setUp(self):
        # Symmetry grouping of the conventional NaCl cell: Na and Cl sublattices
        self.sym_tags = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    def test_no_labels_uses_symmetry(self):
        from soprano.properties.labeling import UniqueSites

        tags = UniqueSites.get(_nacl(), save_info=False)
        self.assertTrue(np.array_equal(tags, self.sym_tags))

    def test_same_grouping_different_label_order(self):
        from soprano.properties.labeling import UniqueSites

        atoms = _nacl()
        # Same partition as symmetry, but the label ranks are ordered differently
        atoms.new_array(
            "labels",
            np.array(["Z1", "A1", "Z1", "A1", "Z1", "A1", "Z1", "A1"]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tags = UniqueSites.get(atoms, save_info=False)
        self.assertEqual(_match_warnings(w), [])
        self.assertTrue(np.array_equal(tags, self.sym_tags))

    def test_mismatched_labels_warn(self):
        from soprano.properties.labeling import UniqueSites

        atoms = _nacl()
        # Splits the symmetry-equivalent Na sites into two groups
        atoms.new_array(
            "labels",
            np.array(["Na1", "Cl1", "Na2", "Cl1", "Na1", "Cl1", "Na2", "Cl1"]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            UniqueSites.get(atoms, save_info=False)
        self.assertGreaterEqual(len(_match_warnings(w)), 1)

    def test_mismatch_default_returns_cif_grouping(self):
        from soprano.properties.labeling import UniqueSites

        atoms = _nacl()
        atoms.new_array(
            "labels",
            np.array(["Na1", "Cl1", "Na2", "Cl1", "Na1", "Cl1", "Na2", "Cl1"]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tags = UniqueSites.get(atoms, save_info=False, override_cif=False)
        self.assertTrue(np.array_equal(tags, np.array([0, 1, 2, 1, 0, 1, 2, 1])))

    def test_mismatch_override_cif_returns_symmetry_grouping(self):
        from soprano.properties.labeling import UniqueSites

        atoms = _nacl()
        atoms.new_array(
            "labels",
            np.array(["Na1", "Cl1", "Na2", "Cl1", "Na1", "Cl1", "Na2", "Cl1"]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tags = UniqueSites.get(atoms, save_info=False, override_cif=True)
        self.assertTrue(np.array_equal(tags, self.sym_tags))


if __name__ == "__main__":
    unittest.main()
