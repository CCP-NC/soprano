#!/usr/bin/env python
"""
Test code for various types of Genes
"""


import os
import sys
import unittest

import ase
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestGenes(unittest.TestCase):

    @pytest.mark.filterwarnings("ignore:crystal system 'orthorhombic'")
    def test_coordhist(self):
        from soprano.analyse.phylogen import Gene
        from soprano.collection import AtomsCollection

        ala = ase.io.read(os.path.join(_TESTDATA_DIR, "mol_crystal.cif"))
        nh3 = ase.io.read(os.path.join(_TESTDATA_DIR, "nh3.cif"))

        c = AtomsCollection([ala, nh3])

        g = Gene("coord_histogram")
        h = g.evaluate(c)

        self.assertTrue((h[0] == [0, 4, 0, 4, 0, 0, 0]).all())
        self.assertTrue((h[1] == 0).all())  # No C at all in this one

        # Now for something different...
        g = Gene("coord_histogram", params={"s1": "N"})
        h = g.evaluate(c)

        self.assertTrue((h[0] == [0, 0, 0, 4, 0, 0, 0]).all())  # 4 NH3 groups
        self.assertTrue((h[1] == [0, 0, 0, 1, 0, 0, 0]).all())


if __name__ == "__main__":
    unittest.main()
