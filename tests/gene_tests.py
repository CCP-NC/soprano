#!/usr/bin/env python
"""
Test code for various types of Genes
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import ase
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestGenes(unittest.TestCase):
    def test_coordhist(self):

        from soprano.collection import AtomsCollection
        from soprano.analyse.phylogen import Gene

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
