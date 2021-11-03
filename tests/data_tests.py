#!/usr/bin/env python
"""
Test code for data retrieval
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from ase.data import vdw_radii as vdw_radii_ase
from soprano.data import vdw_radius, nmr_gamma, nmr_spin, nmr_quadrupole


class TestData(unittest.TestCase):
    def test_vdw(self):

        self.assertEqual(vdw_radius("C", "csd"), 1.77)
        self.assertEqual(vdw_radius("C", "jmol"), 1.95)
        self.assertEqual(vdw_radius("C", "ase"), vdw_radii_ase[6])

    def test_gamma(self):

        self.assertEqual(nmr_gamma("H"), 267522128.0)
        self.assertEqual(nmr_gamma("H", 2), 41066279.1)

    def test_spin(self):

        self.assertEqual(nmr_spin("H"), 0.5)
        self.assertEqual(nmr_spin("H", 2), 1)

    def test_quadrupole(self):

        self.assertEqual(nmr_quadrupole("H"), 0)
        self.assertEqual(nmr_quadrupole("H", 2), 2.86)


if __name__ == "__main__":
    unittest.main()
