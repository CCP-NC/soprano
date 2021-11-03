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
import unittest
import numpy as np
from ase import Atoms
from ase.build import bulk

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa


class TestSymmetry(unittest.TestCase):
    def test_symdataset(self):

        from soprano.properties.symmetry import SymmetryDataset

        # Create atoms objects according to a given symmetry group, check
        # that it works
        pos = np.zeros((6, 3))
        pos[0] = [0, 0.1, 0.2]
        pos[1] = [0, 0.3, 0.8]
        pos[2] = [0.4, 0.2, 0.6]
        pos[3:] = -pos[:3]

        symmA = Atoms(["C"] * 6, positions=pos, cell=[5] * 3, pbc=[True] * 3)

        symdata = SymmetryDataset.get(symmA)

        self.assertTrue(symdata["international"] == "P-1")

    def test_wyckoff(self):

        from soprano.properties.symmetry import WyckoffPoints
        from soprano.utils import minimum_supcell, supcell_gridgen

        si2 = bulk("Si")
        wpoints = WyckoffPoints.get(si2)

        # Try a generic function with the same symmetry as the lattice
        class GaussField(object):
            def __init__(self, a, sigma=2.0, rcut=10):
                shape = minimum_supcell(rcut, a.get_cell())
                igrid, grid = supcell_gridgen(a.get_cell(), shape)
                self.p = a.get_positions()[None, :, :] + grid[:, None, :]
                self.p = self.p.reshape((-1, 3))
                sph_i = np.where(np.linalg.norm(self.p, axis=1) <= rcut)
                self.p = self.p[sph_i]
                self.s = sigma

            def field(self, r):
                dz = (self.p - r) / self.s
                return np.sum(np.exp(-0.5 * np.sum(dz ** 2, axis=1)))

        gtest = GaussField(si2)

        rng = np.random.default_rng(0)

        for wp in wpoints:
            fpos = wp.fpos
            pos = wp.pos
            # Check that the points are really invariant
            for o in wp.operations:
                self.assertTrue(np.isclose((np.dot(o[0], fpos) + o[1]) % 1, fpos).all())
            if wp.hessian == "isotropic":
                dr1 = rng.normal(size=3)
                dr1 /= np.linalg.norm(dr1)
                dr2 = rng.normal(size=3)
                dr2 /= np.linalg.norm(dr2)
                g1 = gtest.field(pos + dr1 * 1e-3)
                g2 = gtest.field(pos + dr2 * 1e-3)

                self.assertTrue(np.isclose(g1, g2, 1e-3))


if __name__ == "__main__":
    unittest.main()
