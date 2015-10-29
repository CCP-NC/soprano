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
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.utils import *
import unittest
import numpy as np


class TestLatticeMethods(unittest.TestCase):

    def test_abc2cart(self):
        abc = np.array([[1, 1, 1], [np.pi/2, np.pi/2, np.pi/2]])
        cart = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(abc2cart(abc), cart))

    def test_cart2abc(self):
        abc = np.array([[1, 1, 1], [np.pi/2, np.pi/2, np.pi/2]])
        cart = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(cart2abc(cart), abc))


class TestSupercellMethods(unittest.TestCase):

    def test_min_supcell(self):
        # Here the test consists of trying to do things our way,
        # then the brutal way, and see if they give the same answers

        n_attempts = 10
        cell_scale = 1.0
        cell_min = 0.2
        cell_skew = 0.2
        r_brutef_bounds = (10, 10, 10)

        for attempt in range(n_attempts):

            # Define a lattice (cartesian)
            cart = np.diag(np.random.random(3)*cell_scale+cell_min)
            cart += np.random.random((3, 3))*cell_skew
            abc = cart2abc(cart)
            centre = np.dot(cart, np.ones(3)*0.5)
            # Now find a "brute force" grid
            grid_bf_i, grid_bf = supcell_gridgen(cart, r_brutef_bounds)
            grid_bf -= centre
            grid_bf_norm = np.linalg.norm(grid_bf, axis=1)
            # Pick a suitable radius
            max_r = np.random.random()*cell_scale+cell_min
            # Find the supercell
            r_bounds = minimum_supcell(max_r, latt_cart=cart)
            grid_i, grid = supcell_gridgen(cart, r_bounds)
            grid -= centre  # We refer to the centre
            grid_norm = np.linalg.norm(grid, axis=1)
            # Now let's check the sphere points
            sphere_p = grid_i[np.where(grid_norm < max_r)]
            sphere_bf_p = grid_bf_i[np.where(grid_bf_norm < max_r)]
            # Now, are they equal?
            sphere_p = set([tuple(p) for p in sphere_p])
            sphere_bf_p = set([tuple(p) for p in sphere_bf_p])
            self.assertEqual(sphere_p, sphere_bf_p)

if __name__ == '__main__':
    unittest.main()
