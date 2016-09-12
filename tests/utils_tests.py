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

class TestOthers(unittest.TestCase):

    def test_seedname(self):
        self.assertEqual(seedname('a/b/seed.txt'), 'seed')

    def test_swing_twist(self):
        
        from ase.quaternions import Quaternion

        test_n = 10

        for t_i in range(test_n):

            # Create two quaternions with random rotations
            theta1, theta2 = np.random.random(2)*2*np.pi
            ax1 = np.random.random(3)
            ax2 = np.cross(np.random.random(3), ax1)
            ax1 /= np.linalg.norm(ax1)
            ax2 /= np.linalg.norm(ax2)

            q1 = Quaternion([np.cos(theta1/2)] + list(ax1*np.sin(theta1/2)))
            q2 = Quaternion([np.cos(theta2/2)] + list(ax2*np.sin(theta2/2)))

            qT = q1*q2

            # Now decompose
            qsw, qtw = swing_twist_decomp(qT, ax2)
            # And check
            q1.q *= np.sign(q1.q[0])
            q2.q *= np.sign(q2.q[0])
            qsw.q *= np.sign(qsw.q[0])
            qtw.q *= np.sign(qtw.q[0])

            self.assertTrue(np.allclose(q1.q, qsw.q))
            self.assertTrue(np.allclose(q2.q, qtw.q))


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
            scell_shape = minimum_supcell(max_r, latt_cart=cart)
            grid_i, grid = supcell_gridgen(cart, scell_shape)
            grid -= centre  # We refer to the centre
            grid_norm = np.linalg.norm(grid, axis=1)
            # Now let's check the sphere points
            sphere_p = grid_i[np.where(grid_norm < max_r)]
            sphere_bf_p = grid_bf_i[np.where(grid_bf_norm < max_r)]
            # Now, are they equal?
            sphere_p = set([tuple(p) for p in sphere_p])
            sphere_bf_p = set([tuple(p) for p in sphere_bf_p])
            self.assertEqual(sphere_p, sphere_bf_p)

    def test_min_periodic(self):

        # Just a simple check

        c = np.identity(3)
        v = [[0.1, 0, 0], [0.9, 0, 0]]

        vp, vcells = minimum_periodic(v, c)
        self.assertTrue(np.isclose(np.linalg.norm(vp, axis=-1), 0.1).all())

if __name__ == '__main__':
    unittest.main()
