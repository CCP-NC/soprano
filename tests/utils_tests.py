#!/usr/bin/env python
"""
Test code for the utils.py functions
"""


import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)


class TestOthers(unittest.TestCase):
    def test_seedname(self):
        from soprano.utils import seedname

        self.assertEqual(seedname("a/b/seed.txt"), "seed")

    def test_swing_twist(self):
        from ase.quaternions import Quaternion

        from soprano.utils import swing_twist_decomp

        test_n = 10

        rng = np.random.default_rng(0)

        for t_i in range(test_n):
            # Create two quaternions with random rotations
            theta1, theta2 = rng.random(2) * 2 * np.pi
            ax1 = rng.random(3)
            ax2 = np.cross(rng.random(3), ax1)
            ax1 /= np.linalg.norm(ax1)
            ax2 /= np.linalg.norm(ax2)

            q1 = Quaternion([np.cos(theta1 / 2)] + list(ax1 * np.sin(theta1 / 2)))
            q2 = Quaternion([np.cos(theta2 / 2)] + list(ax2 * np.sin(theta2 / 2)))

            qT = q1 * q2

            # Now decompose
            qsw, qtw = swing_twist_decomp(qT, ax2)
            # And check
            q1.q *= np.sign(q1.q[0])
            q2.q *= np.sign(q2.q[0])
            qsw.q *= np.sign(qsw.q[0])
            qtw.q *= np.sign(qtw.q[0])

            self.assertTrue(np.allclose(q1.q, qsw.q))
            self.assertTrue(np.allclose(q2.q, qtw.q))

    def test_specsort(self):
        from soprano.utils import graph_specsort

        # Define the Laplacian of a graph
        # This one is chosen so the ordering is 100% unambiguous

        N = 6
        D = np.zeros((N, N))
        A = D.copy()

        # Add edges
        edges = np.array(
            [(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (2, 5), (3, 5)]
        )
        for e in edges:
            D[e[0], e[0]] += 1
            D[e[1], e[1]] += 1
            A[e[0], e[1]] = 1
            A[e[1], e[0]] = 1

        L = D - A
        ssort = graph_specsort(L)

        self.assertTrue((ssort == [0, 1, 4, 2, 3, 5]).all())



    def test_average_quaternions(self):
        """Test the average_quaternions function and numpy compatibility"""
        from ase.quaternions import Quaternion
        from soprano.utils import average_quaternions
        
        # Test with identity quaternions
        q_identity = Quaternion([1, 0, 0, 0])
        result = average_quaternions([q_identity, q_identity, q_identity])
        
        # Result should be close to identity quaternion (normalized)
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_allclose(result.q, expected, atol=1e-10)
        
        # Test averaging a single quaternion
        q = Quaternion([0.7071, 0.7071, 0, 0])  # 90 degree rotation around x-axis
        result = average_quaternions([q])
        
        # Should return the same quaternion (normalized)
        np.testing.assert_allclose(result.q, q.q / np.linalg.norm(q.q), atol=1e-10)
        
        # Test numpy compatibility
        # This test specifically ensures the numpy compatibility works
        q1 = Quaternion([1, 0, 0, 0])
        q2 = Quaternion([0.7071, 0.7071, 0, 0])
        q3 = Quaternion([0.7071, 0, 0.7071, 0])
        
        # This should not raise an AttributeError about .A1
        try:
            result = average_quaternions([q1, q2, q3])
            # Check that result is a valid quaternion (normalized)
            self.assertAlmostEqual(np.linalg.norm(result.q), 1.0, places=10)
            # Check that result is a Quaternion object
            self.assertIsInstance(result, Quaternion)
        except AttributeError as e:
            if ".A1" in str(e):
                self.fail("AttributeError related to .A1 - numpy compatibility issue not fixed")
            else:
                raise
        
        # Test with random quaternions for robustness
        rng = np.random.default_rng(42)  # Use fixed seed for reproducibility
        
        # Generate random quaternions
        quaternions = []
        for _ in range(5):
            # Generate random unit quaternion
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)
            quaternions.append(Quaternion(q))
        
        result = average_quaternions(quaternions)
        
        # Check that result is normalized
        self.assertAlmostEqual(np.linalg.norm(result.q), 1.0, places=10)
        
        # Check that result is a Quaternion object
        self.assertIsInstance(result, Quaternion)
        
        # Test that empty list raises assertion error
        with self.assertRaises(AssertionError):
            average_quaternions([])


class TestLatticeMethods(unittest.TestCase):
    def test_abc2cart(self):
        from soprano.utils import abc2cart

        abc = np.array([[1, 1, 1], [np.pi / 2, np.pi / 2, np.pi / 2]])
        cart = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(abc2cart(abc), cart))

    def test_cart2abc(self):
        from soprano.utils import cart2abc

        abc = np.array([[1, 1, 1], [np.pi / 2, np.pi / 2, np.pi / 2]])
        cart = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(cart2abc(cart), abc))


class TestSupercellMethods(unittest.TestCase):
    def test_min_supcell(self):
        from soprano.utils import minimum_supcell, supcell_gridgen

        # Here the test consists of trying to do things our way,
        # then the brutal way, and see if they give the same answers

        cell_scale = 1.0
        r_brutef_bounds = (11, 11, 11)

        # Lattices to test
        lattices = [
            (np.eye(3) * cell_scale, 2.0),
            (np.diag([1, 2, 1.5]) * cell_scale, 1.5),
            (np.array([[1, 0, 0], [0, 0.7, 0], [0.7, 0, 0.5]]) * cell_scale, 2.0),
        ]

        for cart, max_r in lattices:
            # Define a lattice (cartesian)
            centre = np.dot(np.ones(3) * 0.5, cart)
            # Now find a "brute force" grid
            grid_bf_i, grid_bf = supcell_gridgen(cart, r_brutef_bounds)
            grid_bf -= centre
            grid_bf_norm = np.linalg.norm(grid_bf, axis=1)

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
        from soprano.utils import minimum_periodic

        # Just a simple check

        c = np.identity(3)
        v = [[0.1, 0, 0], [0.9, 0, 0]]

        vp, vcells = minimum_periodic(v, c)
        self.assertTrue(np.isclose(np.linalg.norm(vp, axis=-1), 0.1).all())

    def test_min_periodic_exclude_self_zero_vectors(self):
        """Test that exclude_self works when all input vectors are zero.

        This covers the max_r == 0 branch: the search radius must be
        determined from the shortest non-zero lattice vector rather than
        from the (zero) input vectors.
        """
        from soprano.utils import minimum_periodic

        # Cubic cell – shortest lattice vector is 1.0 along any axis.
        c = np.identity(3)
        v_zero = [[0.0, 0.0, 0.0]]
        vp, vcells = minimum_periodic(v_zero, c, exclude_self=True)
        # Result must be non-zero and have length == 1 (nearest image).
        result_norm = np.linalg.norm(vp[0])
        self.assertGreater(result_norm, 0.0)
        self.assertTrue(np.isclose(result_norm, 1.0))

        # Orthorhombic cell with unequal axes – shortest vector is along b (2 Å).
        c_ortho = np.diag([5.0, 2.0, 4.0])
        vp_ortho, _ = minimum_periodic(v_zero, c_ortho, exclude_self=True)
        result_norm_ortho = np.linalg.norm(vp_ortho[0])
        self.assertGreater(result_norm_ortho, 0.0)
        self.assertTrue(np.isclose(result_norm_ortho, 2.0))

        # Multiple zero vectors – all should return a nearest image.
        v_multi = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        vp_multi, _ = minimum_periodic(v_multi, c, exclude_self=True)
        norms = np.linalg.norm(vp_multi, axis=-1)
        self.assertTrue(np.all(norms > 0))
        self.assertTrue(np.allclose(norms, 1.0))


# test the merging of sites
class TestMergeSites(unittest.TestCase):
    def test_merge_sites(self):
        from ase.build import molecule

        from soprano.utils import merge_sites

        # define a structure (let's use ethanol)
        atoms = molecule("CH3CH2OH", vacuum=10, pbc=True)  # ethanol
        # set some custom arrays
        ms = np.random.rand(len(atoms), 3, 3)
        efg = np.random.rand(len(atoms), 3)
        labels = np.array(["C1", "C2", "O1", "H1", "H2", "H3", "H4", "H5", "H6"])
        atoms.set_array("ms", ms)
        atoms.set_array("efg", efg)
        atoms.set_array("labels", labels)

        # define the indices of the sites to merge
        # use the CH3 group as an example
        merge_indices = [6, 7, 8]  # the H atoms from the CH3 group
        # merge the sites
        atoms = merge_sites(atoms, merge_indices)
        # check that the number of atoms is correct
        self.assertEqual(len(atoms), 7)
        # check that the CH3 group is now a single site
        self.assertEqual(atoms.get_array("labels")[6], "H4,H5,H6")
        # check that the ms and efg arrays are correct
        self.assertTrue(
            np.allclose(atoms.get_array("ms")[6], np.mean(ms[merge_indices], axis=0))
        )
        self.assertTrue(
            np.allclose(atoms.get_array("efg")[6], np.mean(efg[merge_indices], axis=0))
        )

        # now let's try to merge the C1 and C2 sites
        merge_indices = [0, 1]
        atoms = merge_sites(atoms, merge_indices)
        # check that the number of atoms is correct
        self.assertEqual(len(atoms), 6)
        # check that the C1 and C2 sites are now a single site
        self.assertEqual(atoms.get_array("labels")[0], "C1,C2")
        # check that the ms and efg arrays are correct
        self.assertTrue(
            np.allclose(atoms.get_array("ms")[0], np.mean(ms[merge_indices], axis=0))
        )
        self.assertTrue(
            np.allclose(atoms.get_array("efg")[0], np.mean(efg[merge_indices], axis=0))
        )

        # we can also change the behaviour of the merge_sites function
        # by passing a custom function to the merge_func argument
        # let's merge the CH2 groups by taking first of each quantity
        atoms = merge_sites(
            atoms,
            [2, 3],
            merging_strategies={
                "ms": lambda x: x[0],
                "efg": lambda x: x[0],
                "labels": lambda x: x[0],
            },
        )
        # check that the number of atoms is correct
        self.assertEqual(len(atoms), 5)
        # check that the CH2 groups are now a single site
        # here the group got the label of the first atom in the group
        self.assertEqual(atoms.get_array("labels")[2], "H1")
        # check that the ms and efg arrays are correct
        # since C1 and C2 are merges in atoms, but not in ms and efg
        # we need to shift the indices by 1
        self.assertTrue(np.allclose(atoms.get_array("ms")[2], ms[3]))
        self.assertTrue(np.allclose(atoms.get_array("efg")[2], efg[3]))


if __name__ == "__main__":
    unittest.main()
