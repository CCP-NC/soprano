#!/usr/bin/env python
"""
Test code for NMR properties
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
import numpy as np
from ase import io
from ase.quaternions import Quaternion
from soprano.nmr.tensor import NMRTensor
from soprano.nmr.utils import _test_euler_rotation
from soprano.properties.nmr import (
    MSIsotropy,
    MSAnisotropy,
    MSReducedAnisotropy,
    MSAsymmetry,
    MSSpan,
    MSSkew,
    EFGVzz,
    EFGAsymmetry,
    EFGQuadrupolarConstant,
    EFGQuadrupolarProduct,
    EFGQuaternion,
    EFGNQR,
    EFGTensor,
    DipolarCoupling,
    DipolarDiagonal,
    DipolarTensor,
)

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestNMR(unittest.TestCase):
    def test_shielding(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

        # Load the data calculated with MagresView
        with open(os.path.join(_TESTDATA_DIR, "ethanol_ms.dat")) as f:
            data = f.readlines()[8:]

        iso = MSIsotropy.get(eth)
        aniso = MSAnisotropy.get(eth)
        r_aniso = MSReducedAnisotropy.get(eth)
        asymm = MSAsymmetry.get(eth)
        span = MSSpan.get(eth)
        skew = MSSkew.get(eth)

        for i, d in enumerate(data):
            vals = [float(x) for x in d.split()[1:]]
            if len(vals) != 10:
                continue
            # And check...
            self.assertAlmostEqual(iso[i], vals[0])
            self.assertAlmostEqual(aniso[i], vals[1])
            self.assertAlmostEqual(r_aniso[i], vals[2])
            self.assertAlmostEqual(asymm[i], vals[3])
            self.assertAlmostEqual(span[i], max(vals[4:7]) - min(vals[4:7]))
            self.assertAlmostEqual(
                skew[i], 3 * (iso[i] - sorted(vals[4:7])[1]) / span[i]
            )

    def test_efg(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

        # Load the data calculated with MagresView
        with open(os.path.join(_TESTDATA_DIR, "ethanol_efg.dat")) as f:
            data = f.readlines()[8:]
        # load in euler angle references - cross-referenced with
        # TensorView for Matlab
        euler_refs = np.loadtxt(os.path.join(_TESTDATA_DIR, "ethanol_efg_eulers.dat"), skiprows=1)

        asymm = EFGAsymmetry.get(eth)

        qprop = EFGQuadrupolarConstant(isotopes={"H": 2})
        qcnst = qprop(eth)
        efg_tensors = EFGTensor.get(eth)

        euler_angles = []
        for efg in efg_tensors:
            euler_angles.append(efg.euler_angles() * 180 / np.pi)
        
        # compare to reference euler angles:
        self.assertTrue(np.allclose(euler_angles, euler_refs))

        for i, d in enumerate(data):
            vals = [float(x) for x in d.split()[1:]]
            if len(vals) != 8:
                continue
            # And check...
            # The quadrupolar constant has some imprecision due to values
            # of quadrupole moment, so we only ask 2 places in kHz
            self.assertAlmostEqual(qcnst[i] * 1e-3, vals[0] * 1e-3, places=2)
            self.assertAlmostEqual(asymm[i], vals[1])

        # A more basic test for the Vzz
        Vzz_p = EFGVzz.get(eth)
        Vzz_raw = []
        for efg in eth.get_array("efg"):
            evals, _ = np.linalg.eigh(efg)
            Vzz_raw.append(evals[np.argmax(abs(evals))])

        self.assertTrue(np.isclose(Vzz_p, Vzz_raw).all())

        # A basic test for NQR frequencies
        NQR = EFGNQR.get(eth)
        non_zero_NQRs = np.where(NQR)[0]
        # Only the O has NQR & there's only one O atom
        self.assertTrue(len(non_zero_NQRs) == 1)
        self.assertTrue(eth[non_zero_NQRs[0]].symbol == "O")
        NQR_vals = [v for v in NQR[-1].values()]
        NQR_keys = [k for k in NQR[-1].keys()]
        # the first is 0.5 -> 1.5
        self.assertTrue(NQR_keys[0] == "m=0.5->1.5")
        # the first is 1.5 -> 2.5
        self.assertTrue(NQR_keys[1] == "m=1.5->2.5")
        # the ratio bewtween the two transion frequencies should 2 in this case
        self.assertAlmostEqual(NQR_vals[1] / NQR_vals[0], 2.0)

    def test_dipolar(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

        # Load the data calculated with MagresView
        with open(os.path.join(_TESTDATA_DIR, "ethanol_dip.dat")) as f:
            data = f.readlines()[8:]

        dip = DipolarCoupling.get(eth)
        diptens = DipolarTensor.get(eth)
        dipdiag = DipolarDiagonal.get(eth)

        # Magres labels
        symbs = np.array(eth.get_chemical_symbols())
        elems = set(symbs)
        mlabs = [""] * len(eth)
        for e in elems:
            e_i = np.where(symbs == e)[0]
            for i, j in enumerate(e_i):
                mlabs[j] = "{0}_{1}".format(e, i + 1)

        data_dip = {}
        for l in data:
            lab1, lab2, d, a, b = l.split()
            i1 = mlabs.index(lab1)
            i2 = mlabs.index(lab2)
            i1, i2 = (i1, i2) if i1 < i2 else (i2, i1)
            data_dip[(i1, i2)] = [float(x) for x in (d, a, b)]

        for ij, (d, v) in dip.items():
            # The precision is rather low, probably due to the gammas
            self.assertAlmostEqual(d * 2 * np.pi, data_dip[ij][0], places=-3)
            a, b = np.array([np.arccos(-v[2]), np.arctan2(-v[1], -v[0])]) % (2 * np.pi)
            ba_dat = (np.array(data_dip[ij][1:]) * np.pi / 180.0) % (2 * np.pi)
            self.assertTrue(np.isclose([b, a], ba_dat, atol=0.1).all())

            evals = dipdiag[ij]["evals"]
            evecs = dipdiag[ij]["evecs"]
            self.assertAlmostEqual(evals[2], 2 * d)
            self.assertAlmostEqual(np.dot(evecs[:, 2], v), 1)
            self.assertTrue(np.isclose(np.dot(evecs.T, evecs), np.eye(3)).all())

            # Further test the full tensors
            evalstt = np.linalg.eigh(diptens[ij].data)[0]
            self.assertTrue(np.isclose(np.sort(evals), np.sort(evalstt)).all())
            self.assertAlmostEqual(np.linalg.multi_dot([v, diptens[ij].data, v]), 2 * d)

    def test_tensor(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        ms = eth.get_array("ms")

        iso = MSIsotropy.get(eth)
        aniso = MSAnisotropy.get(eth)
        r_aniso = MSReducedAnisotropy.get(eth)
        asymm = MSAsymmetry.get(eth)
        span = MSSpan.get(eth)
        skew = MSSkew.get(eth)

        # Get the eigenvalues
        diag = [np.linalg.eigh((m + m.T) / 2.0) for m in ms]

        for i in range(len(eth)):
            ms_tens = NMRTensor(ms[i])
            evals, evecs = diag[i]

            self.assertAlmostEqual(iso[i], ms_tens.isotropy)
            self.assertAlmostEqual(aniso[i], ms_tens.anisotropy)
            self.assertAlmostEqual(r_aniso[i], ms_tens.reduced_anisotropy)
            self.assertAlmostEqual(asymm[i], ms_tens.asymmetry)
            self.assertAlmostEqual(span[i], ms_tens.span)
            self.assertAlmostEqual(skew[i], ms_tens.skew)

            self.assertTrue(np.isclose(evals, ms_tens.eigenvalues).all())
            self.assertAlmostEqual(
                np.dot(
                    np.cross(ms_tens.eigenvectors[:, 0], ms_tens.eigenvectors[:, 1]),
                    ms_tens.eigenvectors[:, 2],
                ),
                1,
            )
            # Spherical representation tests
            # Calculate the spherical representation
            sph_repr = ms_tens.spherical_repr

            # Isotropic part
            isotropic = np.eye(3) * np.trace(ms_tens._data) / 3

            # Anti-symmetric part
            antisymmetric = (ms_tens._data - ms_tens._data.T) / 2.0

            # Symmetric part - isotropic part
            symmetric = (ms_tens._data + ms_tens._data.T) / 2.0 - isotropic

            # Check isotropic part
            np.testing.assert_array_almost_equal(sph_repr[0], isotropic)

            # Check antisymmetric part
            np.testing.assert_array_almost_equal(sph_repr[1], antisymmetric)
            # should be traceless
            self.assertAlmostEqual(np.trace(sph_repr[1]), 0)

            # Check symmetric part
            np.testing.assert_array_almost_equal(sph_repr[2], symmetric)
            # should be traceless
            self.assertAlmostEqual(np.trace(sph_repr[2]), 0)

            # Check that the sum of the parts equals the original tensor
            np.testing.assert_array_almost_equal(sph_repr[0] + sph_repr[1] + sph_repr[2], ms_tens._data)

    def test_tensor_conventions(self):
        # Let's now try various conventions
        data = np.diag([1, 2, -6])

        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        # check eigenvalues are sorted correctly
        self.assertTrue(np.allclose(tc.eigenvalues, [-6, 1, 2]))
        # and eigenvectors are sorted accordingly
        self.assertTrue(np.allclose(tc.eigenvectors[0], [0, 1, 0]))
        self.assertTrue(np.allclose(tc.eigenvectors[1], [0, 0, 1]))
        self.assertTrue(np.allclose(tc.eigenvectors[2], [1, 0, 0]))

        td = NMRTensor(data, NMRTensor.ORDER_DECREASING)
        self.assertTrue(np.allclose(td.eigenvalues, [2, 1, -6]))
        self.assertTrue(np.allclose(td.eigenvectors[0], [0, 1, 0]))
        self.assertTrue(np.allclose(td.eigenvectors[1], [1, 0, 0]))
        self.assertTrue(np.allclose(td.eigenvectors[2], [0, 0,-1]))

        th = NMRTensor(data, NMRTensor.ORDER_HAEBERLEN)
        self.assertTrue(np.allclose(th.eigenvalues, [2, 1, -6]))
        self.assertTrue(np.allclose(th.eigenvectors[0], [0, 1, 0]))
        self.assertTrue(np.allclose(th.eigenvectors[1], [1, 0, 0]))
        self.assertTrue(np.allclose(th.eigenvectors[2], [0, 0,-1]))


        tn = NMRTensor(data, NMRTensor.ORDER_NQR)
        self.assertTrue(np.allclose(tn.eigenvalues, [1, 2, -6]))
        self.assertTrue(np.allclose(tn.eigenvectors[0], [1, 0, 0]))
        self.assertTrue(np.allclose(tn.eigenvectors[1], [0, 1, 0]))
        self.assertTrue(np.allclose(tn.eigenvectors[2], [0, 0, 1]))



        # The span and skew should always be the same since they are defined only for the 
        # Herzfeld-Berger convention
        # in this case the span is 8 = 2 - (-6)
        self.assertTrue(np.allclose([tc.span, td.span, th.span, tn.span], [8]))
        # in this case the skew is -0.75 = 3 * (-1 - 1) / (2 - (-6))
        self.assertTrue(np.allclose([tc.skew, td.skew, th.skew, tn.skew], [-0.75]))

        # The asymmetry and (reduced) anisotropy should be the same for all the conventions
        # since they are hard-coded (and only defined) for the Haeberlen convention
        self.assertTrue(np.allclose([tc.asymmetry, td.asymmetry, th.asymmetry, tn.asymmetry], [0.2]))
        self.assertTrue(np.allclose([tc.reduced_anisotropy, td.reduced_anisotropy, th.reduced_anisotropy, tn.reduced_anisotropy], [-5]))
        self.assertTrue(np.allclose([tc.anisotropy, td.anisotropy, th.anisotropy, tn.anisotropy], [-7.5]))
    
    def test_tensor_euler_angles(self):
        """
        Test the Euler angles for the tensor class

        All of the Reference Euler angles in this test have been cross-referenced with the output of
        TensorView for MATLAB:
        https://doi.org/10.1016/j.ssnmr.2022.101849

        """

        data = np.diag([1, 2, -6])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        td = NMRTensor(data, NMRTensor.ORDER_DECREASING)
        th = NMRTensor(data, NMRTensor.ORDER_HAEBERLEN)
        tn = NMRTensor(data, NMRTensor.ORDER_NQR)

        # First let's make sure that the calculation of Euler angles fails correctly
        # for conventions other than zyz and zxz
        with self.assertRaises(ValueError):
            tc.equivalent_euler_angles(convention='abc')
        with self.assertRaises(ValueError):
            tc.euler_angles(convention='abc')


        # Where it gets tricky is the Euler angles
        euler_c = tc.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        euler_d = td.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        euler_h = th.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        euler_n = tn.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        
        ref_euler_c = np.array([
                                [ 90, 90,   0],
                                [ 90, 90, 180],
                                [270, 90, 180],
                                [270, 90,   0],
                                ])
        ref_euler_d = np.array([
                                [ 90,   0,   0],
                                [ 90,   0, 180],
                                [270, 180, 180],
                                [270, 180,   0],
                                ])
        ref_euler_n = np.array([
                                [  0,   0,   0],
                                [  0,   0, 180],
                                [180, 180, 180],
                                [180, 180,   0],
                                ]) # == abs ascending = NQR
        # in this case, the Haeberlen convention is the same as the decreasing convention
        ref_euler_h = ref_euler_d
        self.assertTrue(np.allclose(euler_c, ref_euler_c))
        self.assertTrue(np.allclose(euler_d, ref_euler_d))
        self.assertTrue(np.allclose(euler_n, ref_euler_n))
        self.assertTrue(np.allclose(euler_h, ref_euler_h))

        # Now a case without Gimbal lock
        data = np.array([
                    [ 1.00,  0.12,  0.13],
                    [ 0.21,  2.00,  0.23],
                    [ 0.31,  0.32, -6.00],
                    ])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        td = NMRTensor(data, NMRTensor.ORDER_DECREASING)
        th = NMRTensor(data, NMRTensor.ORDER_HAEBERLEN)
        tn = NMRTensor(data, NMRTensor.ORDER_NQR)
        
        # Eigenvalue ordering (make sure we're testing the right thing)
        eigs_ref = np.array([-6.01598555, 0.97774119,  2.03824436])
        self.assertTrue(np.allclose(tc.eigenvalues, eigs_ref[[0, 1, 2]]))
        self.assertTrue(np.allclose(td.eigenvalues, eigs_ref[[2, 1, 0]]))
        self.assertTrue(np.allclose(th.eigenvalues, eigs_ref[[2, 1, 0]]))
        self.assertTrue(np.allclose(tn.eigenvalues, eigs_ref[[1, 2, 0]]))

        # --- Euler ZYZ (active) convention --- #
        euler_c = tc.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        euler_d = td.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        euler_h = th.equivalent_euler_angles(convention='zyz') * 180 / np.pi
        euler_n = tn.equivalent_euler_angles(convention='zyz') * 180 / np.pi

        ref_euler_c = np.array([
            [ 80.51125264,  87.80920208, 178.59212804],
            [ 80.51125264,  87.80920208, 358.59212804],
            [260.51125264,  92.19079792,   1.40787196],
            [260.51125264,  92.19079792, 181.40787196],
            ])
        ref_euler_d = np.array([
            [227.77364892,   2.60398404,  32.71068295],
            [227.77364892,   2.60398404, 212.71068295],
            [ 47.77364892, 177.39601596, 147.28931705],
            [ 47.77364892, 177.39601596, 327.28931705],
            ])
        ref_euler_h = ref_euler_d # should be the same in this case
        ref_euler_n = np.array([
            [227.77364892,   2.60398404, 122.71068295],
            [227.77364892,   2.60398404, 302.71068295],
            [ 47.77364892, 177.39601596,  57.28931705],
            [ 47.77364892, 177.39601596, 237.28931705],
        ])
        self.assertTrue(np.allclose(euler_c, ref_euler_c))
        self.assertTrue(np.allclose(euler_d, ref_euler_d))
        self.assertTrue(np.allclose(euler_n, ref_euler_n))
        self.assertTrue(np.allclose(euler_h, ref_euler_h))

        # now the passive rotations - just check the first one for each
        euler_c = tc.euler_angles(convention='zyz', passive=True) * 180 / np.pi
        euler_d = td.euler_angles(convention='zyz', passive=True) * 180 / np.pi
        euler_h = th.euler_angles(convention='zyz', passive=True) * 180 / np.pi
        euler_n = tn.euler_angles(convention='zyz', passive=True) * 180 / np.pi
        
        self.assertTrue(np.allclose(euler_c, np.array([[1.40787196, 87.80920208, 99.48874736]])))
        self.assertTrue(np.allclose(euler_d, np.array([[147.28931705, 2.60398404, 312.22635108]])))
        # Haebelen convention is the same as decreasing convention for this case
        self.assertTrue(np.allclose(euler_h, euler_d))
        self.assertTrue(np.allclose(euler_n, np.array([[57.28931705, 2.60398404, 312.22635108]])))

        # --- ZXZ (active) convention --- #
        euler_c = tc.euler_angles(convention='zxz', passive=False) * 180 / np.pi
        euler_d = td.euler_angles(convention='zxz', passive=False) * 180 / np.pi
        euler_h = th.euler_angles(convention='zxz', passive=False) * 180 / np.pi
        euler_n = tn.euler_angles(convention='zxz', passive=False) * 180 / np.pi
        ref_euler_c =  [170.51125264,  87.80920208,  88.59212804]
        ref_euler_d =  [317.77364892,   2.60398404, 122.71068295]
        ref_euler_h =  [317.77364892,   2.60398404, 122.71068295]
        ref_euler_n =  [317.77364892,   2.60398404,  32.71068295]

        self.assertTrue(np.allclose(euler_c, ref_euler_c))
        self.assertTrue(np.allclose(euler_d, ref_euler_d))
        self.assertTrue(np.allclose(euler_h, ref_euler_h))
        self.assertTrue(np.allclose(euler_n, ref_euler_n))

        # --- ZXZ (passive) convention --- #
        euler_c = tc.euler_angles(convention='zxz', passive=True) * 180 / np.pi
        euler_d = td.euler_angles(convention='zxz', passive=True) * 180 / np.pi
        euler_h = th.euler_angles(convention='zxz', passive=True) * 180 / np.pi
        euler_n = tn.euler_angles(convention='zxz', passive=True) * 180 / np.pi
        ref_euler_c = [ 91.40787196,  87.80920208,   9.48874736]
        ref_euler_d = [ 57.28931705,   2.60398404, 222.22635108]
        ref_euler_h = [ 57.28931705,   2.60398404, 222.22635108]
        ref_euler_n = [147.28931705,   2.60398404, 222.22635108]

        self.assertTrue(np.allclose(euler_c, ref_euler_c))
        self.assertTrue(np.allclose(euler_d, ref_euler_d))
        self.assertTrue(np.allclose(euler_h, ref_euler_h))
        self.assertTrue(np.allclose(euler_n, ref_euler_n))

        # Check that the tensor is rotated correctly by the Euler angles
        self.assertTrue(_test_euler_rotation(
            tc.euler_angles(convention='zyz', passive=False),
            tc.eigenvalues,
            tc.eigenvectors,
            convention='zyz',
            passive=False,
        ))

        self.assertTrue(_test_euler_rotation(
            tc.euler_angles(convention='zyz', passive=True),
            tc.eigenvalues,
            tc.eigenvectors,
            convention='zyz',
            passive=True,
        ))

        self.assertTrue(_test_euler_rotation(
            tc.euler_angles(convention='zxz', passive=False),
            tc.eigenvalues,
            tc.eigenvectors,
            convention='zxz',
            passive=False,
        ))

        self.assertTrue(_test_euler_rotation(
            tc.euler_angles(convention='zxz', passive=True),
            tc.eigenvalues,
            tc.eigenvectors,
            convention='zxz',
            passive=True,
        ))



    def test_tensor_euler_edge_cases(self):
        # Now a case with 3 degenerate eigenvalues (spherical tensor)
        data = np.diag([1, 1, 1])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        self.assertTrue(np.allclose(tc.euler_angles(convention='zyz', passive = False), np.zeros(3)))
        self.assertTrue(np.allclose(tc.euler_angles(convention='zxz', passive = False), np.zeros(3)))
        self.assertTrue(np.allclose(tc.euler_angles(convention='zyz', passive = True), np.zeros(3)))
        self.assertTrue(np.allclose(tc.euler_angles(convention='zxz', passive = True), np.zeros(3)))

        # Now a case with no degenerate eigenvalues
        # but with Gimbal lock
        data = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        # confirm that the eigenvalues are sorted correctly
        self.assertTrue(np.allclose(tc.eigenvalues, [0.5, 1.5, 2.0]))
        evecs = tc.eigenvectors
        zyza = tc.equivalent_euler_angles(convention='zyz', passive = False) * 180 / np.pi
        zxza = tc.equivalent_euler_angles(convention='zxz', passive = False) * 180 / np.pi
        zyzp = tc.equivalent_euler_angles(convention='zyz', passive = True) * 180 / np.pi
        zxzp = tc.equivalent_euler_angles(convention='zxz', passive = True) * 180 / np.pi
        # TODO: re-enable these once reference values are available! 
        # self.assertTrue(np.allclose(zyza[0], np.array([135, 0, 0]))) 
        # self.assertTrue(np.allclose(zxza[0], np.array([135, 0, 0])))
        # self.assertTrue(np.allclose(zyzp[0], np.array([0, 0, 225])))
        # self.assertTrue(np.allclose(zxzp[0], np.array([0, 0, 225])))


        # More symmetric tensors
        data = np.diag([5,10,5])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        eulers = tc.euler_angles(convention='zyz')*180/np.pi
        
        self.assertTrue(np.allclose(eulers, np.array([90,90,0])))
        # TODO: this is not the same as TensorView for MATLAB, which gives [0,90,90]. soprano gives [90,90,0] - so something is 
        # Check if this is because of the bug in TensorView for MATLAB
        # self.assertTrue(np.allclose(tc.euler_angles(convention='zyz', passive=True)*180/np.pi, np.array([0,90,90])))

        data = np.diag([10,5,5])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        self.assertTrue(np.allclose(tc.euler_angles(convention='zyz')*180/np.pi, np.array([180,90,0])))
        # TODO: according to TensorView for MATLAB, this should be [0,90,0] or equivalent
        # soprano gives [90, 90, 0] - so something is not happening correctly when passive is True
        # Check if this is because of the bug in TensorView for MATLAB
        f = tc.euler_angles(convention='zyz', passive=True)*180/np.pi
        # self.assertTrue(np.allclose(tc.euler_angles(convention='zyz', passive=True)*180/np.pi, np.array([0,90,0])))

        # TODO: add more tests for zxz conventions and passive rotation

    def test_relative_euler_angles(self):
        # Test the relative Euler angles for the tensor class

        # Make sure if the tensors are the same, we get no rotations
        # - in the case of spherical tensors
        t1 = NMRTensor(np.diag([1, 1, 1]), NMRTensor.ORDER_INCREASING)
        releulers = t1.euler_to(t1, convention='zyz', passive=False)
        self.assertTrue(np.allclose(releulers, np.zeros((1,3))))
        # - and in the case of tensors with no degenerate eigenvalues
        t1 = NMRTensor(np.diag([1, 2, -6]), NMRTensor.ORDER_INCREASING)
        releulers = t1.euler_to(t1, convention='zyz', passive=False)
        self.assertTrue(np.allclose(releulers, np.zeros((1,3))))


        # ALA case from the TensorView for MATLAB examples dir
        ala_example_1  = np.array([
        [ -5.9766,   -60.302,   -10.8928],
        [-65.5206,   -23.0881,  -25.2372],
        [ -9.5073,   -28.2399,   56.2779],
        ]) # probably an MS tensor
        t1 = NMRTensor(ala_example_1, order=NMRTensor.ORDER_INCREASING)

        ala_example_2  = np.array([
            [-0.7806, 0.7215, 0.2987],
            [0.7215, 1.3736, 0.9829],
            [0.2987, 0.9829, -0.5929]
            ]) # probably an EFG tensor
        t2 = NMRTensor(ala_example_2, order=NMRTensor.ORDER_INCREASING)
        # first make sure the individual tensors give the correct Euler angles
        euler1 = t1.euler_angles(convention='zyz', passive=False)
        euler2 = t2.euler_angles(convention='zyz', passive=False)
        self.assertTrue(np.allclose(euler1*180/np.pi, np.array([ 300.4327, 30.3820, 112.0611])))
        self.assertTrue(np.allclose(euler2*180/np.pi, np.array([ 73.0924, 68.3864, 46.7300])))
        # relative Euler angles - ZYZ active convention
        releulers = t1.equivalent_euler_to(t2, convention='zyz', passive=False)
        releulers   = releulers * 180 / np.pi
        # this is in the order of TensorView for MATLAB
        # we get a slightly different ordering, but the same values
        ref_eulers = np.array([
            [335.10491563,  89.95022697,  24.80660839],
            [155.10491563,  90.04977303, 335.19339161],
            [155.10491563,  90.04977303, 155.19339161],
            [335.10491563,  89.95022697, 204.80660839],
            [ 24.89508437,  90.04977303, 204.80660839],
            [204.89508437,  89.95022697, 155.19339161],
            [204.89508437,  89.95022697, 335.19339161],
            [ 24.89508437,  90.04977303,  24.80660839],
            [204.89508437,  90.04977303, 204.80660839],
            [ 24.89508437,  89.95022697, 155.19339161],
            [ 24.89508437,  89.95022697, 335.19339161],
            [204.89508437,  90.04977303,  24.80660839],
            [155.10491563,  89.95022697,  24.80660839],
            [335.10491563,  90.04977303, 335.19339161],
            [335.10491563,  90.04977303, 155.19339161],
            [155.10491563,  89.95022697, 204.80660839],
            ])
        # compare the arrays, allowing for different orderings
        order_releulers =  tuple([np.lexsort((releulers[:,2],   releulers[:,0], releulers[:,1]))])
        order_ref_eulers = tuple([np.lexsort((ref_eulers[:,2], ref_eulers[:,0], ref_eulers[:,1]))])
        self.assertTrue(np.allclose(releulers[order_releulers], ref_eulers[order_ref_eulers]))


        # relative Euler angles - ZYZ passive convention
        releulers = t1.equivalent_euler_to(t2, convention='zyz', passive=True)
        releulers   = releulers * 180 / np.pi
        # this is in the order of TensorView for MATLAB
        # we get a slightly different ordering, but the same values
        ref_eulers = np.array([
            [155.19339161,  89.95022697, 204.89508437],
            [204.80660839,  90.04977303,  24.89508437],
            [ 24.80660839,  90.04977303,  24.89508437],
            [335.19339161,  89.95022697, 204.89508437],
            [335.19339161,  90.04977303, 155.10491563],
            [ 24.80660839,  89.95022697, 335.10491563],
            [204.80660839,  89.95022697, 335.10491563],
            [155.19339161,  90.04977303, 155.10491563],
            [335.19339161,  90.04977303, 335.10491563],
            [ 24.80660839,  89.95022697, 155.10491563],
            [204.80660839,  89.95022697, 155.10491563],
            [155.19339161,  90.04977303, 335.10491563],
            [155.19339161,  89.95022697,  24.89508437],
            [204.80660839,  90.04977303, 204.89508437],
            [ 24.80660839,  90.04977303, 204.89508437],
            [335.19339161,  89.95022697,  24.89508437],
        ])
        # compare the arrays, allowing for different orderings
        order_releulers =  tuple([np.lexsort((releulers[:,2],   releulers[:,0], releulers[:,1]))])
        order_ref_eulers = tuple([np.lexsort((ref_eulers[:,2], ref_eulers[:,0], ref_eulers[:,1]))])
        self.assertTrue(np.allclose(releulers[order_releulers], ref_eulers[order_ref_eulers]))
            
        # relative Euler angles - ZXZ active convention
        releulers = t1.equivalent_euler_to(t2, convention='zxz', passive=False)
        releulers   = releulers * 180 / np.pi
        # this is in the order of TensorView for MATLAB
        # we get a slightly different ordering, but the same values
        ref_eulers = np.array([
            [ 65.10491563,  89.95022697, 114.80660839],
            [245.10491563,  90.04977303, 245.19339161],
            [245.10491563,  90.04977303,  65.19339161],
            [ 65.10491563,  89.95022697, 294.80660839],
            [294.89508437,  90.04977303, 294.80660839],
            [114.89508437,  89.95022697,  65.19339161],
            [114.89508437,  89.95022697, 245.19339161],
            [294.89508437,  90.04977303, 114.80660839],
            [114.89508437,  90.04977303, 294.80660839],
            [294.89508437,  89.95022697,  65.19339161],
            [294.89508437,  89.95022697, 245.19339161],
            [114.89508437,  90.04977303, 114.80660839],
            [245.10491563,  89.95022697, 114.80660839],
            [ 65.10491563,  90.04977303, 245.19339161],
            [ 65.10491563,  90.04977303,  65.19339161],
            [245.10491563,  89.95022697, 294.80660839],
       ])
        # compare the arrays, allowing for different orderings
        order_releulers =  tuple([np.lexsort((releulers[:,2],   releulers[:,0], releulers[:,1]))])
        order_ref_eulers = tuple([np.lexsort((ref_eulers[:,2], ref_eulers[:,0], ref_eulers[:,1]))])
        self.assertTrue(np.allclose(releulers[order_releulers], ref_eulers[order_ref_eulers]))

        # relative Euler angles - ZXZ passive convention
        releulers = t1.equivalent_euler_to(t2, convention='zxz', passive=True)
        releulers   = releulers * 180 / np.pi
        # this is in the order of TensorView for MATLAB
        # we get a slightly different ordering, but the same values
        ref_eulers = np.array([
            [ 65.19339161,  89.95022697, 114.89508437],
            [294.80660839,  90.04977303, 294.89508437],
            [114.80660839,  90.04977303, 294.89508437],
            [245.19339161,  89.95022697, 114.89508437],
            [245.19339161,  90.04977303, 245.10491563],
            [114.80660839,  89.95022697,  65.10491563],
            [294.80660839,  89.95022697,  65.10491563],
            [ 65.19339161,  90.04977303, 245.10491563],
            [245.19339161,  90.04977303,  65.10491563],
            [114.80660839,  89.95022697, 245.10491563],
            [294.80660839,  89.95022697, 245.10491563],
            [ 65.19339161,  90.04977303,  65.10491563],
            [ 65.19339161,  89.95022697, 294.89508437],
            [294.80660839,  90.04977303, 114.89508437],
            [114.80660839,  90.04977303, 114.89508437],
            [245.19339161,  89.95022697, 294.89508437],
        ])
        # compare the arrays, allowing for different orderings
        order_releulers =  tuple([np.lexsort((releulers[:,2],   releulers[:,0], releulers[:,1]))])
        order_ref_eulers = tuple([np.lexsort((ref_eulers[:,2], ref_eulers[:,0], ref_eulers[:,1]))])
        self.assertTrue(np.allclose(releulers[order_releulers], ref_eulers[order_ref_eulers]))


    # Relative Euler angles with Gimbal lock
    def test_relative_euler_angles_gimbal(self):
        # Gimbal lock case
        c30 = np.sqrt(3)/2
        c45 = np.sqrt(2)/2
        c60 = 0.5
        data1 = np.array([
            [c30,  c60, 0.0],
            [-c60, c30, 0.0],
            [ 0.0, 0.0, 1.0]
        ])

        data2 = np.array([
            [c45,  c45, 0.0],
            [-c45, c45, 0.0],
            [ 0.0, 0.0, 1.0]
        ])
        t1 = NMRTensor(data1)
        t2 = NMRTensor(data2)

        releulers = t2.euler_to(t1, convention='zyz', passive=False)
        # according to TensorView for MATLAB this should be [0, 0, 0]
        self.assertTrue(np.allclose(releulers, np.zeros(3))) 


    # Axially symmetric tensors
    def test_relative_euler_angles_axial_symmetry(self):
        
        # First an example from the MagresView2 tests 
        # (Both are axially symmetric tensors - tricky case!)
        # The first is *almost* a spherical tensor, but not quite
        data1 = np.array([
            [0.93869474, 0.33129348, -0.09537721],
            [0.33771007, -0.93925902, 0.06119153],
            [-0.06931155, -0.08965002, -0.99355865]
            ])

        data2 = np.array([
            [-0.52412461, 0.49126909, -0.69566377],
            [-0.56320663, 0.41277966, 0.71582906],
            [0.63882054, 0.76698607, 0.06033803]
            ])
        t1 = NMRTensor(data1)
        t2 = NMRTensor(data2)

        self.assertTrue(np.allclose(t1.eigenvalues, np.array([-0.99706147, -0.99706146,  1.        ])))
        self.assertTrue(np.allclose(t2.eigenvalues, np.array([-0.52550346, -0.52550346,  1.        ])))
        # -- ZYZ -- # 
        # first make sure the individual tensors give the correct Euler angles
        euler1 = t1.euler_angles(convention='zyz', passive=False)
        euler2 = t2.euler_angles(convention='zyz', passive=False)
        self.assertTrue(np.allclose(euler1*180/np.pi, np.array([ 189.8040, 87.5997, 0.0])))
        self.assertTrue(np.allclose(euler2*180/np.pi, np.array([ 92.1953, 51.7056, 0.0])))

        # now check the relative Euler angles
        releulers = t1.equivalent_euler_to(t2, convention='zyz', passive=False)
        releulers = releulers * 180 / np.pi
        # From TensorView for MATLAB:
        ref_eulers = np.array([
            [  0.0000,  85.5337,   0.0000], # 1
            [180.0000,  94.4663,   0.0000], # 2
            [180.0000,  94.4663, 180.0000], # 3
            [  0.0000,  85.5337, 180.0000], # 4
            [  0.0000,  94.4663, 180.0000], # 5
            [180.0000,  85.5337, 180.0000], # 6
            [180.0000,  85.5337,   0.0000], # 7
            [  0.0000,  94.4663,   0.0000], # 8
            [180.0000,  94.4663, 180.0000], # 9
            [  0.0000,  85.5337, 180.0000], # 10
            [  0.0000,  85.5337,   0.0000], # 11
            [180.0000,  94.4663,   0.0000], # 12
            [180.0000,  85.5337,   0.0000], # 13
            [  0.0000,  94.4663,   0.0000], # 14
            [  0.0000,  94.4663, 180.0000], # 15
            [180.0000,  85.5337, 180.0000], # 16
        ])
        self.assertTrue(np.allclose(releulers, ref_eulers))



        # -- ZXZ -- # 
        data1 = np.array([
            [0.93869474, 0.33129348, -0.09537721],
            [0.33771007, -0.93925902, 0.06119153],
            [-0.06931155, -0.08965002, -0.99355865]
            ])

        data2 = np.array([
            [-0.52412461, 0.49126909, -0.69566377],
            [-0.56320663, 0.41277966, 0.71582906],
            [0.63882054, 0.76698607, 0.06033803]
            ])
        t1 = NMRTensor(data1)
        t2 = NMRTensor(data2)
        # first make sure the individual tensors give the correct Euler angles
        euler1 = t1.euler_angles(convention='zxz', passive=False)
        euler2 = t2.euler_angles(convention='zxz', passive=False)

        self.assertTrue(np.allclose(euler1*180/np.pi, np.array([ 279.8040, 87.5997, 0.0])))
        self.assertTrue(np.allclose(euler2*180/np.pi, np.array([ 182.1953, 51.7056, 0.0])))

        # now check the relative Euler angles
        releulers = t1.equivalent_euler_to(t2, convention='zxz', passive=False)
        releulers = releulers * 180 / np.pi
        # From TensorView for MATLAB:
        # TODO: these are not the same as the MATLAB results
        # MATLAB gives e.g. [183.58, 85.537, 74.68]
        # But doesn't this go against the equations in 
        # the paper (Table 1) for two axially symmetric tensors? 
        # It might be to do with rounding issues since t1 is close to spherical
        # but not quite. We presumably are using more precision here (?).
        ref_eulers = np.array([
            [ 90.        ,  85.53372296,   0.        ],
            [270.        ,  94.46627704,   0.        ],
            [270.        ,  94.46627704, 180.        ],
            [ 90.        ,  85.53372296, 180.        ],
            [270.        ,  94.46627704, 180.        ],
            [ 90.        ,  85.53372296, 180.        ],
            [ 90.        ,  85.53372296,   0.        ],
            [270.        ,  94.46627704,   0.        ],
            [ 90.        ,  94.46627704, 180.        ],
            [270.        ,  85.53372296, 180.        ],
            [270.        ,  85.53372296,   0.        ],
            [ 90.        ,  94.46627704,   0.        ],
            [270.        ,  85.53372296,   0.        ],
            [ 90.        ,  94.46627704,   0.        ],
            [ 90.        ,  94.46627704, 180.        ],
            [270.        ,  85.53372296, 180.        ]
        ])
        self.assertTrue(np.allclose(releulers, ref_eulers))


        # Now let's test the case where the first tensor is axially symmetry 
        # and the second has no symmetry
        data1 = np.diag([1, 2, 1])
        
        data2 = np.array([
                    [ 1.00,  0.12,  0.13],
                    [ 0.21,  2.00,  0.23],
                    [ 0.31,  0.32, -6.00],
                    ])
        
        t1 = NMRTensor(data1, NMRTensor.ORDER_INCREASING)
        t2 = NMRTensor(data2, NMRTensor.ORDER_INCREASING)

        # first make sure the individual tensors give the correct Euler angles
        euler1 = t1.euler_angles(convention='zyz', passive=False)
        euler2 = t2.euler_angles(convention='zyz', passive=False)
        # Note the order of the equivalent Euler angles is different from the MATLAB results
        # but the 90, 90 0 is one of the equivalent Euler angle sets for this tensor
        self.assertTrue(np.allclose(euler1*180/np.pi, np.array([ 90.0, 90.0, 0.0])))
        self.assertTrue(np.allclose(euler2*180/np.pi, np.array([ 80.511,  87.809, 178.592])))
        # now check the relative Euler angles
       
        # not symmetric to axially symmetric: ZYZ active
        azyz = t2.equivalent_euler_to(t1, convention='zyz', passive=False) * 180 / np.pi
        # Correct angle set wrt to MATLAB, albeit in a different order
        azyz_ref = np.array([
            [  0.,           9.73611618,  78.52514082],
            [180.,         170.26388382, 281.47485918],
            [180.,         170.26388382, 101.47485918],
            [  0.,           9.73611618, 258.52514082],
            [  0.,         170.26388382, 258.52514082],
            [180.,           9.73611618, 101.47485918],
            [180.,           9.73611618, 281.47485918],
            [  0.,         170.26388382,  78.52514082],
            [180.,         170.26388382, 258.52514082],
            [  0.,           9.73611618, 101.47485918],
            [  0.,           9.73611618, 281.47485918],
            [180.,         170.26388382,  78.52514082],
            [180.,           9.73611618,  78.52514082],
            [  0.,         170.26388382, 281.47485918],
            [  0.,         170.26388382, 101.47485918],
            [180.,           9.73611618, 258.52514082],
            ])
        order_azyz =  tuple([np.lexsort((azyz[:,2],   azyz[:,0], azyz[:,1]))])
        order_azyz_ref = tuple([np.lexsort((azyz_ref[:,2], azyz_ref[:,0], azyz_ref[:,1]))])
        self.assertTrue(np.allclose(azyz[order_azyz], azyz_ref[order_azyz_ref]))

        # not symmetric to axially symmetric: ZYZ passive
        pzyz = (t2.equivalent_euler_to(t1, convention='zyz', passive=True) * 180 / np.pi)
        # Correct angle set wrt to MATLAB, albeit in a different order
        pzyz_ref = np.array([
            [ 78.52514082,   9.73611618,   0.        ],
            [281.47485918, 170.26388382, 180.        ],
            [101.47485918, 170.26388382, 180.        ],
            [258.52514082,   9.73611618,   0.        ],
            [258.52514082, 170.26388382,   0.        ],
            [101.47485918,   9.73611618, 180.        ],
            [281.47485918,   9.73611618, 180.        ],
            [ 78.52514082, 170.26388382,   0.        ],
            [258.52514082, 170.26388382, 180.        ],
            [101.47485918,   9.73611618,   0.        ],
            [281.47485918,   9.73611618,   0.        ],
            [ 78.52514082, 170.26388382, 180.        ],
            [ 78.52514082,   9.73611618, 180.        ],
            [281.47485918, 170.26388382,   0.        ],
            [101.47485918, 170.26388382,   0.        ],
            [258.52514082,   9.73611618, 180.        ],
            ])
        order_pzyz =  tuple([np.lexsort((pzyz[:,2],   pzyz[:,0], pzyz[:,1]))])
        order_pzyz_ref = tuple([np.lexsort((pzyz_ref[:,2], pzyz_ref[:,0], pzyz_ref[:,1]))])
        self.assertTrue(np.allclose(pzyz[order_pzyz], pzyz_ref[order_pzyz_ref]))




        # not symmetric to axially symmetric: ZXZ active
        azxz = (t2.equivalent_euler_to(t1, convention='zxz', passive=False) * 180 / np.pi)
        # Correct angle set wrt to MATLAB, albeit in a different order
        azxz_ref = np.array([
            [  0.,           9.73611618, 348.52514082],
            [180.,         170.26388382,  11.47485918],
            [180.,         170.26388382, 191.47485918],
            [  0.,           9.73611618, 168.52514082],
            [  0.,         170.26388382, 168.52514082],
            [180.,           9.73611618, 191.47485918],
            [180.,           9.73611618,  11.47485918],
            [  0.,         170.26388382, 348.52514082],
            [180.,         170.26388382, 168.52514082],
            [  0.,           9.73611618, 191.47485918],
            [  0.,           9.73611618,  11.47485918],
            [180.,         170.26388382, 348.52514082],
            [180.,           9.73611618, 348.52514082],
            [  0.,         170.26388382,  11.47485918],
            [  0.,         170.26388382, 191.47485918],
            [180.,           9.73611618 ,168.52514082]
            ])
        order_azxz =  tuple([np.lexsort((azxz[:,2],   azxz[:,0], azxz[:,1]))])
        order_azxz_ref = tuple([np.lexsort((azxz_ref[:,2], azxz_ref[:,0], azxz_ref[:,1]))])
        self.assertTrue(np.allclose(azxz[order_azxz], azxz_ref[order_azxz_ref]))

        # not symmetric to axially symmetric: ZXZ passive
        pzxz = (t2.equivalent_euler_to(t1, convention='zxz', passive=True) * 180 / np.pi)
        # Correct angle set wrt to MATLAB, albeit in a different order
        pzxz_ref = np.array([
            [348.52514082,   9.73611618,   0.        ],
            [ 11.47485918, 170.26388382, 180.        ],
            [191.47485918, 170.26388382, 180.        ],
            [168.52514082,   9.73611618,   0.        ],
            [168.52514082, 170.26388382,   0.        ],
            [191.47485918,   9.73611618, 180.        ],
            [ 11.47485918,   9.73611618, 180.        ],
            [348.52514082, 170.26388382,   0.        ],
            [168.52514082, 170.26388382, 180.        ],
            [191.47485918,   9.73611618,   0.        ],
            [ 11.47485918,   9.73611618,   0.        ],
            [348.52514082, 170.26388382, 180.        ],
            [348.52514082,   9.73611618, 180.        ],
            [ 11.47485918, 170.26388382,   0.        ],
            [191.47485918, 170.26388382,   0.        ],
            [168.52514082,   9.73611618, 180.        ],
            ])
        order_pzxz =  tuple([np.lexsort((pzxz[:,2],   pzxz[:,0], pzxz[:,1]))])
        order_pzxz_ref = tuple([np.lexsort((pzxz_ref[:,2], pzxz_ref[:,0], pzxz_ref[:,1]))])
        self.assertTrue(np.allclose(pzxz[order_pzxz], pzxz_ref[order_pzxz_ref]))



        # axially symmetric to non-symmetric: ZYZ active
        azyz = t1.equivalent_euler_to(t2, convention='zyz', passive=False) * 180 / np.pi
        order_azyz =  tuple([np.lexsort((azyz[:,2],   azyz[:,0], azyz[:,1]))])
        # active zyz from 1 to 2 should give the same results as passive 2 to 1, ignoring order of 
        self.assertTrue(np.allclose(azyz[order_azyz], pzyz_ref[order_pzyz_ref]))

        # axially symmetric to non-symmetric: ZYZ passive
        pzyz = t1.equivalent_euler_to(t2, convention='zyz', passive=True) * 180 / np.pi
        order_pzyz =  tuple([np.lexsort((pzyz[:,2],   pzyz[:,0], pzyz[:,1]))])
        # passive zyz from 1 to 2 should give the same results as active 2 to 1, ignoring order of
        self.assertTrue(np.allclose(pzyz[order_pzyz], azyz_ref[order_azyz_ref]))

        # axially symmetric to non-symmetric: ZXZ active
        azxz = t1.equivalent_euler_to(t2, convention='zxz', passive=False) * 180 / np.pi
        order_azxz =  tuple([np.lexsort((azxz[:,2],   azxz[:,0], azxz[:,1]))])
        # active zxz from 1 to 2 should give the same results as passive 2 to 1, ignoring order of
        self.assertTrue(np.allclose(azxz[order_azxz], pzxz_ref[order_pzxz_ref]))

        # axially symmetric to non-symmetric: ZXZ passive
        pzxz = t1.equivalent_euler_to(t2, convention='zxz', passive=True) * 180 / np.pi
        order_pzxz =  tuple([np.lexsort((pzxz[:,2],   pzxz[:,0], pzxz[:,1]))])
        # passive zxz from 1 to 2 should give the same results as active 2 to 1, ignoring order of
        self.assertTrue(np.allclose(pzxz[order_pzxz], azxz_ref[order_azxz_ref]))


    def test_diprotavg(self):
        # Test dipolar rotational averaging

        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

        from ase.quaternions import Quaternion
        from soprano.properties.transform import Rotate
        from soprano.collection import AtomsCollection
        from soprano.collection.generate import transformGen
        from soprano.properties.nmr import DipolarTensor

        N = 30  # Number of averaging steps
        axis = np.array([1.0, 1.0, 0])
        axis /= np.linalg.norm(axis)

        rot = Rotate(quaternion=Quaternion.from_axis_angle(axis, 2 * np.pi / N))

        rot_eth = AtomsCollection(transformGen(eth, rot, N))

        rot_dip = [D[(0, 1)].data for D in DipolarTensor.get(rot_eth, sel_i=[0], sel_j=[1])]

        dip_avg_num = np.average(rot_dip, axis=0)

        dip_tens = NMRTensor.make_dipolar(eth, 0, 1, rotation_axis=axis)
        dip_avg_tens = dip_tens.data

        self.assertTrue(np.allclose(dip_avg_num, dip_avg_tens))

        # Test eigenvectors
        evecs = dip_tens.eigenvectors
        self.assertTrue(np.allclose(np.dot(evecs.T, evecs), np.eye(3)))


if __name__ == "__main__":
    unittest.main()
