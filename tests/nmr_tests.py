#!/usr/bin/env python
"""
Test code for NMR properties
"""


import os
import unittest

import numpy as np
from ase import io
from ase.quaternions import Quaternion
import pytest


from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor
from soprano.nmr.utils import _test_euler_rotation
from soprano.properties.nmr import (
    EFGNQR,
    DipolarCoupling,
    DipolarDiagonal,
    DipolarTensor,
    EFGAsymmetry,
    EFGQuadrupolarConstant,
    EFGQuadrupolarProduct,
    EFGTensor,
    EFGVzz,
    MSAnisotropy,
    MSAsymmetry,
    MSIsotropy,
    MSReducedAnisotropy,
    MSSkew,
    MSSpan,
)
from soprano.properties.nmr.efg import EFGAnisotropy, EFGDiagonal, EFGEuler, EFGQuaternion, EFGReducedAnisotropy, EFGSkew, EFGSpan
from soprano.properties.nmr.ms import MSEuler, MSShielding, MSShift, MSTensor

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
                mlabs[j] = f"{e}_{i + 1}"

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
            ms_tens = NMRTensor(ms[i], order=NMRTensor.ORDER_HAEBERLEN)
            evals, evecs = diag[i]

            self.assertAlmostEqual(iso[i], ms_tens.isotropy)
            self.assertAlmostEqual(aniso[i], ms_tens.anisotropy)
            self.assertAlmostEqual(r_aniso[i], ms_tens.reduced_anisotropy)
            self.assertAlmostEqual(asymm[i], ms_tens.asymmetry)
            self.assertAlmostEqual(span[i], ms_tens.span)
            self.assertAlmostEqual(skew[i], ms_tens.skew)

            self.assertTrue(np.isclose(evals, sorted(ms_tens.eigenvalues)).all())
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
        with pytest.warns(UserWarning, match="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles."):
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
        with pytest.warns(UserWarning, match="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles."):
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
        
        # TODO: re-enable these once reference values are available!
        # with pytest.warns(UserWarning, match="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles."):
        #     zyza = tc.equivalent_euler_angles(convention='zyz', passive = False) * 180 / np.pi
        #     zxza = tc.equivalent_euler_angles(convention='zxz', passive = False) * 180 / np.pi
        #     zyzp = tc.equivalent_euler_angles(convention='zyz', passive = True) * 180 / np.pi
        #     zxzp = tc.equivalent_euler_angles(convention='zxz', passive = True) * 180 / np.pi
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
        with pytest.warns(UserWarning, match="The tensors are identical. Returning zero Euler angles."):
            releulers = t1.euler_to(t1, convention='zyz', passive=False)
            self.assertTrue(np.allclose(releulers, np.zeros((1,3))))
        # - and in the case of tensors with no degenerate eigenvalues
        t1 = NMRTensor(np.diag([1, 2, -6]), NMRTensor.ORDER_INCREASING)
        with pytest.warns(UserWarning, match="The tensors are identical. Returning zero Euler angles."):
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

        with pytest.warns(UserWarning, match="The tensors are perfectly aligned. Returning zero Euler angles."):
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

        from soprano.collection import AtomsCollection
        from soprano.collection.generate import transformGen
        from soprano.properties.nmr import DipolarTensor
        from soprano.properties.transform import Rotate

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

    def test_tensor_arithmetic(self):
        """Test the arithmetic operations for NMRTensor."""
        # Create simple tensors for testing
        data1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        data2 = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]
        ])
        data3 = np.ones((3, 3)) * 3.0
        
        t1 = NMRTensor(data1)
        t2 = NMRTensor(data2)
        t3 = NMRTensor(data3)
        
        # Test addition
        result = t1 + t2
        self.assertTrue(np.allclose(result.data, data1 + data2))
        
        # Test scalar addition
        result = t1 + 5
        self.assertTrue(np.allclose(result.data, data1 + 5))
        
        # Test right scalar addition
        result = 5 + t1
        self.assertTrue(np.allclose(result.data, 5 + data1))
        
        # Test array addition
        arr = np.ones((3, 3))
        result = t1 + arr
        self.assertTrue(np.allclose(result.data, data1 + arr))
        
        # Test right array addition
        result = arr + t1
        self.assertTrue(np.allclose(result.data, arr + data1))
        
        # Test invalid array addition
        with self.assertRaises(ValueError):
            t1 + np.ones(4)
            
        # Test subtraction
        result = t1 - t2
        self.assertTrue(np.allclose(result.data, data1 - data2))
        
        # Test scalar subtraction
        result = t1 - 2
        self.assertTrue(np.allclose(result.data, data1 - 2))
        
        # Test right scalar subtraction
        result = 10 - t1
        self.assertTrue(np.allclose(result.data, 10 - data1))
        
        # Test array subtraction
        result = t1 - arr
        self.assertTrue(np.allclose(result.data, data1 - arr))
        
        # Test right array subtraction
        result = arr - t1
        self.assertTrue(np.allclose(result.data, arr - data1))
        
        # Test invalid array subtraction
        with self.assertRaises(ValueError):
            t1 - np.ones(4)
            
        # Test multiplication
        result = t1 * t2
        self.assertTrue(np.allclose(result.data, data1 * data2))
        
        # Test scalar multiplication
        result = t1 * 3
        self.assertTrue(np.allclose(result.data, data1 * 3))
        
        # Test right scalar multiplication
        result = 3 * t1
        self.assertTrue(np.allclose(result.data, 3 * data1))
        
        # Test array multiplication
        result = t1 * arr
        self.assertTrue(np.allclose(result.data, data1 * arr))
        
        # Test right array multiplication
        result = arr * t1
        self.assertTrue(np.allclose(result.data, arr * data1))
        
        # Test invalid array multiplication
        with self.assertRaises(ValueError):
            t1 * np.ones(4)
            
        # Test division
        result = t1 / t3
        self.assertTrue(np.allclose(result.data, data1 / data3))
        
        # Test scalar division
        result = t1 / 2
        self.assertTrue(np.allclose(result.data, data1 / 2))
        
        # Test right scalar division
        result = 6 / t3
        self.assertTrue(np.allclose(result.data, 6 / data3))
        
        # Test array division
        result = t1 / arr
        self.assertTrue(np.allclose(result.data, data1 / arr))
        
        # Test right array division
        result = arr / t3
        self.assertTrue(np.allclose(result.data, arr / data3))
        
        # Test invalid array division
        with self.assertRaises(ValueError):
            t1 / np.ones(4)
            
        # Test matrix multiplication
        result = t1 @ t2
        self.assertTrue(np.allclose(result.data, data1 @ data2))
        
        # Test matrix multiplication with array
        result = t1 @ arr
        self.assertTrue(np.allclose(result.data, data1 @ arr))
        
        # Test right matrix multiplication with array
        result = arr @ t1
        self.assertTrue(np.allclose(result.data, arr @ data1))
        
        # Test matrix multiplication with vector
        vec = np.array([1.0, 2.0, 3.0])
        result = t1 @ vec
        self.assertTrue(np.allclose(result, data1 @ vec))
        
        # Test invalid matrix multiplication
        with self.assertRaises(ValueError):
            t1 @ np.ones(4)
            
        # Test negation
        result = -t1
        self.assertTrue(np.allclose(result.data, -data1))
        
        # Test positive 
        result = +t1
        self.assertTrue(np.allclose(result.data, data1))
        
        # Test equality
        t3 = NMRTensor(data1.copy())
        self.assertTrue(t1 == t3)
        self.assertFalse(t1 == t2)
        
        # Test equality with array
        self.assertTrue(t1 == data1)
        self.assertFalse(t1 == data2)
        
        # Test inequality with non-tensor/array
        self.assertFalse(t1 == "string")
        self.assertFalse(t1 == 123)

    def test_tensor_mean(self):
        """Test the mean class method for NMRTensor."""
        # Create some tensors for testing
        data1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        data2 = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [0.0, 0.0, 6.0]
        ])
        data3 = np.array([
            [7.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 9.0]
        ])
        
        t1 = NMRTensor(data1)
        t2 = NMRTensor(data2)
        t3 = NMRTensor(data3)
        
        # Calculate the mean
        mean_tensor = NMRTensor.mean([t1, t2, t3])
        
        # Calculate the expected result
        expected_mean = (data1 + data2 + data3) / 3
        
        # Check if the result is correct
        self.assertTrue(np.allclose(mean_tensor.data, expected_mean))
        
        # Test with empty list
        with self.assertRaises(ValueError):
            NMRTensor.mean([])
        
        # # Test flattening of nested list with axis=None
        # nested_tensors = [[t1], [t2, t3]]
        # mean_flat = NMRTensor.mean(nested_tensors)
        # expected_flat = (data1 + data2 + data3) / 3
        # self.assertTrue(np.allclose(mean_flat.data, expected_flat))
        
    def test_tensor_mean_with_weights(self):
        """Test the mean class method with weights."""
        # Create test tensors
        data1 = np.eye(3)
        data2 = 2 * np.eye(3)
        data3 = 3 * np.eye(3)
        
        t1 = NMRTensor(data1)
        t2 = NMRTensor(data2)
        t3 = NMRTensor(data3)
        
        # Test with weights - simple case
        weights = np.array([1.0, 2.0, 3.0])
        mean_tensor = NMRTensor.mean([t1, t2, t3], weights=weights)
        
        # Calculate expected weighted average: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
        expected_mean = (data1 * weights[0] + data2 * weights[1] + data3 * weights[2]) / np.sum(weights)
        self.assertTrue(np.allclose(mean_tensor.data, expected_mean))
        
        # Test with weights on a nested structure
        nested_tensors = [[t1, t2], [t3, t1]]
        nested_weights = np.array([0.5, 1.5])  # Weights for the two rows
        
        # For axis=0, weights apply to rows
        mean_columns = NMRTensor.mean(nested_tensors, axis=0, weights=nested_weights)
        
        # Expected: first column is (0.5*t1 + 1.5*t3)/(0.5+1.5), second is (0.5*t2 + 1.5*t1)/(0.5+1.5)
        expected_col1 = (0.5 * data1 + 1.5 * data3) / 2.0
        expected_col2 = (0.5 * data2 + 1.5 * data1) / 2.0
        
        self.assertTrue(np.allclose(mean_columns[0].data, expected_col1))
        self.assertTrue(np.allclose(mean_columns[1].data, expected_col2))
        
    def test_tensor_mean_axis(self):
        """Test the mean class method with axis selection."""
        # Create a 2D grid of tensors
        data_base = np.eye(3)
        tensors_2d = [
            [NMRTensor(data_base * 1), NMRTensor(data_base * 2), NMRTensor(data_base * 3)],
            [NMRTensor(data_base * 4), NMRTensor(data_base * 5), NMRTensor(data_base * 6)]
        ]
        
        # Test mean along axis=0 (average the rows)
        mean_axis0 = NMRTensor.mean(tensors_2d, axis=0)
        
        # Expected: 3 tensors, each an average of the corresponding column
        expected0_0 = (data_base * 1 + data_base * 4) / 2  # First column average
        expected0_1 = (data_base * 2 + data_base * 5) / 2  # Second column average
        expected0_2 = (data_base * 3 + data_base * 6) / 2  # Third column average
        
        self.assertTrue(np.allclose(mean_axis0[0].data, expected0_0))
        self.assertTrue(np.allclose(mean_axis0[1].data, expected0_1))
        self.assertTrue(np.allclose(mean_axis0[2].data, expected0_2))
        self.assertEqual(len(mean_axis0), 3)
        
        # Test mean along axis=1 (average the columns)
        mean_axis1 = NMRTensor.mean(tensors_2d, axis=1)
        
        # Expected: 2 tensors, each an average of the corresponding row
        expected1_0 = (data_base * 1 + data_base * 2 + data_base * 3) / 3  # First row average
        expected1_1 = (data_base * 4 + data_base * 5 + data_base * 6) / 3  # Second row average
        
        self.assertTrue(np.allclose(mean_axis1[0].data, expected1_0))
        self.assertTrue(np.allclose(mean_axis1[1].data, expected1_1))
        self.assertEqual(len(mean_axis1), 2)
        
        # Test with a single list of tensors - should behave the same with axis=None or axis=0
        tensors_1d = [NMRTensor(data_base * i) for i in range(1, 4)]
        mean_none = NMRTensor.mean(tensors_1d)
        mean_axis0_1d = NMRTensor.mean(tensors_1d, axis=0)
        expected_1d = data_base * 2  # (1+2+3)/3 = 2
        
        self.assertTrue(np.allclose(mean_none.data, expected_1d))
        self.assertTrue(np.allclose(mean_axis0_1d.data, expected_1d))
        
    def test_tensor_mean_errors(self):
        """Test error cases for the mean class method."""
        data_base = np.eye(3)
        t1 = NMRTensor(data_base)
        t2 = NMRTensor(data_base * 2)
        t3 = NMRTensor(data_base * 3)
        
        tensors_1d = [t1, t2, t3]
        tensors_2d = [[t1, t2], [t3, t1]]
        
        # Test with invalid axis
        with self.assertRaises(ValueError):
            NMRTensor.mean(tensors_1d, axis=2)  # Axis out of bounds
            
        with self.assertRaises(ValueError):
            NMRTensor.mean(tensors_2d, axis=3)  # Axis out of bounds
        
        # Test with wrong weights length
        with self.assertRaises(ValueError):
            NMRTensor.mean(tensors_1d, weights=np.array([1, 2]))  # Should be length 3
            
        with self.assertRaises(ValueError):
            NMRTensor.mean(tensors_2d, axis=0, weights=np.array([1, 2, 3]))  # Should be length 2
            
        with self.assertRaises(ValueError):
            NMRTensor.mean(tensors_2d, axis=1, weights=np.array([1, 2, 3]))  # Should be length 2 and 3 respectively
    
    def test_tensor_mean_with_subclass(self):
        """Test mean works properly with tensor subclasses."""
        # Create magnetic shielding tensors for testing
        ms1 = MagneticShielding(np.eye(3), species='13C', reference=100)
        ms2 = MagneticShielding(np.eye(3) * 2, species='13C', reference=100)
        ms3 = MagneticShielding(np.eye(3) * 3, species='13C', reference=100)
        
        # Test with same subclass
        mean_ms = MagneticShielding.mean([ms1, ms2, ms3])
        self.assertIsInstance(mean_ms, MagneticShielding)
        self.assertEqual(mean_ms.species, '13C')
        self.assertEqual(mean_ms.reference, 100)
        
        # Test mean with different axes
        ms_2d = [[ms1, ms2], [ms3, ms1]]
        
        # Mean along axis 0 (columns)
        mean_col = MagneticShielding.mean(ms_2d, axis=0)
        self.assertEqual(len(mean_col), 2)
        self.assertIsInstance(mean_col[0], MagneticShielding)
        self.assertIsInstance(mean_col[1], MagneticShielding)
        self.assertTrue(np.allclose(mean_col[0].data, (np.eye(3) + np.eye(3) * 3) / 2))
        
        # Mean along axis 1 (rows)
        mean_row = MagneticShielding.mean(ms_2d, axis=1)
        self.assertEqual(len(mean_row), 2)
        self.assertIsInstance(mean_row[0], MagneticShielding)
        self.assertIsInstance(mean_row[1], MagneticShielding)
        self.assertTrue(np.allclose(mean_row[0].data, (np.eye(3) + np.eye(3) * 2) / 2))
        
        # Test with weights
        weights = [1.0, 2.0, 3.0]
        mean_weighted = MagneticShielding.mean([ms1, ms2, ms3], weights=weights)
        expected = (np.eye(3) + np.eye(3) * 2 * 2 + np.eye(3) * 3 * 3) / 6
        self.assertTrue(np.allclose(mean_weighted.data, expected))
        
        # Test weighted mean with axis
        row_weights = [1.0, 2.0]
        mean_row_weighted = MagneticShielding.mean(ms_2d, axis=1, weights=row_weights)
        self.assertEqual(len(mean_row_weighted), 2)
        expected_row1 = (np.eye(3) * 1 + np.eye(3) * 2 * 2) / 3
        self.assertTrue(np.allclose(mean_row_weighted[0].data, expected_row1))

class TestMagneticShielding(unittest.TestCase):

    def setUp(self):
        # Setup a sample tensor for testing
        evals = np.array([1.0, 2.0, -6.0])
        evecs = np.eye(3)
        self.tensor = MagneticShielding([evals, evecs], species='2H', reference=0)

    def test_initialization(self):
        # Test if the object is initialized correctly
        self.assertIsInstance(self.tensor, MagneticShielding)
        self.assertIsInstance(self.tensor, NMRTensor)

        # Test if the order is correct
        self.assertEqual(self.tensor.order, NMRTensor.ORDER_HAEBERLEN)

        # Test if the species and isotope are set correctly
        self.assertEqual(self.tensor.species, '2H')
        self.assertEqual(self.tensor.element, 'H')

    def test_properties(self):
        # Should be sorted according to Haeberlen convention by default
        np.testing.assert_array_equal(self.tensor.eigenvalues, np.array([2.0, 1.0, -6.0]))
        # 'x' and 'y' swapped and 'z' is negative since we ensure a right-handed coordinate system
        np.testing.assert_array_equal(self.tensor.eigenvectors, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

        # shift = -isotropy in this case (reference = 0)
        self.assertAlmostEqual(self.tensor.shift, -self.tensor.isotropy)
        np.testing.assert_array_equal(self.tensor.shift_tensor.data, -self.tensor.data)

        # update the reference and gradient
        self.tensor.reference = 10.0
        self.tensor.gradient = -2.0
        self.assertAlmostEqual(self.tensor.shift, 10 + (-2.0 * self.tensor.isotropy) / (1 + 10e-6))

    def test_haeberlen_values(self):
        haeb = self.tensor.haeberlen_values
        self.assertAlmostEqual(haeb.sigma_iso, -1.0)
        self.assertAlmostEqual(haeb.sigma, -5.0)
        self.assertAlmostEqual(haeb.delta, -7.5)
        self.assertAlmostEqual(haeb.eta, 0.2)

    def test_herzfeldberger_values(self):
        herz = self.tensor.herzfeldberger_values
        self.assertAlmostEqual(herz.sigma_iso, -1.0)
        self.assertAlmostEqual(herz.omega, 8.0) # span
        self.assertAlmostEqual(herz.kappa, -0.75) # skew

    def test_iupac_values(self):
        iupac = self.tensor.iupac_values
        self.assertAlmostEqual(iupac.sigma_iso, -1.0)
        self.assertAlmostEqual(iupac.sigma_11, -6.0)
        self.assertAlmostEqual(iupac.sigma_22,  1.0)
        self.assertAlmostEqual(iupac.sigma_33,  2.0)

    def test_maryland_values(self):
        mary = self.tensor.maryland_values
        # should be equivalent to the herzfeld-berger values
        herz = self.tensor.herzfeldberger_values
        self.assertAlmostEqual(mary.sigma_iso, herz.sigma_iso)
        self.assertAlmostEqual(mary.omega, herz.omega)
        self.assertAlmostEqual(mary.kappa, herz.kappa)

    def test_mehring_values(self):
        mehr = self.tensor.mehring_values
        # should be equivalent to the iupac values
        iupac = self.tensor.iupac_values
        self.assertAlmostEqual(mehr.sigma_iso, iupac.sigma_iso)
        self.assertAlmostEqual(mehr.sigma_11, iupac.sigma_11)
        self.assertAlmostEqual(mehr.sigma_22, iupac.sigma_22)
        self.assertAlmostEqual(mehr.sigma_33, iupac.sigma_33)

    def test_array_ufunc(self):
        # Test array ufunc operations
        tensor2 = MagneticShielding([np.array([2.0, 3.0, -5.0]), np.eye(3)], species='2H', reference=0)
        result = self.tensor + tensor2
        self.assertIsInstance(result, MagneticShielding)
        np.testing.assert_array_equal(result.data, self.tensor.data + tensor2.data)

    def test_parameter_consistency_warnings(self):
        """Test that warnings are issued when operating on tensors with different parameters."""
        # Create base tensor
        ms1 = MagneticShielding(np.diag([1, 2, 3]), species='13C', reference=100, gradient=-1.0, tag='ms')
        
        # Create tensors with different parameters
        ms2_species = MagneticShielding(np.diag([1, 2, 3]), species='1H', reference=100, gradient=-1.0, tag='ms')
        ms2_reference = MagneticShielding(np.diag([1, 2, 3]), species='13C', reference=50, gradient=-1.0, tag='ms')
        ms2_gradient = MagneticShielding(np.diag([1, 2, 3]), species='13C', reference=100, gradient=-0.5, tag='ms')
        ms2_tag = MagneticShielding(np.diag([1, 2, 3]), species='13C', reference=100, gradient=-1.0, tag='ms_fc')
        ms2_order = MagneticShielding(np.diag([1, 2, 3]), species='13C', reference=100, gradient=-1.0, tag='ms', order='i')
        
        # Test the direct _check_compatible method
        with self.assertRaises(ValueError, msg="Should raise error for different species"):
            MagneticShielding._check_compatible([ms1, ms2_species])
            
        with self.assertRaises(ValueError, msg="Should raise error for different references"):
            MagneticShielding._check_compatible([ms1, ms2_reference])
            
        with self.assertRaises(ValueError, msg="Should raise error for different gradients"):
            MagneticShielding._check_compatible([ms1, ms2_gradient])
            
        with self.assertRaises(ValueError, msg="Should raise error for different tags"):
            MagneticShielding._check_compatible([ms1, ms2_tag])
            
        with self.assertRaises(ValueError, msg="Should raise error for different orders"):
            MagneticShielding._check_compatible([ms1, ms2_order])

class TestElectricFieldGradient(unittest.TestCase):

    def setUp(self):
        # Setup a sample tensor for
        atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.atoms = atoms
        data =atoms.get_array('efg')[0] # First atom is an H

        self.tensor = ElectricFieldGradient(data, species='2H')

    def test_initialization(self):
        # Test if the object is initialized correctly
        self.assertIsInstance(self.tensor, ElectricFieldGradient)
        self.assertIsInstance(self.tensor, NMRTensor)

        # Test if the species and isotope are set correctly
        self.assertEqual(self.tensor.species, '2H')
        self.assertEqual(self.tensor.element, 'H')

    def test_properties(self):
        # Quadruoplar moment
        self.assertAlmostEqual(self.tensor.quadrupole_moment, 2.86, places=2) # in milibarns
        # Because of lack of precision in the quadrupolar moment, the Cq value is imprecise
        # We only ask for 2 decimal places of precision in kHz
        self.assertAlmostEqual(self.tensor.Cq *1e-3, 193809.07262337 * 1e-3, places=2)
        self.assertAlmostEqual(self.tensor.eta, 0.01819691)

        # For 2H, I = 1
        self.assertEqual(self.tensor.spin, 1)
        # nu_Q = 3 * Cq / 2I(2I-1)
        self.assertAlmostEqual(self.tensor.nuq, 3/2 * self.tensor.Cq)
        # For 2H, gamma = 41066279.1 rad/T/s
        self.assertAlmostEqual(self.tensor.gamma, 41066279.1)

        # Quadrupolar product
        method1 = EFGQuadrupolarProduct.get(self.atoms, isotopes={'H': 2})[0]
        method2 = self.tensor.Pq
        manual = self.tensor.Cq * (1 + self.tensor.eta ** 2 / 3)**0.5
        self.assertAlmostEqual(method1, method2)
        self.assertAlmostEqual(method1, manual)

    def test_larmor_frequency(self):
        # Test the larmor frequency method
        Bext = 10.0
        gamma = self.tensor.gamma
        expected_nu_larmor = gamma * Bext / (2 * np.pi) # in Hz
        nu_larmor = self.tensor.get_larmor_frequency(Bext)
        self.assertAlmostEqual(nu_larmor, expected_nu_larmor)

    def test_quadrupolar_perturbation(self):
        # Test the quadrupolar_perturbation method
        Bext = 10.0  # Example external magnetic field in Tesla
        expected_nu_larmor = self.tensor.get_larmor_frequency(Bext)
        expected_spin = self.tensor.spin
        expected_nuq = self.tensor.nuq
        expected_a = (expected_nuq**2 / expected_nu_larmor) * (expected_spin * (expected_spin + 1) - 3/2)

        result = self.tensor.get_quadrupolar_perturbation(Bext)
        self.assertAlmostEqual(result, expected_a)

    def test_array_ufunc(self):
        # Test array ufunc operations
        tensor2 = ElectricFieldGradient([np.array([2.0, 3.0, -5.0]), np.eye(3)], species='2H')
        result = self.tensor + tensor2
        self.assertIsInstance(result, ElectricFieldGradient)
        np.testing.assert_array_equal(result.data, self.tensor.data + tensor2.data)

    def test_parameter_consistency_warnings(self):
        """Test that warnings are issued when operating on tensors with different parameters."""
        # Create base tensor with known parameters
        efg1 = ElectricFieldGradient(np.diag([-1, -1, 2]), species='2H')
        
        # Get the default values to create consistent tensors with only one difference
        quadrupole_moment = efg1.quadrupole_moment
        gamma = efg1.gamma
        
        # Create tensors with different parameters
        efg2_species = ElectricFieldGradient(np.diag([-1, -1, 2]), species='17O', 
                                          quadrupole_moment=quadrupole_moment, gamma=gamma)
        efg2_quadrupole = ElectricFieldGradient(np.diag([-1, -1, 2]), species='2H',
                                             quadrupole_moment=quadrupole_moment*2, gamma=gamma)
        efg2_gamma = ElectricFieldGradient(np.diag([-1, -1, 2]), species='2H',
                                        quadrupole_moment=quadrupole_moment, gamma=gamma*2)
        efg2_order = ElectricFieldGradient(np.diag([-1, -1, 2]), species='2H',
                                        quadrupole_moment=quadrupole_moment, gamma=gamma, order='i')
        
        # Test the direct _check_compatible method
        with self.assertRaises(ValueError, msg="Should raise error for different species"):
            ElectricFieldGradient._check_compatible([efg1, efg2_species])
            
        with self.assertRaises(ValueError, msg="Should raise error for different quadrupole moments"):
            ElectricFieldGradient._check_compatible([efg1, efg2_quadrupole])
            
        with self.assertRaises(ValueError, msg="Should raise error for different gamma values"):
            ElectricFieldGradient._check_compatible([efg1, efg2_gamma])
            
        with self.assertRaises(ValueError, msg="Should raise error for different orders"):
            ElectricFieldGradient._check_compatible([efg1, efg2_order])

class TestMSMeanProperties(unittest.TestCase):
    def setUp(self):
        """Set up a test collection with predictable MS values."""
        # Load the ethanol structure
        self.eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        
        # Create a second structure with scaled MS values
        from soprano.collection import AtomsCollection
        eth2 = self.eth.copy()
        ms_orig = self.eth.get_array("ms").copy()
        # Rotate each ms tensors by 90 degrees
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        ms_new = np.einsum('ij,njk,kl->nil', R, ms_orig, R.T)
        ms_new = ms_new * 2.0  # Double all MS values
        eth2.set_array("ms", ms_new)

        eth_mean = self.eth.copy()
        eth_mean.set_array("ms", np.mean([self.eth.get_array("ms"), eth2.get_array("ms")], axis=0))
        self.eth_mean = eth_mean
        
        # Create collection with both structures
        self.collection = AtomsCollection([self.eth, eth2])
        
        # Calculate properties for both structures for comparison
        self.iso1 = MSShielding.get(self.eth)
        self.iso2 = MSShielding.get(eth2)
        
        self.aniso1 = MSAnisotropy.get(self.eth)
        self.aniso2 = MSAnisotropy.get(eth2)
        
        self.red_aniso1 = MSReducedAnisotropy.get(self.eth)
        self.red_aniso2 = MSReducedAnisotropy.get(eth2)
        
        self.asymm1 = MSAsymmetry.get(self.eth)
        self.asymm2 = MSAsymmetry.get(eth2)
        
        self.span1 = MSSpan.get(self.eth)
        self.span2 = MSSpan.get(eth2)
        
        self.skew1 = MSSkew.get(self.eth)
        self.skew2 = MSSkew.get(eth2)
        
        # Reference settings for testing shift calculations
        self.ref = {'C': 175.0, 'H': 30.0, 'O': 200.0}
        
        self.shift1 = MSShift.get(self.eth, ref=self.ref)
        self.shift2 = MSShift.get(eth2, ref=self.ref)

    def test_shielding_mean(self):
        """Test MSShielding.mean."""
        # Calculate mean shielding manually
        expected_mean = (self.iso1 + self.iso2) / 2
        
        # Use MSShielding.mean
        result_mean = MSShielding().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_anisotropy_mean(self):
        """Test MSAnisotropy.mean."""
        # Use MSAnisotropy.mean
        result_mean = MSAnisotropy().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, MSAnisotropy.get(self.eth_mean)))
        
    def test_reduced_anisotropy_mean(self):
        """Test MSReducedAnisotropy.mean."""
        # Calculate mean reduced anisotropy manually
        expected_mean = MSReducedAnisotropy.get(self.eth_mean)
        # Use MSReducedAnisotropy.mean
        result_mean = MSReducedAnisotropy().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_asymmetry_mean(self):
        """Test MSAsymmetry.mean."""
        # Calculate mean asymmetry manually
        expected_mean = MSAsymmetry.get(self.eth_mean)
        
        # Use MSAsymmetry.mean
        result_mean = MSAsymmetry().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_span_mean(self):
        """Test MSSpan.mean."""
        # Calculate mean span manually
        expected_mean = MSSpan.get(self.eth_mean)
        
        # Use MSSpan.mean
        result_mean = MSSpan().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_skew_mean(self):
        """Test MSSkew.mean."""
        # Calculate mean skew manually
        expected_mean = MSSkew.get(self.eth_mean)
        
        # Use MSSkew.mean
        result_mean = MSSkew().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_shift_mean_with_reference(self):
        """Test MSShift.mean with references."""
        # Calculate mean shift manually
        expected_mean = MSShift.get(self.eth_mean, ref=self.ref)
        
        # Use MSShift.mean
        result_mean = MSShift().mean(self.collection, axis=0, ref=self.ref)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_isotropy_mean_with_reference(self):
        """Test MSIsotropy.mean with references."""
        # Calculate mean shift
        expected_mean = MSIsotropy.get(self.eth_mean, ref=self.ref)
        
        # Use MSIsotropy.mean with reference (should give shift)
        result_mean = MSIsotropy().mean(self.collection, ref=self.ref, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_isotropy_mean_without_reference(self):
        """Test MSIsotropy.mean without references."""
        # Calculate mean shielding
        expected_mean = MSIsotropy.get(self.eth_mean)
        
        # Use MSIsotropy.mean without reference (should give shielding)
        result_mean = MSIsotropy().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_mean, expected_mean))
        
    def test_weighted_mean(self):
        """Test weighted mean calculations."""
        weights = [0.25, 0.75]  # 25% first structure, 75% second
        
        # Calculate weighted mean shielding manually
        expected_weighted_mean = weights[0] * self.iso1 + weights[1] * self.iso2
        
        # Use MSShielding.mean with weights
        result_weighted_mean = MSShielding().mean(self.collection, weights=weights, axis=0)
        
        self.assertTrue(np.allclose(result_weighted_mean, expected_weighted_mean))
        
    def test_euler_mean(self):
        """Test MSEuler.mean."""
        # Get tensors from both structures
        tensors1 = MSTensor.get(self.eth)
        tensors2 = MSTensor.get(self.collection[1])[0]
        
        # Calculate mean tensors manually
        mean_tensors = []
        for t1, t2 in zip(tensors1, tensors2):
            mean_data = (t1.data + t2.data) / 2
            mean_tensors.append(MagneticShielding(mean_data, species=t1.species, order=t1.order))
        
        # Get Euler angles from mean tensors
        expected_eulers = np.array([t.euler_angles() for t in mean_tensors])
        
        # Use MSEuler.mean
        result_eulers = MSEuler().mean(self.collection, axis=0)
        
        self.assertTrue(np.allclose(result_eulers, expected_eulers))

class TestEFGMeanProperties(unittest.TestCase):
    def setUp(self):
        """Set up a test collection with predictable EFG values."""
        # Load the ethanol structure
        self.eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        
        # Create a second structure with scaled EFG values
        from soprano.collection import AtomsCollection
        eth2 = self.eth.copy()
        efg_orig = self.eth.get_array("efg").copy()
        # "Rotate the EFG tensor by 90 deg"
        R = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        # efg_new = np.dot(R, np.dot(efg_orig, R.T))
        efg_new = np.einsum('ij,njk,kl->nil', R, efg_orig, R.T)
        eth2.set_array("efg", efg_new)
        self.eth2 = eth2

        eth_mean = self.eth.copy()
        # Set the EFG to the mean of the two structures
        efg_mean = (efg_orig + efg_new) / 2
        eth_mean.set_array("efg", efg_mean)
        self.eth_mean = eth_mean
        
        # Create collection with both structures
        self.collection = AtomsCollection([self.eth, self.eth2])
        
        # Calculate mean properties
        self.vzz_mean = EFGVzz.get(self.eth_mean)
        self.aniso_mean = EFGAnisotropy.get(self.eth_mean)
        self.red_aniso_mean = EFGReducedAnisotropy.get(self.eth_mean)
        self.asymm_mean = EFGAsymmetry.get(self.eth_mean)
        self.span_mean = EFGSpan.get(self.eth_mean)
        self.skew_mean = EFGSkew.get(self.eth_mean)
        self.euler_mean = EFGEuler.get(self.eth_mean)
        self.quat_mean = EFGQuaternion.get(self.eth_mean)
        
        # Isotope settings for testing quadrupolar properties
        self.isotopes = {'H': 2, 'O': 17, 'C': 13}
        self.qconst_mean = EFGQuadrupolarConstant.get(self.eth_mean, isotopes=self.isotopes)
        self.qprod_mean = EFGQuadrupolarProduct.get(self.eth_mean, isotopes=self.isotopes)
        self.nqr_mean = EFGNQR.get(self.eth_mean, isotopes=self.isotopes)

    def test_vzz_mean(self):
        """Test EFGVzz.mean."""
        # Use EFGVzz.mean
        result_mean = EFGVzz().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, self.vzz_mean))
        
    def test_anisotropy_mean(self):
        """Test EFGAnisotropy.mean."""
        
        # Use EFGAnisotropy.mean
        result_mean = EFGAnisotropy().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, self.aniso_mean))
        
    def test_reduced_anisotropy_mean(self):
        """Test EFGReducedAnisotropy.mean."""
        # Use EFGReducedAnisotropy.mean
        result_mean = EFGReducedAnisotropy().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, self.red_aniso_mean))
        
    def test_asymmetry_mean(self):
        """Test EFGAsymmetry.mean."""
        # Use EFGAsymmetry.mean
        result_mean = EFGAsymmetry().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, self.asymm_mean))
        
    def test_span_mean(self):
        """Test EFGSpan.mean."""
        # Use EFGSpan.mean
        result_mean = EFGSpan().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, self.span_mean))
        
    def test_skew_mean(self):
        """Test EFGSkew.mean."""
        # Use EFGSkew.mean
        result_mean = EFGSkew().mean(self.collection, axis=0)
        self.assertTrue(np.allclose(result_mean, self.skew_mean))
        
    def test_quadrupolar_constant_mean(self):
        """Test EFGQuadrupolarConstant.mean."""
        # Use EFGQuadrupolarConstant.mean with isotopes
        result_mean = EFGQuadrupolarConstant().mean(self.collection, isotopes=self.isotopes, axis=0)
        self.assertTrue(np.allclose(result_mean, self.qconst_mean))
        
    def test_quadrupolar_product_mean(self):
        """Test EFGQuadrupolarProduct.mean."""
        # Use EFGQuadrupolarProduct.mean with isotopes
        result_mean = EFGQuadrupolarProduct().mean(self.collection, isotopes=self.isotopes, axis=0)
        self.assertTrue(np.allclose(result_mean, self.qprod_mean))

    def test_nqr_mean(self):
        """Test EFGNQR.mean."""
        nqr_mean = self.nqr_mean
        # Use EFGNQR.mean with isotopes
        result_mean = EFGNQR().mean(self.collection, isotopes=self.isotopes, axis=0)

        
        # Check each atom's NQR frequencies
        for i in range(len(self.eth)):
            # If there are no transitions (not quadrupolar), both should be empty
            if not nqr_mean[i]:
                self.assertEqual(len(result_mean[i]), 0)
                continue
                
            # For atoms with transitions, compare the frequencies
            for key in nqr_mean[i]:
                # Check if key exists in both original and mean result
                self.assertIn(key, result_mean[i])
                # Calculate expected mean frequency for this transition
                expected_freq = nqr_mean[i][key]
                # Check if mean frequency matches expected
                self.assertAlmostEqual(result_mean[i][key], expected_freq)
                
    def test_euler_mean(self):
        """Test EFGEuler.mean."""
        # Use EFGEuler.mean
        result_eulers = EFGEuler().mean(self.collection, isotopes=self.isotopes, axis=0)
        self.assertTrue(np.allclose(result_eulers, self.euler_mean))
        
    def test_quaternion_mean(self):
        """Test EFGQuaternion.mean."""
        # Use EFGQuaternion.mean
        result_quats = EFGQuaternion().mean(self.collection, isotopes=self.isotopes, axis=0)
        
        # Compare quaternions - check q or -q since they represent the same rotation
        for q_exp, q_res in zip(self.quat_mean, result_quats):
            # Quaternions q and -q represent the same rotation
            self.assertTrue(np.allclose(q_exp.q, q_res.q) or np.allclose(q_exp.q, -q_res.q))

    def test_weighted_mean(self):
        """Test weighted mean calculations."""
        weights = [0.25, 0.75]  # 25% first structure, 75% second
        # New structure with the efg tensors weighted by the weights
        atoms1 = self.eth.copy()
        atoms2 = self.eth2.copy()

        # weighted EFG tensors
        efg1 = atoms1.get_array("efg")
        efg2 = atoms2.get_array("efg")
        # Set the EFG to the weighted mean of the two structures
        efg_weighted = (weights[0] * efg1 + weights[1] * efg2) / sum(weights)
        atomsout = atoms1.copy()
        atomsout.set_array("efg", efg_weighted)
        expected_weighted_mean = EFGVzz.get(atomsout)
        
        # Use EFGVzz.mean with weights
        result_weighted_mean = EFGVzz().mean(self.collection, weights=weights, axis=0)
        
        self.assertTrue(np.allclose(result_weighted_mean, expected_weighted_mean))

if __name__ == "__main__":
    unittest.main()
