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
    EFGQuaternion,
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
                skew[i], 3 * (sorted(vals[4:7])[1] - iso[i]) / span[i]
            )

    def test_efg(self):

        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

        # Load the data calculated with MagresView
        with open(os.path.join(_TESTDATA_DIR, "ethanol_efg.dat")) as f:
            data = f.readlines()[8:]

        asymm = EFGAsymmetry.get(eth)

        qprop = EFGQuadrupolarConstant(isotopes={"H": 2})
        qcnst = qprop(eth)
        quats = EFGQuaternion.get(eth)

        for i, d in enumerate(data):
            vals = [float(x) for x in d.split()[1:]]
            if len(vals) != 8:
                continue
            # And check...
            # The quadrupolar constant has some imprecision due to values
            # of quadrupole moment, so we only ask 2 places in kHz
            self.assertAlmostEqual(qcnst[i] * 1e-3, vals[0] * 1e-3, places=2)
            self.assertAlmostEqual(asymm[i], vals[1])
            vq = Quaternion.from_euler_angles(*(np.array(vals[-3:]) * np.pi / 180.0))
            # Product to see if they go back to the origin
            # The datafile contains conjugate quaternions
            pq = vq * quats[i]
            cosphi = np.clip(pq.q[0], -1, 1)
            phi = np.arccos(cosphi)
            # 180 degrees rotations are possible (signs are not fixed)
            self.assertTrue(
                np.isclose((phi * 2) % np.pi, 0) or np.isclose((phi * 2) % np.pi, np.pi)
            )

        # A more basic test for the Vzz
        Vzz_p = EFGVzz.get(eth)
        Vzz_raw = []
        for efg in eth.get_array("efg"):
            evals, _ = np.linalg.eigh(efg)
            Vzz_raw.append(evals[np.argmax(abs(evals))])

        self.assertTrue(np.isclose(Vzz_p, Vzz_raw).all())

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
            evalstt = np.linalg.eigh(diptens[ij])[0]
            self.assertTrue(np.isclose(np.sort(evals), np.sort(evalstt)).all())
            self.assertAlmostEqual(np.linalg.multi_dot([v, diptens[ij], v]), 2 * d)

    def test_tensor_basic(self):

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
        # in this case the skew is 0.75 = 3 * (1 - (-1)) / (2 - (-6))
        self.assertTrue(np.allclose([tc.skew, td.skew, th.skew, tn.skew], [0.75]))

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
        zyza = tc.equivalent_euler_angles(convention='ZYZ', passive = False) * 180 / np.pi
        zxza = tc.equivalent_euler_angles(convention='zxz', passive = False) * 180 / np.pi
        zyzp = tc.equivalent_euler_angles(convention='zyz', passive = True) * 180 / np.pi
        zxzp = tc.equivalent_euler_angles(convention='zxz', passive = True) * 180 / np.pi
        self.assertTrue(np.allclose(zyza[0], np.array([135, 0, 0])))
        self.assertTrue(np.allclose(zxza[0], np.array([135, 0, 0]))) # this  one used to work, but now give [315, 0, 0]
        self.assertTrue(np.allclose(zyzp[0], np.array([0, 0, 225])))
        self.assertTrue(np.allclose(zxzp[0], np.array([0, 0, 225])))


        # More symmetric tensors
        data = np.diag([5,10,5])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        self.assertTrue(np.allclose(tc.euler_angles(convention='zyz')*180/np.pi, np.array([90,90,0])))
        # TODO: this is not the same as TensorView for MATLAB, which gives [0,90,90]. soprano gives [90,90,0] - so something is 
        # not happening correctly when passive is True - is it just for this case?
        # self.assertTrue(np.allclose(tc.euler_angles(convention='zyz', passive=True)*180/np.pi, np.array([0,90,90])))

        data = np.diag([10,5,5])
        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        self.assertTrue(np.allclose(tc.euler_angles(convention='zyz')*180/np.pi, np.array([180,90,0])))
        # TODO: according to TensorView for MATLAB, this should be [0,90,0] or equivalent
        # soprano gives [90, 90, 0] - so something is not happening correctly when passive is True
        f = tc.euler_angles(convention='zyz', passive=True)*180/np.pi
        self.assertTrue(np.allclose(tc.euler_angles(convention='zyz', passive=True)*180/np.pi, np.array([0,90,0])))

        # TODO: add more tests for zxz conventions and passive rotation


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

        rot_dip = [D[(0, 1)] for D in DipolarTensor.get(rot_eth, sel_i=[0], sel_j=[1])]

        dip_avg_num = np.average(rot_dip, axis=0)

        dip_tens = NMRTensor.make_dipolar(eth, 0, 1, rotation_axis=axis)
        dip_avg_tens = dip_tens.data

        self.assertTrue(np.allclose(dip_avg_num, dip_avg_tens))

        # Test eigenvectors
        evecs = dip_tens.eigenvectors
        self.assertTrue(np.allclose(np.dot(evecs.T, evecs), np.eye(3)))


if __name__ == "__main__":
    unittest.main()
