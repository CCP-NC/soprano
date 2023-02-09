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
    EFGNQR,
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

        # A basic test for NQR frequencies
        NQR = EFGNQR.get(eth)
        non_zero_NQRs = np.where(NQR)[0]
        # Only the O has NQR & there's only one O atom
        self.assertTrue(len(non_zero_NQRs) == 1)
        self.assertTrue(eth[non_zero_NQRs[0]].symbol == "O")
        NQR_vals = [v for v in NQR[-1].values()]
        NQR_keys = [k for k in NQR[-1].keys()]
        # the first is 0.5 -> 1.5
        self.assertTrue(NQR_keys[0] == 'm=0.5->1.5')
        # the first is 1.5 -> 2.5
        self.assertTrue(NQR_keys[1] == 'm=1.5->2.5')
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
            evalstt = np.linalg.eigh(diptens[ij])[0]
            self.assertTrue(np.isclose(np.sort(evals), np.sort(evalstt)).all())
            self.assertAlmostEqual(np.linalg.multi_dot([v, diptens[ij], v]), 2 * d)

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

        # Let's now try various conventions
        data = np.diag([1, 2, -6])

        tc = NMRTensor(data, NMRTensor.ORDER_INCREASING)
        self.assertTrue(np.allclose(tc.eigenvalues, [-6, 1, 2]))

        td = NMRTensor(data, NMRTensor.ORDER_DECREASING)
        self.assertTrue(np.allclose(td.eigenvalues, [2, 1, -6]))

        th = NMRTensor(data, NMRTensor.ORDER_HAEBERLEN)
        self.assertTrue(np.allclose(th.eigenvalues, [2, 1, -6]))

        tn = NMRTensor(data, NMRTensor.ORDER_NQR)
        self.assertTrue(np.allclose(tn.eigenvalues, [1, 2, -6]))

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
