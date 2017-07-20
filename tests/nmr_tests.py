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
from soprano.properties.nmr import (MSIsotropy, MSAnisotropy,
                                    MSReducedAnisotropy, MSAsymmetry,
                                    MSSpan, MSSkew,
                                    EFGVzz, EFGAsymmetry,
                                    EFGQuadrupolarConstant,
                                    EFGQuaternion, DipolarCoupling)
from soprano.selection import AtomSelection

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")


class TestNMR(unittest.TestCase):

    def test_shielding(self):

        eth = io.read(os.path.join(_TESTDATA_DIR, 'ethanol.magres'))

        # Load the data calculated with MagresView
        data = open(os.path.join(_TESTDATA_DIR,
                                 'ethanol_ms.dat')).readlines()[8:]

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
            self.assertAlmostEqual(span[i], max(vals[4:7])-min(vals[4:7]))
            self.assertAlmostEqual(skew[i],
                                   3*(sorted(vals[4:7])[1]-iso[i])/span[i])

    def test_efg(self):

        eth = io.read(os.path.join(_TESTDATA_DIR, 'ethanol.magres'))

        # Load the data calculated with MagresView
        data = open(os.path.join(_TESTDATA_DIR,
                                 'ethanol_efg.dat')).readlines()[8:]

        asymm = EFGAsymmetry.get(eth)

        qprop = EFGQuadrupolarConstant(isotopes={'H': 2})
        qcnst = qprop(eth)
        quats = EFGQuaternion.get(eth)

        for i, d in enumerate(data):
            vals = [float(x) for x in d.split()[1:]]
            if len(vals) != 8:
                continue
            # And check...
            # The quadrupolar constant has some imprecision due to values
            # of quadrupole moment, so we only ask 2 places in kHz
            self.assertAlmostEqual(qcnst[i]*1e-3, vals[0]*1e-3, places=2)
            self.assertAlmostEqual(asymm[i], vals[1])
            vq = Quaternion.from_euler_angles(*(np.array(vals[-3:]
                                                         )*np.pi/180.0))
            # Product to see if they go back to the origin
            # The datafile contains conjugate quaternions
            pq = vq*quats[i]
            cosphi = np.clip(pq.q[0], -1, 1)
            phi = np.arccos(cosphi)
            # 180 degrees rotations are possible (signs are not fixed)
            self.assertTrue(np.isclose((phi*2) % np.pi, 0) or
                            np.isclose((phi*2) % np.pi, np.pi))

    def test_dipolar(self):

        eth = io.read(os.path.join(_TESTDATA_DIR, 'ethanol.magres'))

        # Load the data calculated with MagresView
        data = open(os.path.join(_TESTDATA_DIR,
                                 'ethanol_dip.dat')).readlines()[8:]

        dip = DipolarCoupling.get(eth)

        # Magres labels
        symbs = np.array(eth.get_chemical_symbols())
        elems = set(symbs)
        mlabs = ['']*len(eth)
        for e in elems:
            e_i = np.where(symbs == e)[0]
            for i, j in enumerate(e_i):
                mlabs[j] = '{0}_{1}'.format(e, i+1)

        data_dip = {}
        for l in data:
            lab1, lab2, d, a, b = l.split()
            i1 = mlabs.index(lab1)
            i2 = mlabs.index(lab2)
            i1, i2 = (i1, i2) if i1 < i2 else (i2, i1)
            data_dip[(i1, i2)] = [float(x) for x in (d, a, b)]

        for ij, (d, v) in dip.iteritems():
            # The precision is rather low, probably due to the gammas
            self.assertAlmostEqual(d*2*np.pi, data_dip[ij][0], places=-3)
            a, b = np.array([np.arccos(-v[2]),
                             np.arctan2(-v[1], -v[0])]) % (2*np.pi)
            ba_dat = (np.array(data_dip[ij][1:])*np.pi/180.0) % (2*np.pi)
            self.assertTrue(np.isclose([b, a], ba_dat, atol=0.1).all())

if __name__ == '__main__':
    unittest.main()
