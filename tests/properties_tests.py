#!/usr/bin/env python
"""
Test code for the AtomsProperty class and its children
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
from ase import Atoms
from ase.io import read, write
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
from soprano.properties import AtomsProperty
import unittest
import numpy as np

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")


class TestPropertyLoad(unittest.TestCase):

    def test_dummyprop(self):

        # Define a dummy derived property and test it

        class DummyProperty(AtomsProperty):

            default_name = 'dummy'
            default_params = {'mul': 2.0}

            @staticmethod
            def extract(s, mul):
                return s.positions.shape[0]*mul

        # Now two atoms objects to test it on
        a1 = Atoms('C')
        a2 = Atoms('CC')

        dummyDoubled = DummyProperty(name='doubledummy', mul=4)

        self.assertEqual(DummyProperty.get(a1), 2.0)
        self.assertEqual(dummyDoubled(a2), 8.0)

        # Also check that improper parameters are rejected
        self.assertRaises(ValueError, DummyProperty, wrong='this is wrong')

        # Test behaviour on a collection instead
        c1 = AtomsCollection([a1, a2])
        c2 = AtomsCollection([a2, a1])

        DummyProperty.get(c1, store_array=True)
        dummyDoubled(c2, store_array=True)
        self.assertTrue(np.all(c1.get_array('dummy') == [2, 4]))
        self.assertTrue(np.all(c2.get_array('doubledummy') == [8, 4]))

    def test_basicprop(self):

        from soprano.utils import cart2abc
        from soprano.properties.basic import (LatticeCart, LatticeABC,
                                              CalcEnergy)

        cell = np.array([[1, 0, 0], [0, 2, 0], [0, 0.5, 3.0]])
        a = Atoms(cell=cell)
        linLatt = LatticeCart(shape=(9,))
        degLatt = LatticeABC(deg=True)

        ans = cell.reshape((9,))
        self.assertTrue(np.all(linLatt(a) == ans))
        ans = cart2abc(cell)
        ans[1, :] *= 180.0/np.pi
        self.assertTrue(np.all(degLatt(a) == ans))

    def test_propertymap(self):

        from soprano.properties.basic import NumAtoms

        num_atoms = np.random.random_integers(1, 10, 10)
        coll = AtomsCollection([Atoms('H'*n) for n in num_atoms])

        num_atoms_prop = coll.all.map(NumAtoms.get)

        self.assertTrue((num_atoms == num_atoms_prop).all)

    def test_linkageprops(self):

        from soprano.properties.linkage import (LinkageList, Bonds,
                                                Molecules, MoleculeNumber,
                                                MoleculeMass,
                                                MoleculeCOMLinkage,
                                                MoleculeRelativeRotation,
                                                HydrogenBonds,
                                                HydrogenBondsNumber)

        from soprano.properties.transform import Rotate

        a = read(os.path.join(_TESTDATA_DIR, 'mol_crystal.cif'))

        # Test bonds
        testA = Atoms(['C', 'C', 'C', 'C'],
                      cell=[5,5,5],
                      positions=np.array([[0,0,0],[4.9,0,0],[3,3,3],[3,4,3]]))

        # Test molecules
        mols = Molecules.get(a)

        self.assertTrue(MoleculeNumber.get(a) == 4)
        self.assertTrue(np.isclose(MoleculeMass.get(a), 89.09408).all())
        self.assertTrue(len(MoleculeCOMLinkage.get(a)) == 6)

        # Now testing hydrogen bonds
        hbs = HydrogenBonds.get(a)
        hbn = HydrogenBondsNumber.get(a)

        self.assertTrue(hbn['NH..O'] == 12)
        self.assertTrue(hbn['OH..O'] == 0)

    def test_labelprops(self):

        from soprano.properties.labeling import (MoleculeSites,
                                                 HydrogenBondTypes)

        a = read(os.path.join(_TESTDATA_DIR, 'nh3.cif'))

        nh3_sites = MoleculeSites.get(a)[0]

        # Check the name
        self.assertEqual(nh3_sites['name'], 'H[N[H,H]]')
        # Check the sites
        self.assertEqual(set(nh3_sites['sites'].values()),
                         set(['N_1', 'H_1']))

        # Now we test hydrogen bond types with alanine
        a = read(os.path.join(_TESTDATA_DIR, 'mol_crystal.cif'))
        # We expect 12 identical ones
        hbtypes = ['C[C[C[H,H,H],H,N[H,H,H]],O,O]<N_1,H_3>'
                   '..C[C[C[H,H,H],H,N[H,H,H]],O,O]<O_1>']*12

        self.assertEqual(HydrogenBondTypes.get(a), hbtypes)

    def test_transformprops(self):

        from ase.quaternions import Quaternion
        from soprano.selection import AtomSelection
        from soprano.properties.transform import (Translate, Rotate, Mirror)

        a = Atoms('CH', positions=[[0, 0, 0], [0.5, 0, 0]])

        sel = AtomSelection.from_element(a, 'C')
        transl = Translate(selection=sel, vector=[0.5, 0, 0])
        rot = Rotate(selection=sel, center=[0.25, 0.0, 0.25],
                     quaternion=Quaternion([np.cos(np.pi/4.0),
                                            0,
                                            np.sin(np.pi/4.0),
                                            0]))
        mirr = Mirror(selection=sel, plane=[1, 0, 0, -0.25])

        aT = transl(a)
        aR = rot(a)
        aM = mirr(a)

        self.assertAlmostEqual(np.linalg.norm(aT.get_positions()[0]),
                               np.linalg.norm(aT.get_positions()[1]))
        self.assertAlmostEqual(np.linalg.norm(aR.get_positions()[0]),
                               np.linalg.norm(aR.get_positions()[1]))
        self.assertAlmostEqual(np.linalg.norm(aM.get_positions()[0]),
                               np.linalg.norm(aM.get_positions()[1]))

if __name__ == '__main__':
    unittest.main()
