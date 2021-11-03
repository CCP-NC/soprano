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
import unittest
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestPropertyLoad(unittest.TestCase):
    def test_dummyprop(self):

        from soprano.properties import AtomsProperty
        from soprano.collection import AtomsCollection

        # Define a dummy derived property and test it

        class DummyProperty(AtomsProperty):

            default_name = "dummy"
            default_params = {"mul": 2.0}

            @staticmethod
            def extract(s, mul):
                return s.positions.shape[0] * mul

        # Now two atoms objects to test it on
        a1 = Atoms("C")
        a2 = Atoms("CC")

        dummyDoubled = DummyProperty(name="doubledummy", mul=4)

        self.assertEqual(DummyProperty.get(a1), 2.0)
        self.assertEqual(dummyDoubled(a2), 8.0)
        self.assertEqual(DummyProperty.get(a1, mul=3), 3.0)

        # Also check that improper parameters are rejected
        self.assertRaises(ValueError, DummyProperty, wrong="this is wrong")

        # Test behaviour on a collection instead
        c1 = AtomsCollection([a1, a2])
        c2 = AtomsCollection([a2, a1])

        DummyProperty.get(c1, store_array=True)
        dummyDoubled(c2, store_array=True)
        self.assertTrue(np.all(c1.get_array("dummy") == [2, 4]))
        self.assertTrue(np.all(c2.get_array("doubledummy") == [8, 4]))

        # And with additional arguments...
        self.assertTrue(np.all(DummyProperty.get(c1, mul=3) == [3, 6]))

    def test_basicprop(self):

        from soprano.utils import cart2abc
        from soprano.properties.basic import LatticeCart, LatticeABC

        cell = np.array([[1, 0, 0], [0, 2, 0], [0, 0.5, 3.0]])
        a = Atoms(cell=cell)
        linLatt = LatticeCart(shape=(9,))
        degLatt = LatticeABC(deg=True)

        ans = np.array(cell).reshape((9,))
        self.assertTrue(np.all(linLatt(a) == ans))
        ans = cart2abc(cell)
        ans[1, :] *= 180.0 / np.pi
        self.assertTrue(np.all(degLatt(a) == ans))

    def test_propertymap(self):

        from soprano.collection import AtomsCollection
        from soprano.properties.basic import NumAtoms

        num_atoms = np.random.randint(1, 11, size=10)
        coll = AtomsCollection([Atoms("H" * n) for n in num_atoms])

        num_atoms_prop = coll.all.map(NumAtoms.get)

        self.assertTrue((num_atoms == num_atoms_prop).all)

    def test_remap(self):

        from soprano.properties.map import RemapIndices, Remap

        # Create a test reference structure
        s1 = bulk('Au', cubic=True)
        shuffle = [3,0,2,1]
        ref = s1[shuffle]

        # Rattle the atoms a bit
        rng = np.random.default_rng(0)
        s1.positions += (rng.random((4,3))-0.5)/10.0

        indices = RemapIndices.extract(s1, ref, True, False)
        self.assertTrue(all([i == j for (i, j) in zip(shuffle, indices)]))

        # Test using scaled coordinates instead

        indices = RemapIndices.extract(s1, ref, True, True)
        self.assertTrue(all([i == j for (i, j) in zip(shuffle, indices)]))

        # Now just the remapping
        s2 = Remap.extract(s1, ref, True, False)

        self.assertTrue((s2.positions == s1.positions[shuffle]).all())

    def test_linkageprops(self):

        from soprano.properties.linkage import (
            Bonds,
            Molecules,
            MoleculeNumber,
            MoleculeMass,
            MoleculeCOMLinkage,
            MoleculeSpectralSort,
            CoordinationHistogram,
            HydrogenBonds,
            HydrogenBondsNumber,
        )

        a = read(os.path.join(_TESTDATA_DIR, "mol_crystal.cif"))

        # Test bonds
        testAtoms = Atoms(
            ["C", "C", "C", "C"],
            cell=[5, 5, 5],
            positions=np.array([[0, 0, 0], [4, 0, 0], [3, 3, 3], [3, 3.5, 3]]),
            pbc=True,
        )
        testBonds = Bonds.get(testAtoms)
        self.assertTrue(testBonds[0][:2] == (0, 1))
        self.assertTrue(testBonds[1][:2] == (2, 3))
        self.assertTrue(np.all(testBonds[0][2] == (-1, 0, 0)))
        self.assertAlmostEqual(testBonds[0][3], 2 * testBonds[1][3])

        # Also test coordination histogram
        coord_hist = CoordinationHistogram.get(a)
        # Testing some qualities of the Alanine crystal...
        self.assertTrue(coord_hist["H"]["C"][1], 16)  # 16 H bonded to a C
        self.assertTrue(coord_hist["H"]["N"][1], 12)  # 12 H bonded to a N
        self.assertTrue(coord_hist["C"]["H"][3], 4)  # 4 CH3 groups
        self.assertTrue(coord_hist["C"]["O"][2], 4)  # 4 COO groups

        # Test molecules
        mols = Molecules.get(a)

        self.assertTrue(MoleculeNumber.get(a) == 4)
        self.assertTrue(np.isclose(MoleculeMass.get(a), 89.09408).all())
        self.assertTrue(len(MoleculeCOMLinkage.get(a)) == 6)

        # Spectral sorting
        elems = np.array(a.get_chemical_symbols())
        mol_specsort = MoleculeSpectralSort.get(a)
        for i in range(len(mols) - 1):
            for j in range(i + 1, len(mols)):
                self.assertTrue(
                    (
                        elems[mol_specsort[i].indices] == elems[mol_specsort[j].indices]
                    ).all()
                )

        # Now testing hydrogen bonds
        hbs = HydrogenBonds.get(a)
        hbn = HydrogenBondsNumber.get(a)

        # Test the presence of one specific hydrogen bond
        self.assertEqual(len(hbs["NH..O"]), 12)
        self.assertAlmostEqual(hbs["NH..O"][0]["length"], 2.82, 2)

        self.assertTrue(hbn["NH..O"] == 12)
        self.assertTrue(hbn["OH..O"] == 0)

    def test_labelprops(self):

        from soprano.properties.labeling import MoleculeSites, HydrogenBondTypes

        a = read(os.path.join(_TESTDATA_DIR, "nh3.cif"))

        nh3_sites = MoleculeSites.get(a)[0]

        # Check the name
        self.assertEqual(nh3_sites["name"], "H[N[H,H]]")
        # Check the sites
        self.assertEqual(set(nh3_sites["sites"].values()), set(["N_1", "H_1"]))

        # Now we test hydrogen bond types with alanine
        a = read(os.path.join(_TESTDATA_DIR, "mol_crystal.cif"))
        # We expect 12 identical ones
        hbtypes = [
            "C[C[C[H,H,H],H,N[H,H,H]],O,O]<N_1,H_2>"
            "..C[C[C[H,H,H],H,N[H,H,H]],O,O]<O_1>"
        ] * 12

        self.assertEqual(HydrogenBondTypes.get(a), hbtypes)

    def test_transformprops(self):

        from ase.quaternions import Quaternion
        from soprano.selection import AtomSelection
        from soprano.properties.transform import Translate, Rotate, Mirror

        a = Atoms("CH", positions=[[0, 0, 0], [0.5, 0, 0]])

        sel = AtomSelection.from_element(a, "C")
        transl = Translate(selection=sel, vector=[0.5, 0, 0])
        rot = Rotate(
            selection=sel,
            center=[0.25, 0.0, 0.25],
            quaternion=Quaternion([np.cos(np.pi / 4.0), 0, np.sin(np.pi / 4.0), 0]),
        )
        mirr = Mirror(selection=sel, plane=[1, 0, 0, -0.25])

        aT = transl(a)
        aR = rot(a)
        aM = mirr(a)

        self.assertAlmostEqual(
            np.linalg.norm(aT.get_positions()[0]), np.linalg.norm(aT.get_positions()[1])
        )
        self.assertAlmostEqual(
            np.linalg.norm(aR.get_positions()[0]), np.linalg.norm(aR.get_positions()[1])
        )
        self.assertAlmostEqual(
            np.linalg.norm(aM.get_positions()[0]), np.linalg.norm(aM.get_positions()[1])
        )


if __name__ == "__main__":
    unittest.main()
