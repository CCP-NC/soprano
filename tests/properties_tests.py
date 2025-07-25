#!/usr/bin/env python
"""
Test code for the AtomsProperty class and its children
"""


import os
import sys
import unittest

import numpy as np
from ase import Atom, Atoms
from ase.build import bulk
from ase.io import read

from tests.test_utils import skip_if_problematic_ase

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestPropertyLoad(unittest.TestCase):
    def test_dummyprop(self):
        from soprano.collection import AtomsCollection
        from soprano.properties import AtomsProperty

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

    def test_mean_property(self):
        from soprano.collection import AtomsCollection
        from soprano.properties import AtomsProperty

        class DummyProperty(AtomsProperty):
            default_name = "dummy"
            default_params = {"mul": 2.0}

            @staticmethod
            def extract(s, mul):
                return s.positions.shape[0] * mul

        # Now two atoms objects to test it on
        a1 = Atoms("C")
        a2 = Atoms("CC")
        a3 = Atoms("CCC")

        c1 = AtomsCollection([a1, a2, a3])

        dummy_prop = DummyProperty()

        # Test mean of scalar properties
        mean_value = dummy_prop.mean(c1)
        self.assertEqual(mean_value, 4.0)

        # Test mean of array properties
        class ArrayProperty(AtomsProperty):
            default_name = "array_prop"
            default_params = {}

            @staticmethod
            def extract(s):
                return np.array([s.positions.shape[0], s.positions.shape[0] * 2])

        array_prop = ArrayProperty()
        mean_array = array_prop.mean(c1, axis=0)
        np.testing.assert_array_equal(mean_array, np.array([2.0, 4.0]))

        # Test mean of dictionary properties
        class DictProperty(AtomsProperty):
            default_name = "dict_prop"
            default_params = {}

            @staticmethod
            def extract(s):
                return {"count": s.positions.shape[0], "double_count": s.positions.shape[0] * 2}

        dict_prop = DictProperty()
        mean_dict = dict_prop.mean(c1)
        self.assertEqual(mean_dict["count"], 2.0)
        self.assertEqual(mean_dict["double_count"], 4.0)

    def test_basicprop(self):
        from soprano.properties.basic import LatticeABC, LatticeCart
        from soprano.utils import cart2abc

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

        num_atoms = np.array([3, 5, 4, 1, 6, 10, 12, 2, 7, 9])
        coll = AtomsCollection([Atoms("H" * n) for n in num_atoms])

        num_atoms_prop = coll.all.map(NumAtoms.get)

        self.assertTrue((num_atoms == num_atoms_prop).all)

    def test_property_selection(self):
        from soprano.properties import AtomsProperty
        from soprano.selection import AtomSelection

        # Create a test property that returns the number of atoms of a specific element
        class ElementCountProperty(AtomsProperty):
            default_name = "elem_count"
            default_params = {"element": "H"}

            @staticmethod
            def extract(s, element):
                return sum(1 for atom in s if atom.symbol == element)

        # Create a test structure with mixed atoms
        a = Atoms('CH2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Create the property
        elem_count = ElementCountProperty()
        
        # Test without selection (all atoms)
        count_all = elem_count(a)
        self.assertEqual(count_all, 2)  # There are 2 H atoms
        
        # Test with selection string
        count_string = ElementCountProperty.get(a, selection="C")
        self.assertEqual(count_string, 0)  # No H atoms in the C-only selection
        
        # Test with AtomSelection object
        sel = AtomSelection.from_element(a, "C")
        count_sel = ElementCountProperty.get(a, selection=sel)
        self.assertEqual(count_sel, 0)  # No H atoms in the C-only selection
        
        # Test with O selection
        count_o = ElementCountProperty.get(a, selection="O")
        self.assertEqual(count_o, 0)  # No H atoms in the O-only selection
        
        # Test with H selection
        count_h = ElementCountProperty.get(a, selection="H")
        self.assertEqual(count_h, 2)  # 2 H atoms in the H-only selection

        # Test with H.1 selection
        count_h = ElementCountProperty.get(a, selection="H.1")
        self.assertEqual(count_h, 1)  # 1 H atoms in the H.1 selection

        # Test using instance call with selection
        h_sel = AtomSelection.from_element(a, "H")
        count_h_instance = elem_count(a, selection=h_sel)
        self.assertEqual(count_h_instance, 2)
        
        # Test property with parameters and selection
        custom_count = ElementCountProperty.get(a, selection="H", element="O")
        self.assertEqual(custom_count, 0)  # No O atoms among the H atoms

    def test_remap(self):
        from soprano.properties.map import Remap, RemapIndices

        # Create a test reference structure
        ref = bulk("Au", cubic=True)
        # label the 4 atoms a,b,c,d
        ref.set_array("ref_labels", np.array(["a", "b", "c", "d"]))
        shuffle = [3, 0, 2, 1]
        unshuffle = [1, 3, 2, 0]
        # shuffle the order and store as s1 atoms object
        s1 = ref[shuffle]

        # Rattle the atoms a bit
        rng = np.random.default_rng(0)
        s1.positions += (rng.random((4, 3)) - 0.5) / 10.0

        # Test naive remapping
        indices = RemapIndices.extract(
            s1, ref, mic=False, check_species=True, tolerance=1.0
        )
        # indices should be those needed to 'unshuffle' s1 back to ref ordering
        self.assertTrue(all([i == j for (i, j) in zip(unshuffle, indices)]))

        # Test using minimum image convention instead
        indices = RemapIndices.extract(
            s1, ref, mic=True, check_species=True, tolerance=1.0
        )
        # indices should be those needed to 'unshuffle' s1 back to ref ordering
        self.assertTrue(all([i == j for (i, j) in zip(unshuffle, indices)]))

        # since they're all the same, we can switch off the check_species
        indices = RemapIndices.extract(
            s1, ref, mic=True, check_species=False, tolerance=1.0
        )
        self.assertTrue(all([i == j for (i, j) in zip(unshuffle, indices)]))

        # Now the remapping itself
        s2 = Remap.extract(s1, ref, mic=True, check_species=True, tolerance=1.0)
        self.assertTrue((s2.positions == s1[unshuffle].positions).all())
        self.assertTrue(
            (s2.get_array("ref_labels") == ref.get_array("ref_labels")).all()
        )

        # Now test that a tight tolerance catches the slight mismatch
        with self.assertRaises(Exception) as context:
            indices = RemapIndices.extract(
                s1, ref, mic=True, check_species=True, tolerance=0.01
            )
        self.assertTrue(
            "are not within the tolerance distance" in str(context.exception)
        )

        # Add in a different species almost on top of atom 0
        # and check that it fails unless check species is True
        ref.append(Atom("H", position=ref[0].position + 0.00001))
        s1.append(Atom("H", position=ref[1].position + 0.00001))
        # Make sure the output
        s2 = Remap.extract(s1, ref, mic=True, check_species=False, tolerance=5.0)
        self.assertFalse(all(ref.symbols == s2.symbols))
        s2 = Remap.extract(s1, ref, mic=True, check_species=True, tolerance=5.0)
        self.assertTrue(all(ref.symbols == s2.symbols))

    def test_linkageprops(self):
        from soprano.properties.linkage import (
            Bonds,
            CoordinationHistogram,
            ElementPairs,
            HydrogenBonds,
            HydrogenBondsNumber,
            MoleculeCOMLinkage,
            MoleculeMass,
            MoleculeNumber,
            Molecules,
            MoleculeSpectralSort,
        )

        a = read(os.path.join(_TESTDATA_DIR, "mol_crystal.cif"))
        ethanol = read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

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

        # Test element pairs
        testdists, testpairs = ElementPairs.get(
            ethanol, element1="C", element2="O", return_pairs=True
        )
        self.assertTrue(testpairs[0] == (7, 8))  # The closest C-O pair
        self.assertTrue(testpairs[1] == (6, 8))  # The other C-O pair
        self.assertTrue(testdists[0] < testdists[1])
        # there are only two C-O pairs
        self.assertTrue(len(testdists) == 2)
        # Test the return_pairs=False option
        testdists = ElementPairs.get(
            ethanol, element1="C", element2="O", return_pairs=False
        )
        self.assertTrue(len(testdists) == 2)
        self.assertAlmostEqual(testdists[0], 1.428537, 6)
        self.assertAlmostEqual(testdists[1], 2.448431, 6)

        # what if we set the same element for both?
        testdists, testpairs = ElementPairs.get(
            ethanol, element1="C", element2="C", return_pairs=True
        )

        # correctly raise exception if element not present
        with self.assertRaises(ValueError) as context:
            testdists, testpairs = ElementPairs.get(
                ethanol, element1="O", element2="P", return_pairs=True
            )
            testdists, testpairs = ElementPairs.get(
                ethanol, element1="P", element2="O", return_pairs=True
            )

        # Test force certain size return pairs
        # less than total number of pairs
        testdists, testpairs = ElementPairs.get(
            ethanol, element1="C", element2="O", return_pairs=True, maxsize=1
        )
        self.assertTrue(len(testdists) == 1)
        # more than total number of pairs
        testdists, testpairs = ElementPairs.get(
            ethanol, element1="C", element2="O", return_pairs=True, maxsize=3
        )
        self.assertTrue(len(testdists) == 2)

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
        from collections import OrderedDict

        from soprano.properties.labeling import (
            HydrogenBondTypes,
            MoleculeSites,
            UniqueSites,
        )

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
    
    @skip_if_problematic_ase
    def test_labelprops_with_cif_labels(self):
        from soprano.properties.labeling import UniqueSites
        from collections import OrderedDict
        # Now we test labelleing Unique Sites
        a = read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        tagged_sites = UniqueSites.get(a, symprec=1e-3)
        max_tag = max(tagged_sites)
        # Z = 4 for this molecule, so we should have 4 copies of each
        # site.
        self.assertEqual(len(tagged_sites) / len(set(tagged_sites)), 4)

        # Now we make sure that
        # the symmetry-unique sites that remain are those we
        # would expect based on the existing CIF labels.

        # get indices of unique cif labels using OrderedDict
        unique_sites = [
            np.argmax(np.array(tagged_sites) == i) for i in range(max_tag + 1)
        ]
        labels = a.get_array("labels")
        unique_cif_labels = list(OrderedDict.fromkeys(labels))
        # take first match of each unique cif label
        sel_i_cif = [np.argmax(np.array(labels) == i) for i in unique_cif_labels]
        # test that they all match
        all_matched = all(np.array(unique_sites) == np.array(sel_i_cif))
        self.assertTrue(all_matched)

    def test_transformprops(self):
        from ase.quaternions import Quaternion

        from soprano.properties.transform import Mirror, Rotate, Translate
        from soprano.selection import AtomSelection

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
