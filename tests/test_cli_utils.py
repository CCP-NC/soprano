#!/usr/bin/env python
"""
Test code for CLI utilities
"""


import os
import unittest

import numpy as np
from ase import Atoms, io

from soprano.scripts.molecules import (
    extract_molecules,
    redefine_unit_cell,
)
from soprano.scripts.nmr import (
    merge_tagged_sites,
    tag_functional_groups,
)

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestCLIUtils(unittest.TestCase):
    def test_tag_functional_groups(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        eth_tagged = tag_functional_groups("CH3", eth)
        # make sure we still have the correct number of atoms
        self.assertEqual(len(eth_tagged), 9)
        # make sure there are right number of unique tags
        unique_tags, counts = np.unique(eth_tagged.get_tags(), return_counts=True)
        self.assertEqual(len(unique_tags), 7)
        # make sure there are right number of atoms with each tag
        # (only one tag with more that one site)
        self.assertEqual(len(unique_tags[counts > 1]), 1)
        # make sure the tagged atoms are the right ones
        special_tag = unique_tags[counts > 1]
        tagged_atoms = np.where(eth_tagged.get_tags() == special_tag)[0]
        self.assertEqual(len(tagged_atoms), 3)
        symbols = eth_tagged.get_chemical_symbols()
        self.assertTrue(np.all([symbols[idx] == "H" for idx in tagged_atoms]))
        # the indices in this case happen to be 0, 1, 2
        self.assertTrue(np.all(tagged_atoms == np.arange(3)))

    def test_merge_tagged_sites(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        # custom tagging
        tags = [0, 0, 0, 1, 1, 2, 3, 4, 5]
        labels = ["H1a", "H1b", "H1c", "H2a", "H2b", "H3", "C1", "C2", "O1"]
        eth.set_tags(tags)
        eth.set_array("labels", None)
        eth.set_array("labels", np.array(labels), dtype="U25")
        eth_merged = merge_tagged_sites(eth)
        # useful for visual inspection
        # view([eth, eth_merged])
        # make sure we have the correct number of atoms
        self.assertEqual(len(eth_merged), 6)
        # make sure there are right number of unique tags
        unique_tags, counts = np.unique(eth_merged.get_tags(), return_counts=True)
        self.assertEqual(len(unique_tags), 6)
        # make sure the atoms left are:
        # H,H,H,C,C,O
        symbols = np.array(eth_merged.get_chemical_symbols())
        expected_symbols = np.array(["H", "H", "H", "C", "C", "O"])
        self.assertTrue(np.all(symbols == expected_symbols))
        # make sure the tags are correct
        expected_tags = np.arange(6)
        self.assertTrue(np.all(eth_merged.get_tags() == expected_tags))
        # make sure the labels are correct
        expected_labels = ["H1a,H1b,H1c", "H2a,H2b", "H3", "C1", "C2", "O1"]
        labels = eth_merged.get_array("labels")
        self.assertTrue(np.all(labels == expected_labels))


class TestExtractMolecules(unittest.TestCase):
    def test_extract_molecules_simple(self):
        # Load a structure file from the test_data directory
        atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        # Call the function with the atoms and use_cell_indices parameters
        molecules = extract_molecules(atoms, use_cell_indices=True)
        assert len(molecules) == 1

    def test_extract_molecules_complex(self):
        # Load another structure file from the test_data directory
        atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        # Call the function with the atoms and use_cell_indices parameters
        molecules = extract_molecules(atoms, use_cell_indices=True)
        assert len(molecules) == 4

        # Load another structure file from the test_data directory
        atoms = io.read(os.path.join(_TESTDATA_DIR, "mol_crystal.cif"))
        # Call the function with the atoms and use_cell_indices parameters
        molecules = extract_molecules(atoms, use_cell_indices=True)
        assert len(molecules) == 4


class TestRedefineUnitCell(unittest.TestCase):
    def test_redefine_unit_cell(self):
        # Create an Atoms object
        atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.7)])

        # Define a new cell
        new_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Call the function with the atoms, new cell, and center parameters
        result = redefine_unit_cell(atoms, new_cell, center=True, vacuum=0.5)

        # Check that the function returned the correct result
        # This will depend on the specific atoms and the expected result
        expected_result = Atoms('H2', positions=[(0.5, 0.5, 0.5), (0.5, 0.5, 1.2)], cell=new_cell, pbc=True)
        #TODO  - what should the expected result be? - add assert statements here
    def test_redefine_unit_cell_error(self):
        # Create an Atoms object
        atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.7)])

        # Define a new cell
        new_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Call the function with the atoms, new cell, and center parameters
        # Expect a RuntimeError because center is False and vacuum is not None
        with self.assertRaises(RuntimeError):
            redefine_unit_cell(atoms, new_cell, center=False, vacuum=0.5)

if __name__ == "__main__":
    unittest.main()
