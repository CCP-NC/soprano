#!/usr/bin/env python
"""
Test code for CLI utilities
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
from ase.visualize import view
from soprano.scripts.nmr import (
    tag_functional_groups,
    merge_tagged_sites,
)

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

class TestCLIUtils(unittest.TestCase):
    def test_tag_functional_groups(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        eth_tagged = tag_functional_groups('CH3', eth)
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
        self.assertTrue(np.all([symbols[idx] == 'H' for idx in tagged_atoms]))
        # the indices in this case happen to be 0, 1, 2
        self.assertTrue(np.all(tagged_atoms == np.arange(3)))

    def test_merge_tagged_sites(self):
        eth = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        # custom tagging
        tags = [0,0,0,1,1,2,3,4,5]
        labels = ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H3', 'C1', 'C2', 'O1']
        eth.set_tags(tags)
        eth.set_array('labels', None)
        eth.set_array('labels', np.array(labels), dtype='U25')
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
        expected_symbols = np.array(['H','H','H','C','C','O'])
        self.assertTrue(np.all(symbols == expected_symbols))
        # make sure the tags are correct
        expected_tags = np.arange(6)
        self.assertTrue(np.all(eth_merged.get_tags() == expected_tags))
        # make sure the labels are correct
        expected_labels = ['H1a,H1b,H1c', 'H2a,H2b', 'H3', 'C1', 'C2', 'O1']
        labels = eth_merged.get_array('labels')
        self.assertTrue(np.all(labels == expected_labels))




if __name__ == "__main__":
    unittest.main()
