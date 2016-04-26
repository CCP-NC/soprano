#!/usr/bin/env python
"""
Test code for AtomSelection
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
import shutil
from ase import io, Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.selection import AtomSelection

import unittest
import numpy as np


class TestSelection(unittest.TestCase):

    def test_basic(self):

    	# Create an Atoms object
    	a = Atoms('HHH')

    	# Try a valid selection
    	s1 = AtomSelection(a, [0, 2])

    	# Try an invalid one
    	self.assertRaises(ValueError, AtomSelection, a, [0,3])

    	# Check validation
    	self.assertTrue(s1.validate(a))

    	# Now make a subset
    	a_s = s1.subset(a)
    	self.assertTrue(len(a_s) == 2)

    def test_operators(self):

    	# Create an Atoms object
    	a1 = Atoms('HHH')
    	a2 = Atoms('CC')

    	s1 = AtomSelection(a1, [0, 2])
    	s2 = AtomSelection(a1, [0, 1])

    	self.assertTrue(set((s1+s2).indices) == set([0,1,2]))
    	self.assertTrue(set((s1-s2).indices) == set([2]))
    	self.assertTrue(set((s1*s2).indices) == set([0]))

    def test_selectors(self):

    	# Multiple tests for various methods
    	a = Atoms('HCHC', positions=[[i]*3 for i in range(4)],
    			  cell=[4]*3, pbc=[True]*3)

    	# Element test
    	s1 = AtomSelection.from_element(a, 'C')

    	self.assertTrue(set(s1.indices) == set([1,3]))

    	# Box test
    	s1 = AtomSelection.from_box(a, [1.5]*3, [4.5]*3, periodic=True)
    	s2 = AtomSelection.from_box(a, [1.5]*3, [4.5]*3, periodic=False)
    	s3 = AtomSelection.from_box(a, [0.375]*3, [1.125]*3, periodic=True,
    								scaled=True)

    	self.assertTrue(set(s1.indices) == set([0,2,3]))
    	self.assertTrue(set(s2.indices) == set([2,3]))
    	self.assertTrue(set(s3.indices) == set([0,2,3]))


if __name__ == '__main__':
    unittest.main()