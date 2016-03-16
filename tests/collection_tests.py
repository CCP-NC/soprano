#!/usr/bin/env python
"""
Test code for the AtomsCollection class
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
from ase import io, Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
import unittest
import numpy as np

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")

class TestCollection(unittest.TestCase):

    def test_loadres(self):

        # Load some test files and check if regular loading works
        reslist = glob.glob(os.path.join(_TESTDATA_DIR, 'rescollection', '*.res'))
        testcoll = AtomsCollection(reslist)
        self.assertEqual(testcoll.length, len(reslist))

        # Now try the same, but with single atoms objects
        aselist = [io.read(str(fname)) for fname in reslist]
        testcoll = AtomsCollection(aselist)
        self.assertEqual(testcoll.length, len(reslist))

    def test_arrays(self):

        # Generate a few random structures
        elrnd = ['H','C','O','N']
        asernd = []
        for n in range(4):
            aselen = np.random.randint(1, 10)
            asernd.append(Atoms(symbols=np.random.choice(elrnd, aselen),
                                positions=np.random.random((aselen, 3))))

        testcoll = AtomsCollection(asernd)
        # Now try assigning some arrays
        arr = range(testcoll.length)
        testcoll.set_array('testarr', arr, shape=(1,))
        testcoll.set_array('testarr_2', zip(arr, arr), shape=(2,))
        testcoll.set_array('testarr_func',
                           lambda a: len(a.get_positions()),
                           shape=(1,))

        self.assertTrue(np.all(testcoll.get_array('testarr') == arr))

    def test_calculator(self):

        from ase.calculators.test import TestPotential
        from ase.calculators.lj import LennardJones

        # Generate a few random structures
        elrnd = ['H','C','O','N']
        asernd = []
        for n in range(4):
            aselen = np.random.randint(2, 10)
            asernd.append(Atoms(symbols=np.random.choice(elrnd, aselen),
                                positions=np.random.random((aselen, 3))))

        testcoll = AtomsCollection(asernd)

        testcoll.set_calculators(LennardJones,
                                 labels=['a', 'b', 'c', 'd'],
                                 params={'epsilon': 1.2})
        testcoll.run_calculators()
        extract_nrg = lambda s: s.calc.results['energy']
        energies = np.array(testcoll.all.map(extract_nrg))

        self.assertTrue(np.all(energies > 0.0))

    def test_sum(self):

        # Generate a few random structures
        elrnd = ['H','C','O','N']
        asernd = []
        for n in range(4):
            aselen = np.random.randint(1, 10)
            asernd.append(Atoms(symbols=np.random.choice(elrnd, aselen),
                                positions=np.random.random((aselen, 3))))

        testcoll1 = AtomsCollection(asernd[:2])
        testcoll2 = AtomsCollection(asernd[2:])

        testcoll1.set_array('joint', ['t', 'e'])
        testcoll2.set_array('joint', ['s', 't'])

        testcoll = testcoll1
        testcoll += testcoll2

        self.assertTrue(''.join(testcoll.get_array('joint')) == 'test')

    def test_chunks(self):

        full_len = 10
        chunk_len = 3
        chunk_n = 2

        # Generate a few random structures
        elrnd = ['H','C','O','N']
        asernd = []
        for n in range(full_len):
            aselen = np.random.randint(1, 10)
            asernd.append(Atoms(symbols=np.random.choice(elrnd, aselen),
                                positions=np.random.random((aselen, 3))))

        testcoll = AtomsCollection(asernd)

        # Test with size
        chunks = testcoll.chunkify(chunk_size=chunk_len)
        self.assertEqual(len(chunks), np.ceil(full_len/(1.0*chunk_len)))
        self.assertEqual(chunks[-1].length, full_len%chunk_len)

        # Test with number
        chunks = testcoll.chunkify(chunk_n=chunk_n)
        self.assertEqual(len(chunks), chunk_n)

    def test_sorting(self):

        # Generate a few structures
        struct_n = 5

        aselist = []
        for n in range(struct_n):
            aselist.append(Atoms())

        testcoll = AtomsCollection(aselist)

        testcoll.set_array('sorter', np.array(range(struct_n, 0, -1)))
        testcoll.set_array('sorted', np.array(range(1, struct_n+1)))

        testcoll = testcoll.sorted_byarray('sorter')

        self.assertTrue(np.all(testcoll.get_array('sorted') == range(struct_n,
                                                                     0, -1)))

    def test_slices(self):

        aselist = [
            Atoms('C'),
            Atoms('C'),
            Atoms('H'),
            Atoms('C')
        ]

        testcoll = AtomsCollection(aselist)

        coll_h = testcoll[2]
        coll_c = testcoll[(0, 1, 3)]

        self.assertTrue(all([s.get_chemical_formula() == 'H'
                             for s in coll_h.structures]))
        self.assertTrue(all([s.get_chemical_formula() == 'C'
                             for s in coll_c.structures]))



if __name__ == '__main__':
    unittest.main()