#!/usr/bin/env python
"""
Test code for the PhylogenCluster and Gene classes
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from ase import Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
from soprano.analyse.phylogen import PhylogenCluster, Gene, load_genefile
import unittest
import numpy as np

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")


class TestPhylogen(unittest.TestCase):

    def test_instantiate(self):

        c1 = AtomsCollection([Atoms('C')])
        g1 = Gene('latt_abc_len', 1.0, {})
        p1 = PhylogenCluster(c1, [g1])

    def test_genefail(self):

        # Test that it DOES fail if a gene with wrong parameters is created
        self.assertRaises(ValueError, Gene, 'latt_abc_len',
                          1.0, {'wrong': True})

    def test_gene(self):

        c1 = AtomsCollection([Atoms('CCC', positions=[[0, 0, 0],
                                                      [0.2, 0, 0],
                                                      [0.85, 0, 0]],
                                    cell=[1]*3)])
        g1 = Gene('linkage_list', 1.0, {'size': 3})
        p1 = PhylogenCluster(c1, [g1])

        graw = p1.get_genome_vectors()
        self.assertTrue(np.allclose(graw[0], [0.15, 0.2, 0.35]))

    def test_customgene(self):

        c1 = AtomsCollection([Atoms('C', positions=[[i/5.0, 0, 0]])
                              for i in range(5)])

        # First, check failure
        self.assertRaises(RuntimeError, Gene, 'first_x_coord')

        # Then actual success
        def first_x_parser(c):
            return np.array(c.all.get_positions())[:, 0, 0]

        g1 = Gene('first_x_coord', parser=first_x_parser)
        self.assertTrue(np.all(g1.evaluate(c1) == [i/5.0 for i in range(5)]))

    def test_loadgene(self):

        gfpath = os.path.join(_TESTDATA_DIR, 'testfile.gene')

        g0 = Gene('latt_abc_len', 1.0, {})
        g1 = Gene('linkage_list', 0.5, {'size': 3})
        glist = load_genefile(gfpath)

        self.assertEqual(glist[0], g0)
        self.assertEqual(glist[1], g1)

        # Now test that it works also when instantiating a cluster
        c1 = AtomsCollection([Atoms('CCC',
                                    positions=np.random.random((3, 3)),
                                    cell=[1]*3)])
        p1 = PhylogenCluster(c1, gfpath)

    def test_loadarray(self):

        g0 = Gene('latt_abc_len', 1.0, {})

        c1 = AtomsCollection([Atoms('C'), Atoms('C')])
        c1.set_array('latt_abc_len', [1, 2])
        p1 = PhylogenCluster(c1, [])

        p1.set_genes([g0], load_arrays=True)

        self.assertTrue(np.all(p1.get_genome_vectors()[0] == [[1], [2]]))

    def test_cluster(self):

        a1 = Atoms(cell=[1, 1, 1], info={'name': 'a1'})
        a2 = Atoms(cell=[1, 1, 2], info={'name': 'a2'})
        a3 = Atoms(cell=[3, 3, 3], info={'name': 'a3'})

        c1 = AtomsCollection([a1, a2, a3])
        p1 = PhylogenCluster(c1, [Gene('latt_abc_len', 1.0, {})])

        clinds, clslice = p1.get_hier_clusters(0.5, method='complete')

        getname = lambda a: a.info['name']

        cnames = [sorted(c1[sl].all.map(getname)) for sl in clslice]
        cnames.sort(key=lambda x: len(x))

        self.assertTrue(cnames == [['a3'], ['a1', 'a2']])

        # Retry with k-means

        clustc = p1.get_kmeans_clusters(2)

        cnames = [sorted(c1[sl].all.map(getname)) for sl in clslice]
        cnames.sort(key=lambda x: len(x))

        self.assertTrue(cnames == [['a3'], ['a1', 'a2']])

if __name__ == '__main__':
    unittest.main()
