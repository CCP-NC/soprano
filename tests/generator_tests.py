#!/usr/bin/env python
"""
Test code for the collection Generators
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
from ase.build import bulk
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
from soprano.collection.generate import (airssGen, linspaceGen, rattleGen,
                                         defectGen)
from soprano.utils import minimum_periodic
import unittest
import numpy as np

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")


class TestGenerate(unittest.TestCase):

    def test_airss(self):

        to_gen = 10

        # Load the Al.cell file (capture the annoying ASE output...)
        _stdout, sys.stdout = sys.stdout, StringIO()
        agen = airssGen(os.path.join(_TESTDATA_DIR, 'Al.cell'), n=to_gen)

        try:
            acoll = AtomsCollection(agen)
        except RuntimeError as e:
            if 'Buildcell' in str(e):
                sys.stdout = _stdout
                # Then we just don't have the program
                print('WARNING - The AIRSS generator could not be tested as '
                      'no AIRSS installation has been found on this system.')
                return
            else:
                raise e

        sys.stdout = _stdout

        # Some basic checks
        self.assertEqual(acoll.length, to_gen)
        self.assertTrue(all([chem == 'Al8'
                             for chem in acoll.all.get_chemical_formula()]))

    def test_linspace(self):

        a1 = Atoms('CO', [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]], cell=[1]*3)
        a2 = Atoms('CO', [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]], cell=[1]*3)

        lgen = linspaceGen(a1, a2, steps=5, periodic=False)
        lcoll = AtomsCollection(lgen)

        self.assertTrue(np.all(np.isclose(lcoll.all.get_positions()[:, 1, 1],
                                          np.linspace(0.2, 0.8, 5)
                                          )
                               ))

        # With periodicity
        lgen = linspaceGen(a1, a2, steps=5, periodic=True)
        lcoll = AtomsCollection(lgen)

        self.assertTrue(np.all(np.isclose(lcoll.all.get_positions()[:, 1, 1],
                                          np.linspace(0.2, -0.2, 5)
                                          )
                               ))

    def test_rattle(self):

        a = Atoms('CO', [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
        pos = a.get_positions()

        # Some exception tests
        wronggen = rattleGen(a, [3, 4, 5])
        self.assertRaises(ValueError, next, wronggen)
        wronggen = rattleGen(a, [[1, 2], [4, 5]])
        self.assertRaises(ValueError, next, wronggen)

        rgen = rattleGen(a, [[0.01, 0, 0], [0, 0.02, 0]])
        rcoll = AtomsCollection(rgen)
        rpos = rcoll.all.get_positions()

        self.assertTrue(np.all(np.abs((rpos-pos)[:, 0, 0]) <= 0.01))
        self.assertTrue(np.all(np.abs((rpos-pos)[:, 1, 1]) <= 0.02))

    def test_defect(self):

        si2 = bulk('Si')

        poisson_r = 0.5

        dGen = defectGen(si2, 'H', poisson_r)
        dColl = AtomsCollection(dGen)

        dPos = dColl.all.get_positions()[:, 0]

        holds = True
        for i, p1 in enumerate(dPos[:-1]):
            vecs, _ = minimum_periodic(dPos[i+1:]-p1, si2.get_cell())
            p_holds = (np.linalg.norm(vecs, axis=1) <= poisson_r).all()
            holds = holds and p_holds

        self.assertTrue(holds)


if __name__ == '__main__':
    unittest.main()
