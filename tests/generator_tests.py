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
import unittest
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestGenerate(unittest.TestCase):
    def test_airss(self):

        from soprano.collection import AtomsCollection
        from soprano.collection.generate import airssGen

        to_gen = 10

        # Load the Al.cell file (capture the annoying ASE output...)
        _stdout, sys.stdout = sys.stdout, StringIO()
        agen = airssGen(os.path.join(_TESTDATA_DIR, "Al.cell"), n=to_gen)

        try:
            acoll = AtomsCollection(agen)
        except RuntimeError as e:
            if "Invalid output" in str(e):
                sys.stdout = _stdout
                print(
                    "WARNING - AIRSS' buildcell seems to have failed to run"
                    )
                return
            if "Buildcell" in str(e):
                sys.stdout = _stdout
                # Then we just don't have the program
                print(
                    "WARNING - The AIRSS generator could not be tested as "
                    "no AIRSS installation has been found on this system."
                )
                return
            else:
                raise e

        sys.stdout = _stdout

        # Some basic checks
        self.assertEqual(acoll.length, to_gen)
        self.assertTrue(
            all([chem == "Al8" for chem in acoll.all.get_chemical_formula()])
        )

    def test_linspace(self):

        from soprano.collection import AtomsCollection
        from soprano.collection.generate import linspaceGen

        a1 = Atoms("CO", [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]], cell=[1] * 3)
        a2 = Atoms("CO", [[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]], cell=[1] * 3)

        lgen = linspaceGen(a1, a2, steps=5, periodic=False)
        lcoll = AtomsCollection(lgen)

        self.assertTrue(
            np.all(
                np.isclose(lcoll.all.get_positions()[:, 1, 1], np.linspace(0.2, 0.8, 5))
            )
        )

        # With periodicity
        lgen = linspaceGen(a1, a2, steps=5, periodic=True)
        lcoll = AtomsCollection(lgen)

        self.assertTrue(
            np.all(
                np.isclose(
                    lcoll.all.get_positions()[:, 1, 1], np.linspace(0.2, -0.2, 5)
                )
            )
        )

    def test_rattle(self):

        from soprano.collection import AtomsCollection
        from soprano.collection.generate import rattleGen

        a = Atoms("CO", [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
        pos = a.get_positions()

        # Some exception tests
        wronggen = rattleGen(a, [3, 4, 5])
        self.assertRaises(ValueError, next, wronggen)
        wronggen = rattleGen(a, [[1, 2], [4, 5]])
        self.assertRaises(ValueError, next, wronggen)

        rgen = rattleGen(a, [[0.01, 0, 0], [0, 0.02, 0]])
        rcoll = AtomsCollection(rgen)
        rpos = rcoll.all.get_positions()

        self.assertTrue(np.all(np.abs((rpos - pos)[:, 0, 0]) <= 0.01))
        self.assertTrue(np.all(np.abs((rpos - pos)[:, 1, 1]) <= 0.02))

    def test_defect(self):

        from soprano.utils import minimum_periodic
        from soprano.collection import AtomsCollection
        from soprano.collection.generate import defectGen

        si2 = bulk("Si")

        poisson_r = 0.5

        dGen = defectGen(si2, "H", poisson_r)
        dColl = AtomsCollection(dGen)

        dPos = dColl.all.get_positions()[:, 0]

        holds = True
        for i, p1 in enumerate(dPos[:-1]):
            vecs, _ = minimum_periodic(dPos[i + 1 :] - p1, si2.get_cell())
            p_holds = (np.linalg.norm(vecs, axis=1) >= poisson_r).all()
            holds = holds and p_holds

        self.assertTrue(holds)

    def test_substitution_defect(self):
            
        from soprano.collection import AtomsCollection
        from soprano.selection import AtomSelection
        from soprano.collection.generate import substitutionGen
        import itertools

        # make an alloy
        atoms = bulk('Al', 'fcc') * (3,3,3) # 27 Al atoms
        # All Al-Al bonds are 2.864 Angstroms

        alsel = AtomSelection.from_element(atoms, 'Al')

        def _min_sep(s, subs):
            # return true if all the subs are at least 3.0 Angstroms apart
            for i, j in itertools.combinations(subs, 2):
                if s.get_distance(i, j, mic=True) < 3.0:
                    return False
            return True

        # Substitute two hydrogens at a time with chlorine, but only if neither is bonded to sulfur
        sG = substitutionGen(atoms, 'Sn', to_replace=alsel, n=3, accept=_min_sep)
        sColl = AtomsCollection(sG)

        # Check that all the substitutions are at least 3.0 Angstroms apart
        for s in sColl:
            self.assertTrue(_min_sep(s, s.get_indices('Sn')))
        
        # check that we generate the correct number of possible configurations
        # the total number of possible configurations is 27 choose 3
        # (27! / (3! * (27-3)!)) = 2925
        # but we have to remove the ones where the substitutions are too close
        # to each other
        # so we actually end up with 387 possible configurations:
        self.assertEqual(len(sColl), 387)


    def test_molneigh(self):

        from soprano.properties.linkage import Molecules
        from soprano.collection.generate import molecularNeighbourhoodGen

        metmol = molecule("CH4")

        cellmol = metmol.copy()
        cellmol.set_pbc(True)
        cellmol.set_cell([6, 6, 6])
        c2 = cellmol.copy()
        c2.set_positions(c2.get_positions() + 3)
        cellmol += c2

        mols = Molecules.get(cellmol)

        mnGen = molecularNeighbourhoodGen(cellmol, mols, max_R=5.2)
        all_neigh = [a for a in mnGen]

        self.assertEqual(len(all_neigh), 9)
        self.assertAlmostEqual(
            all_neigh[0].info["neighbourhood_info"]["molecule_distance"], 0
        )
        self.assertAlmostEqual(
            all_neigh[1].info["neighbourhood_info"]["molecule_distance"], 3 ** 1.5
        )

        mnGen = molecularNeighbourhoodGen(cellmol, mols, max_R=5, method="nearest")
        all_neigh = [a for a in mnGen]

        self.assertEqual(len(all_neigh), 9)


if __name__ == "__main__":
    unittest.main()
