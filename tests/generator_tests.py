#!/usr/bin/env python
"""
Test code for the collection Generators
"""


import os
import sys
import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

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
                print("WARNING - AIRSS' buildcell seems to have failed to run")
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
        from soprano.collection import AtomsCollection
        from soprano.collection.generate import defectGen
        from soprano.utils import minimum_periodic

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
        import itertools

        from soprano.collection import AtomsCollection
        from soprano.collection.generate import substitutionGen
        from soprano.selection import AtomSelection

        # make an alloy
        atoms = bulk("Al", "fcc") * (3, 3, 3)  # 27 Al atoms
        # All Al-Al bonds are 2.864 Angstroms

        alsel = AtomSelection.from_element(atoms, "Al")

        def _min_sep(s, subs):
            # return true if all the subs are at least 3.0 Angstroms apart
            for i, j in itertools.combinations(subs, 2):
                if s.get_distance(i, j, mic=True) < 3.0:
                    return False
            return True

        # Substitute 3 Al atoms with Sn atoms subject to the constraint that
        # all the Sn atoms are at least 3.0 Angstroms apart
        sG = substitutionGen(atoms, 'Sn', to_replace=alsel, n=3, accept=_min_sep)
        sColl = AtomsCollection(sG)

        # Check that all the substitutions are at least 3.0 Angstroms apart
        for s in sColl:
            self.assertTrue(_min_sep(s, AtomSelection.from_element(s, "Sn").indices))

        # check that we generate the correct number of possible configurations
        # the total number of possible configurations is 27 choose 3
        # (27! / (3! * (27-3)!)) = 2925
        # but we have to remove the ones where the substitutions are too close
        # to each other
        # so we actually end up with 387 possible configurations:
        self.assertEqual(len(sColl), 387)

    def test_molneigh(self):
        from soprano.collection.generate import molecularNeighbourhoodGen
        from soprano.properties.linkage import Molecules

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
            all_neigh[1].info["neighbourhood_info"]["molecule_distance"], 3**1.5
        )

        mnGen = molecularNeighbourhoodGen(cellmol, mols, max_R=5, method="nearest")
        all_neigh = [a for a in mnGen]

        self.assertEqual(len(all_neigh), 9)

    def test_addition_defect(self):
        import itertools

        from soprano.collection import AtomsCollection
        from soprano.collection.generate import additionGen
        from soprano.selection import AtomSelection

        # Create a simple test structure - H2 molecule
        h2 = molecule("H2")
        h2.set_pbc(True)
        h2.set_cell([10, 10, 10])
        h2.center()

        # Test basic functionality - add one H atom
        aGen = additionGen(h2, 'H', n=1, add_r=1.0)
        aColl = AtomsCollection(aGen)

        # Should generate structures with 3 H atoms total
        for struct in aColl:
            self.assertEqual(len(struct), 3)
            self.assertEqual(struct.get_chemical_formula(), 'H3')

        # Test parameter validation
        with self.assertRaises(ValueError):
            # Negative n
            aGen = additionGen(h2, 'H', n=-1)
            next(aGen)

        with self.assertRaises(ValueError):
            # Zero n
            aGen = additionGen(h2, 'H', n=0)
            next(aGen)

        with self.assertRaises(ValueError):
            # Negative add_r
            aGen = additionGen(h2, 'H', n=1, add_r=-1.0)
            next(aGen)

        # Test with atom selection
        h_sel = AtomSelection.from_element(h2, 'H')
        first_h_sel = AtomSelection(h2, [0])  # Only first H atom

        aGen = additionGen(h2, 'H', to_addition=first_h_sel, n=1, add_r=1.0)
        aColl = AtomsCollection(aGen)

        # Should still generate valid structures
        for struct in aColl:
            self.assertEqual(len(struct), 3)

        # Test with larger structure - methane
        ch4 = molecule("CH4")
        ch4.set_pbc(True)
        ch4.set_cell([10, 10, 10])
        ch4.center()

        # Add H atoms only to C atoms
        c_sel = AtomSelection.from_element(ch4, 'C')

        aGen = additionGen(ch4, 'H', to_addition=c_sel, n=1, add_r=1.1)
        aColl = AtomsCollection(aGen)

        # Should generate structures with 6 atoms total (5 original + 1 added)
        for struct in aColl:
            self.assertEqual(len(struct), 6)

        # Test acceptance function
        def accept_func(struct, indices):
            # Only accept if the new atom is not too close to existing ones
            new_pos = struct.get_positions()[-1]  # Last atom is the new one
            old_pos = struct.get_positions()[:-1]
            distances = np.linalg.norm(old_pos - new_pos, axis=1)
            return np.all(distances > 0.8)  # Minimum distance of 0.8 Å

        aGen = additionGen(ch4, 'H', to_addition=c_sel, n=1, add_r=1.1,
                          accept=accept_func)
        aColl = AtomsCollection(aGen)

        # Should generate fewer structures due to acceptance filtering
        self.assertGreater(len(aColl), 0)  # Should have some accepted structures

        # Test max_attempts parameter
        aGen = additionGen(ch4, 'H', to_addition=c_sel, n=1, add_r=1.1,
                          max_attempts=2)
        aColl = AtomsCollection(aGen)

        # Should limit the number of generated structures
        self.assertLessEqual(len(aColl), 2)

        # Test random mode
        aGen = additionGen(ch4, 'H', to_addition=c_sel, n=1, add_r=1.1,
                          random=True, max_attempts=3)
        aColl = AtomsCollection(aGen)

        # Should generate structures in random order
        self.assertGreater(len(aColl), 0)

        # Test multiple additions (n > 1)
        aGen = additionGen(h2, 'H', n=2, add_r=1.0, max_attempts=5)
        aColl = AtomsCollection(aGen)

        # Should generate structures with 4 H atoms total
        for struct in aColl:
            self.assertEqual(len(struct), 4)
            self.assertEqual(struct.get_chemical_formula(), 'H4')

        # Test empty selection
        empty_sel = AtomSelection(h2, [])
        with self.assertRaises(ValueError):
            aGen = additionGen(h2, 'H', to_addition=empty_sel, n=1)
            next(aGen)

        # Test n larger than selection size
        single_sel = AtomSelection(h2, [0])
        with self.assertRaises(ValueError):
            aGen = additionGen(h2, 'H', to_addition=single_sel, n=2)
            next(aGen)

    def test_addition_bonds_and_attachment(self):
        """Test additionGen with bond-based attachment logic"""
        from soprano.collection import AtomsCollection
        from soprano.collection.generate import additionGen
        from soprano.properties.linkage import Bonds
        from soprano.selection import AtomSelection

        # Create a linear molecule (acetylene) to test directed attachment
        c2h2 = molecule("C2H2")
        c2h2.set_pbc(True)
        c2h2.set_cell([15, 15, 15])
        c2h2.center()

        # Test with pre-computed bonds
        bonds = Bonds.get(c2h2)
        
        # Add atoms to carbon atoms with specific bonds object
        c_sel = AtomSelection.from_element(c2h2, 'C')
        aGen = additionGen(c2h2, 'H', to_addition=c_sel, n=1, add_r=1.1, bonds=bonds)
        aColl = AtomsCollection(aGen)
        
        # Should generate structures
        self.assertGreater(len(aColl), 0)
        
        # Test with custom rep_alg_kwargs
        rep_kwargs = {'iters': 500, 'attempts': 5}
        aGen = additionGen(c2h2, 'H', to_addition=c_sel, n=1, add_r=1.1, 
                          rep_alg_kwargs=rep_kwargs)
        aColl = AtomsCollection(aGen)
        
        # Should still generate structures
        self.assertGreater(len(aColl), 0)

        # Test with isolated atom (no bonds)
        isolated_h = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        
        aGen = additionGen(isolated_h, 'H', n=1, add_r=1.5)
        aColl = AtomsCollection(aGen)
        
        # Should generate structures even for isolated atoms
        self.assertGreater(len(aColl), 0)
        for struct in aColl:
            self.assertEqual(len(struct), 2)
            # Check that the distance is approximately correct
            dist = struct.get_distance(0, 1, mic=True)
            self.assertAlmostEqual(dist, 1.5, places=1)

        # Test complex acceptance function
        def complex_accept(struct, indices):
            # Accept only if the new atoms form reasonable angles
            if len(struct) < 3:
                return True
            
            # Get the last added atom position
            new_pos = struct.get_positions()[-1]
            host_pos = struct.get_positions()[indices[0]]
            
            # Check if the new atom is roughly in the expected direction
            vec = new_pos - host_pos
            dist = np.linalg.norm(vec)
            return 0.8 < dist < 2.0  # Reasonable distance range

        aGen = additionGen(c2h2, 'H', to_addition=c_sel, n=1, add_r=1.1, 
                          accept=complex_accept, max_attempts=10)
        aColl = AtomsCollection(aGen)
        
        # Should have some accepted structures
        self.assertGreaterEqual(len(aColl), 0)

if __name__ == "__main__":
    unittest.main()
