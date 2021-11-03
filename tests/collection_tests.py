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
import unittest
import numpy as np
from ase import io, Atoms

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")
_TESTSAVE_DIR = os.path.join(_TEST_DIR, "test_save")


class TestCollection(unittest.TestCase):
    def test_save(self):

        from soprano.collection import AtomsCollection

        # Test saving and loading collection
        testcoll = AtomsCollection([Atoms("H")])
        testcoll.set_array("test", np.array([2]))

        outf = os.path.join(_TEST_DIR, "collection.pkl")
        testcoll.save(outf)

        # Reload
        testcoll = AtomsCollection.load(outf)
        self.assertEqual(testcoll.get_array("test")[0], 2)

        os.remove(outf)

    def test_loadres(self):

        from soprano.collection import AtomsCollection

        # Load some test files and check if regular loading works
        reslist = glob.glob(os.path.join(_TESTDATA_DIR, "rescollection", "*.res"))
        testcoll = AtomsCollection(reslist)
        self.assertEqual(testcoll.length, len(reslist))

        # Now try the same, but with single atoms objects
        aselist = [io.read(str(fname)) for fname in reslist]
        testcoll = AtomsCollection(aselist)
        self.assertEqual(testcoll.length, len(reslist))

    def test_arrays(self):

        from soprano.collection import AtomsCollection

        # Generate a few random structures
        elrnd = ["H", "C", "O", "N"]
        asernd = []

        rng = np.random.default_rng(0)

        for n in range(4):
            aselen = rng.integers(1, 10)
            asernd.append(
                Atoms(
                    symbols=rng.choice(elrnd, aselen),
                    positions=rng.random((aselen, 3)),
                )
            )

        testcoll = AtomsCollection(asernd)
        # Now try assigning some arrays
        arr = np.arange(testcoll.length)
        testcoll.set_array("testarr", arr, shape=(1,))
        testcoll.set_array("testarr_2", list(zip(arr, arr)), shape=(2,))
        testcoll.set_array("testarr_func", lambda a: len(a.get_positions()), shape=(1,))

        self.assertTrue(np.all(testcoll.get_array("testarr") == arr))

    def test_calculator(self):

        from ase.calculators.lj import LennardJones
        from soprano.collection import AtomsCollection

        # Generate a few random structures
        elrnd = ["H", "C", "O", "N"]
        asernd = []

        rng = np.random.default_rng(0)

        for n in range(4):
            aselen = rng.integers(2, 10)
            asernd.append(
                Atoms(
                    symbols=rng.choice(elrnd, aselen),
                    positions=rng.random((aselen, 3)),
                )
            )

        testcoll = AtomsCollection(asernd)

        testcoll.set_calculators(
            LennardJones, labels=["a", "b", "c", "d"], params={"epsilon": 1.2}
        )
        testcoll.run_calculators()

        def extract_nrg(s):
            return s.calc.results["energy"]

        energies = np.array(testcoll.all.map(extract_nrg))

        self.assertTrue(not np.isnan(energies).any())

    def test_sum(self):

        from soprano.collection import AtomsCollection

        # Generate a few random structures
        elrnd = ["H", "C", "O", "N"]
        asernd = []

        rng = np.random.default_rng(0)

        for n in range(4):
            aselen = rng.integers(1, 10)
            asernd.append(
                Atoms(
                    symbols=rng.choice(elrnd, aselen),
                    positions=rng.random((aselen, 3)),
                )
            )

        testcoll1 = AtomsCollection(asernd[:2])
        testcoll2 = AtomsCollection(asernd[2:])

        testcoll1.set_array("joint", ["t", "e"])
        testcoll2.set_array("joint", ["s", "t"])

        testcoll = testcoll1
        testcoll += testcoll2

        self.assertTrue("".join(testcoll.get_array("joint")) == "test")

    def test_chunks(self):

        from soprano.collection import AtomsCollection

        full_len = 10
        chunk_len = 3
        chunk_n = 2

        # Generate a few random structures
        elrnd = ["H", "C", "O", "N"]
        asernd = []

        rng = np.random.default_rng(0)

        for n in range(full_len):
            aselen = rng.integers(1, 10)
            asernd.append(
                Atoms(
                    symbols=rng.choice(elrnd, aselen),
                    positions=rng.random((aselen, 3)),
                )
            )

        testcoll = AtomsCollection(asernd)

        # Test with size
        chunks = testcoll.chunkify(chunk_size=chunk_len)
        self.assertEqual(len(chunks), np.ceil(full_len / (1.0 * chunk_len)))
        self.assertEqual(chunks[-1].length, full_len % chunk_len)

        # Test with number
        chunks = testcoll.chunkify(chunk_n=chunk_n)
        self.assertEqual(len(chunks), chunk_n)

    def test_sorting(self):

        from soprano.collection import AtomsCollection

        # Generate a few structures
        struct_n = 5

        aselist = []
        for n in range(struct_n):
            aselist.append(Atoms())

        testcoll = AtomsCollection(aselist)

        testcoll.set_array("sorter", np.array(range(struct_n, 0, -1)))
        testcoll.set_array("sorted", np.array(range(1, struct_n + 1)))

        testcoll = testcoll.sorted_byarray("sorter")

        self.assertTrue(np.all(testcoll.get_array("sorted") == range(struct_n, 0, -1)))

    def test_slices(self):

        from soprano.collection import AtomsCollection

        aselist = [Atoms("C"), Atoms("C"), Atoms("H"), Atoms("C")]

        testcoll = AtomsCollection(aselist)

        coll_h = testcoll[2]
        coll_c = testcoll[(0, 1, 3)]
        coll_b = testcoll[[False, False, True, False]]

        self.assertTrue(all([f == "H" for f in coll_h.all.get_chemical_formula()]))
        self.assertTrue(all([f == "C" for f in coll_c.all.get_chemical_formula()]))
        self.assertTrue(all([f == "H" for f in coll_b.all.get_chemical_formula()]))

    def test_tree(self):

        from soprano.collection import AtomsCollection

        aselist = [Atoms("C"), Atoms("H"), Atoms("N"), Atoms("O")]

        testcoll = AtomsCollection(aselist)

        testcoll.save_tree(_TESTSAVE_DIR, "xyz", safety_check=0)
        loadcoll = AtomsCollection.load_tree(_TESTSAVE_DIR, "xyz")

        self.assertTrue("".join(loadcoll.all.get_chemical_formula()) == "CHNO")

        # Try a custom save format
        def custom_save(a, path, format_string):
            with open(os.path.join(path, "struct.custom"), "w") as f:
                f.write(format_string.format("".join(a.get_chemical_formula())))

        def custom_load(path, marker):
            with open(os.path.join(path, "struct.custom")) as f:
                l = f.read().strip()
                return Atoms(l.split(marker)[1].strip())

        testcoll.save_tree(
            _TESTSAVE_DIR,
            custom_save,
            opt_args={"format_string": "Formula:{0}"},
            safety_check=2,
        )
        loadcoll = AtomsCollection.load_tree(
            _TESTSAVE_DIR, custom_load, opt_args={"marker": ":"}
        )

        self.assertTrue("".join(loadcoll.all.get_chemical_formula()) == "CHNO")


if __name__ == "__main__":
    unittest.main()
