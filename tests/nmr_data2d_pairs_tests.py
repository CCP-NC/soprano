#!/usr/bin/env python
"""Regression tests: NMRData2D user ``pairs`` are indices into the input structure."""

import os
import unittest
import warnings

import numpy as np
from ase import io

from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.utils import get_atom_labels
from soprano.nmr.extract import label_atoms, nmr_extract_atoms

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def _ethanol_o_first():
    """Ethanol reordered so O precedes C and H: O(0), H(1-6), C(7), C(8)."""
    atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
    return atoms[[8, 0, 1, 2, 3, 4, 5, 6, 7]]


class TestPairsAreInputIndices(unittest.TestCase):

    def setUp(self):
        self.atoms = _ethanol_o_first()
        self.symbols = self.atoms.get_chemical_symbols()
        self.kw = dict(xelement="C", yelement="H", correlation_strength_metric="distance")

    def test_pairs_pick_the_input_atoms(self):
        """Input (C, H) indices pick those atoms even though O comes first."""
        pairs = [(7, 1), (8, 6)]
        nd = NMRData2D(self.atoms, **self.kw, pairs=pairs)

        self.assertEqual(len(nd.pair_distances), len(pairs))
        for (i_in, j_in), dist in zip(pairs, nd.pair_distances):
            self.assertAlmostEqual(
                dist, self.atoms.get_distance(i_in, j_in, mic=True), places=6,
            )

    def test_pairs_point_at_input_positions_and_labels(self):
        """Each derived pair holds the input atoms' positions and labels."""
        pairs = [(7, 1), (8, 6)]
        nd = NMRData2D(self.atoms, **self.kw, pairs=pairs)

        src_labels = get_atom_labels(label_atoms(self.atoms.copy()), None)
        nd_labels = get_atom_labels(nd.atoms, None)
        for (i_in, j_in), (i, j) in zip(pairs, nd.pairs):
            np.testing.assert_allclose(nd.atoms.positions[i], self.atoms.positions[i_in])
            np.testing.assert_allclose(nd.atoms.positions[j], self.atoms.positions[j_in])
            self.assertEqual(nd_labels[i], src_labels[i_in])
            self.assertEqual(nd_labels[j], src_labels[j_in])

    def test_wrong_element_x_raises(self):
        """An x atom that is not of xelement raises, naming its index and element."""
        o_idx = self.symbols.index("O")
        h_idx = self.symbols.index("H")
        with self.assertRaisesRegex(ValueError, rf"atom {o_idx} is O"):
            NMRData2D(self.atoms, **self.kw, pairs=[(o_idx, h_idx)])

    def test_wrong_element_y_raises(self):
        """A y atom that is not of yelement raises, naming its index and element."""
        c_idx = self.symbols.index("C")
        o_idx = self.symbols.index("O")
        with self.assertRaisesRegex(ValueError, rf"atom {o_idx} is O"):
            NMRData2D(self.atoms, **self.kw, pairs=[(c_idx, o_idx)])

    def test_wrong_element_raises_with_reduce(self):
        """The element check also applies under reduce=True."""
        o_idx = self.symbols.index("O")
        h_idx = self.symbols.index("H")
        with self.assertRaisesRegex(ValueError, rf"atom {o_idx} is O"):
            NMRData2D(self.atoms, **self.kw, reduce=True, pairs=[(o_idx, h_idx)])


class TestPairsReduceConsistency(unittest.TestCase):
    """With reduce=True, input pairs land on the sites their atoms merge into."""

    @classmethod
    def setUpClass(cls):
        cls.atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        _, cls.reduce_map = nmr_extract_atoms(
            label_atoms(cls.atoms.copy()), reduce=True, symprec=1e-4,
            return_index_map=True,
        )
        cls.reduce_map = np.asarray(cls.reduce_map)
        cls.h_idx = [i for i, s in enumerate(cls.atoms.get_chemical_symbols()) if s == "H"]
        cls.kw = dict(xelement="H", yelement="H", correlation_strength_metric="distance")

    def _pair_on_different_sites(self):
        """Two H atoms in the input that merge into different sites."""
        for a in self.h_idx:
            for b in self.h_idx:
                if self.reduce_map[a] != self.reduce_map[b]:
                    return a, b
        self.fail("no two H atoms map to different sites")

    def test_reduce_true_uses_the_requested_atom_pair(self):
        """pairs=[(a, b)] with reduce=True evaluates a and b themselves.

        Not the site representatives, which may sit on different molecules.
        """
        reps = {int(self.reduce_map[a]): a for a in reversed(self.h_idx)}
        a, b = next(
            (a, b) for a in self.h_idx for b in self.h_idx
            if self.reduce_map[a] != self.reduce_map[b]
            and a != reps[int(self.reduce_map[a])]
            and b != reps[int(self.reduce_map[b])]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nd_red = NMRData2D(self.atoms, **self.kw, reduce=True, pairs=[(a, b)])

        self.assertEqual(nd_red.src_pairs, [(a, b)])
        self.assertAlmostEqual(
            nd_red.pair_distances[0], self.atoms.get_distance(a, b, mic=True), places=6,
        )

    def test_reduced_pair_sites_contain_the_input_atoms(self):
        """The derived pair's sites contain the input atoms, through site_map."""
        a, b = self._pair_on_different_sites()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nd = NMRData2D(self.atoms, **self.kw, reduce=True, pairs=[(a, b)])

        i, j = nd.pairs[0]
        self.assertIn(a, nd.site_map.members(i).tolist())
        self.assertIn(b, nd.site_map.members(j).tolist())


if __name__ == "__main__":
    unittest.main()
