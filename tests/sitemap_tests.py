#!/usr/bin/env python
"""Tests for SiteMap, the atom-to-site index correspondence.

Two things are worth checking beyond the obvious algebra:

- the constructors must reproduce what the operations they describe actually
  do, so ``from_selection`` is checked against ``AtomSelection.subset`` and
  ``from_tags`` against ``merge_tagged_sites``;
- composition must survive a real reduce-then-filter pipeline, which is the
  case that motivated the class.
"""

import os
import sys
import unittest

import numpy as np
from ase import Atoms
from ase import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from soprano.selection import AtomSelection  # noqa: E402
from soprano.sitemap import DROPPED, SiteMap  # noqa: E402

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestSiteMapConstruction(unittest.TestCase):
    def test_identity(self):
        m = SiteMap.identity(4)
        self.assertEqual((m.n_atoms, m.n_sites), (4, 4))
        np.testing.assert_array_equal(m.to_sites, [0, 1, 2, 3])
        self.assertEqual(len(m), 4)

    def test_identity_empty(self):
        m = SiteMap.identity(0)
        self.assertEqual((m.n_atoms, m.n_sites), (0, 0))

    def test_from_tags_groups_by_sorted_unique_tag(self):
        m = SiteMap.from_tags([7, 3, 7, 3, 9])
        # sorted unique tags are 3, 7, 9 -> sites 0, 1, 2
        np.testing.assert_array_equal(m.to_sites, [1, 0, 1, 0, 2])
        self.assertEqual(m.n_sites, 3)

    def test_from_tags_handles_negative_tags(self):
        """Functional-group tagging uses large negative tags."""
        m = SiteMap.from_tags([-100000, -100000, 0, 1])
        np.testing.assert_array_equal(m.to_sites, [0, 0, 1, 2])
        self.assertEqual(m.n_sites, 3)

    def test_from_selection_accepts_atoms_or_length(self):
        atoms = Atoms("H4")
        sel = AtomSelection(atoms, [2, 0])
        by_atoms = SiteMap.from_selection(sel, atoms)
        by_length = SiteMap.from_selection(sel, 4)
        np.testing.assert_array_equal(by_atoms.to_sites, by_length.to_sites)

    def test_from_selection_preserves_selection_order(self):
        """Selection order is not guaranteed sorted, so the map must follow it."""
        atoms = Atoms("H4")
        sel = AtomSelection(atoms, [3, 1])
        m = SiteMap.from_selection(sel, atoms)
        self.assertEqual(m.members(0).tolist(), [3])
        self.assertEqual(m.members(1).tolist(), [1])
        np.testing.assert_array_equal(m.to_sites, [DROPPED, 1, DROPPED, 0])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            SiteMap(np.array([[0, 1]]), 2)  # not 1-D
        with self.assertRaises(ValueError):
            SiteMap(np.array([0, 1]), -1)  # negative n_sites
        with self.assertRaises(ValueError):
            SiteMap(np.array([0, 5]), 2)  # site index out of range
        with self.assertRaises(ValueError):
            SiteMap(np.array([0, -2]), 2)  # below the DROPPED sentinel
        with self.assertRaises(ValueError):
            SiteMap.from_selection(AtomSelection(Atoms("H2"), [1]), 1)

    def test_to_sites_is_read_only(self):
        m = SiteMap.identity(3)
        with self.assertRaises(ValueError):
            m.to_sites[0] = 2


class TestSiteMapUse(unittest.TestCase):
    def test_members(self):
        m = SiteMap.from_tags([0, 1, 0, 0])
        self.assertEqual(m.members(0).tolist(), [0, 2, 3])
        self.assertEqual(m.members(1).tolist(), [1])

    def test_members_rejects_out_of_range(self):
        m = SiteMap.identity(2)
        for bad in (-1, 2):
            with self.assertRaises(IndexError):
                m.members(bad)

    def test_compose(self):
        first = SiteMap.from_tags([0, 0, 1, 2])          # 4 atoms -> 3 sites
        second = SiteMap(np.array([DROPPED, 0, 1]), 2)   # 3 sites -> 2 sites
        combined = first.compose(second)
        np.testing.assert_array_equal(combined.to_sites, [DROPPED, DROPPED, 0, 1])
        self.assertEqual((combined.n_atoms, combined.n_sites), (4, 2))

    def test_compose_rejects_mismatched_spaces(self):
        with self.assertRaises(ValueError):
            SiteMap.identity(4).compose(SiteMap.identity(3))

    def test_compose_with_identity_is_a_no_op(self):
        m = SiteMap.from_tags([0, 0, 1])
        np.testing.assert_array_equal(
            m.compose(SiteMap.identity(m.n_sites)).to_sites, m.to_sites
        )
        np.testing.assert_array_equal(
            SiteMap.identity(m.n_atoms).compose(m).to_sites, m.to_sites
        )

    def test_compose_is_associative(self):
        a = SiteMap.from_tags([0, 0, 1, 2, 2])
        b = SiteMap(np.array([0, DROPPED, 1]), 2)
        c = SiteMap(np.array([DROPPED, 0]), 1)
        np.testing.assert_array_equal(
            a.compose(b).compose(c).to_sites, a.compose(b.compose(c)).to_sites
        )

    def test_validate(self):
        m = SiteMap.identity(3)
        m.validate(Atoms("H3"))
        with self.assertRaises(ValueError):
            m.validate(Atoms("H4"))


class TestSiteMapAgainstRealOperations(unittest.TestCase):
    """The constructors claim to describe real operations. Check that they do."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))

    def test_from_selection_matches_subset(self):
        sel = AtomSelection.from_element(self.atoms, "H")
        subset = sel.subset(self.atoms)
        smap = SiteMap.from_selection(sel, self.atoms)

        self.assertEqual(smap.n_sites, len(subset))
        for k in range(len(subset)):
            (atom,) = smap.members(k)
            self.assertEqual(self.atoms[atom].symbol, subset[k].symbol)
            np.testing.assert_allclose(
                self.atoms.positions[atom], subset[k].position
            )

    def test_from_tags_matches_nmr_extract_atoms_index_map(self):
        from soprano.nmr.extract import label_atoms, nmr_extract_atoms

        src = label_atoms(self.atoms.copy())
        reduced, index_map = nmr_extract_atoms(
            src.copy(), reduce=True, symprec=1e-4, return_index_map=True
        )
        smap = SiteMap(np.asarray(index_map), len(reduced))

        self.assertEqual(smap.n_sites, len(reduced))
        # Every atom of a site must agree with that site on species and label.
        src_labels = np.asarray(src.get_array("labels"))
        red_labels = np.asarray(reduced.get_array("labels"))
        for k in range(len(reduced)):
            members = smap.members(k)
            self.assertGreater(len(members), 0)
            for i in members:
                self.assertEqual(src[i].symbol, reduced[k].symbol)
                self.assertEqual(src_labels[i], red_labels[k])

    def test_members_reproduces_the_label_join_it_replaced(self):
        """NMRData2D used to recover site membership by matching CIF labels."""
        from soprano.nmr.extract import label_atoms, nmr_extract_atoms

        src = label_atoms(self.atoms.copy())
        reduced, index_map = nmr_extract_atoms(
            src.copy(), reduce=True, symprec=1e-4, return_index_map=True
        )
        smap = SiteMap(np.asarray(index_map), len(reduced))
        src_labels = np.asarray(src.get_array("labels"))
        red_labels = np.asarray(reduced.get_array("labels"))

        for k in range(len(reduced)):
            by_label = np.flatnonzero(src_labels == red_labels[k])
            np.testing.assert_array_equal(sorted(smap.members(k)), sorted(by_label))

    def test_compose_survives_reduce_then_filter(self):
        """The pipeline NMRData2D runs: reduce, then keep one element."""
        from soprano.nmr.extract import label_atoms, nmr_extract_atoms

        src = label_atoms(self.atoms.copy())
        reduced, index_map = nmr_extract_atoms(
            src.copy(), reduce=True, symprec=1e-4, return_index_map=True
        )
        sel = AtomSelection.from_element(reduced, "H")
        final = sel.subset(reduced)

        combined = SiteMap(np.asarray(index_map), len(reduced)).compose(
            SiteMap.from_selection(sel, reduced)
        )

        self.assertEqual(combined.n_atoms, len(src))
        self.assertEqual(combined.n_sites, len(final))
        combined.validate(src)
        for k in range(len(final)):
            members = combined.members(k)
            self.assertGreater(len(members), 0)
            for i in members:
                self.assertEqual(src[i].symbol, final[k].symbol)
        # Nothing but hydrogen survives.
        kept = np.flatnonzero(combined.to_sites != DROPPED)
        self.assertTrue(all(src[i].symbol == "H" for i in kept))


if __name__ == "__main__":
    unittest.main()
