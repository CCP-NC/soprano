#!/usr/bin/env python
"""
Test code for NMR calculations - i.e. in the calculate/nmr directory
"""


import os
import unittest
from typing import List, Tuple

import numpy as np
from ase.build import bulk, molecule

from soprano.calculate.nmr.nmr import (
    Peak2D,
    calculate_distances,
    get_pair_dipolar_couplings,
    merge_peaks,
)
from soprano.calculate.nmr.utils import _build_equiv_groups, _nearest_equiv_site

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")




class TestMergePeaks(unittest.TestCase):

    def setUp(self):
        # Create some sample peaks for testing
        self.peaks: List[Peak2D] = [
            Peak2D(x=1.0, y=2.0000001, correlation_strength=0.5, xlabel='H1', ylabel='H2', idx_x=0, idx_y=1, color='red'),
            Peak2D(x=1.0, y=2.0000001, correlation_strength=0.5, xlabel='H1', ylabel='H2', idx_x=0, idx_y=1, color='red'),
            Peak2D(x=1.0, y=2.0999999, correlation_strength=0.5, xlabel='H1', ylabel='H3', idx_x=0, idx_y=2, color='blue'),
            Peak2D(x=1.2, y=2.2000003, correlation_strength=0.5, xlabel='H2', ylabel='H3', idx_x=1, idx_y=0, color='green'),
            Peak2D(x=1.2, y=2.2000004, correlation_strength=0.7, xlabel='H2', ylabel='H3', idx_x=2, idx_y=3, color='yellow')
        ]

    def test_merge_identical_peaks(self):
        merged_peaks = merge_peaks(self.peaks, xtol=1e-5, ytol=1e-5, corr_tol=1e-5, ignore_correlation_strength=False)
        self.assertEqual(len(merged_peaks), 4)
        self.assertEqual(merged_peaks[0].xlabel, 'H1')
        self.assertEqual(merged_peaks[0].ylabel, 'H2')

    def test_merge_with_tolerance(self):
        # peaks 0, 1, 2 should be merged
        merged_peaks = merge_peaks(self.peaks, xtol=0.1, ytol=0.1, corr_tol=0.1, ignore_correlation_strength=False)
        print(merged_peaks)
        self.assertEqual(len(merged_peaks), 3)
        self.assertEqual(merged_peaks[0].xlabel, 'H1')
        self.assertEqual(merged_peaks[0].ylabel, 'H2/H3')

    def test_merge_ignore_correlation_strength(self):
        # peaks 0, 1 should be merged and 3 and 4 should be merged
        merged_peaks = merge_peaks(self.peaks, xtol=1e-5, ytol=1e-5, corr_tol=1e-5, ignore_correlation_strength=True)
        self.assertEqual(len(merged_peaks), 3)
        self.assertEqual(merged_peaks[0].xlabel, 'H1')
        self.assertEqual(merged_peaks[0].ylabel, 'H2')
        self.assertEqual(merged_peaks[2].xlabel, 'H2')
        self.assertEqual(merged_peaks[2].ylabel, 'H3')


class TestBuildEquivGroups(unittest.TestCase):
    """Direct tests for _build_equiv_groups."""

    def setUp(self):
        # NaCl rocksalt cubic (8 atoms: 4 Na + 4 Cl)
        self.nacl = bulk('NaCl', 'rocksalt', a=5.64, cubic=True)

    def test_nacl_two_groups(self):
        """All Na and all Cl should each form one equivalence group."""
        groups = _build_equiv_groups(self.nacl)
        # Exactly two groups: one for Na, one for Cl
        self.assertEqual(len(groups), 2)
        all_indices = sorted(idx for members in groups.values() for idx in members)
        self.assertEqual(all_indices, list(range(len(self.nacl))))

    def test_nacl_group_sizes(self):
        """Each group should contain 4 atoms (4 Na and 4 Cl in the cubic cell)."""
        groups = _build_equiv_groups(self.nacl)
        sizes = sorted(len(v) for v in groups.values())
        self.assertEqual(sizes, [4, 4])

    def test_nacl_groups_are_element_pure(self):
        """Every group should contain only one element type."""
        symbols = np.array(self.nacl.get_chemical_symbols())
        groups = _build_equiv_groups(self.nacl)
        for members in groups.values():
            elems_in_group = set(symbols[members])
            self.assertEqual(len(elems_in_group), 1,
                             f"Group {members} mixes elements: {elems_in_group}")

    def test_molecule_fallback_each_atom_own_group(self):
        """Non-periodic structures (spglib fails) → each atom in its own group."""
        mol = molecule('H2O')
        mol.pbc = False
        groups = _build_equiv_groups(mol)
        self.assertEqual(len(groups), len(mol))


class TestNearestEquivSite(unittest.TestCase):
    """Direct tests for _nearest_equiv_site."""

    def setUp(self):
        self.nacl = bulk('NaCl', 'rocksalt', a=5.64, cubic=True)
        self.groups = _build_equiv_groups(self.nacl)
        # FCC nearest-neighbour distance for this cell
        self.expected_dist = 5.64 / np.sqrt(2)

    def test_nearest_equiv_is_same_element(self):
        """Nearest equivalent of Na[0] should itself be a Na atom."""
        symbols = self.nacl.get_chemical_symbols()
        partner, _ = _nearest_equiv_site(self.nacl, 0, self.groups)
        self.assertEqual(symbols[partner], 'Na')

    def test_nearest_equiv_distance_na(self):
        """Nearest equivalent of Na[0] → a/√2 (FCC in-plane neighbour)."""
        _, dist = _nearest_equiv_site(self.nacl, 0, self.groups)
        self.assertAlmostEqual(dist, self.expected_dist, places=4)

    def test_nearest_equiv_distance_cl(self):
        """Nearest equivalent of Cl[1] → same a/√2 by symmetry."""
        _, dist = _nearest_equiv_site(self.nacl, 1, self.groups)
        self.assertAlmostEqual(dist, self.expected_dist, places=4)

    def test_nearest_equiv_consistent_with_explicit_pair(self):
        """Self-pair distance should equal the distance to the explicit nearest-neighbour pair."""
        partner, dist_self = _nearest_equiv_site(self.nacl, 1, self.groups)
        dist_explicit = float(self.nacl.get_distance(1, partner, mic=True))
        self.assertAlmostEqual(dist_self, dist_explicit, places=10)

    def test_nearest_equiv_not_self_when_equiv_exists(self):
        """When other equivalent atoms exist in the cell, partner index should differ."""
        partner, _ = _nearest_equiv_site(self.nacl, 0, self.groups)
        # The partner is a different atom reached directly, not idx itself via a PBC image
        other_na = [i for i in self.groups[list(self.groups.keys())[0]] if i != 0]
        self.assertIn(partner, other_na)


class DipolarCoupling:
    @staticmethod
    def get(atoms, sel_i, sel_j, isotopes):
        # Mock implementation of DipolarCoupling.get
        return {0: {0: 1.0}}

class TestGetPairDipolarCouplings(unittest.TestCase):

    def setUp(self):
        # Create a mock Atoms object
        self.atoms = bulk('NaCl', 'rocksalt', a=5.64, cubic=True)
        self.pairs: List[Tuple[int, int]] = [(0, 1), # Na-Cl
                                             (1, 1), # Same atom -> nearest equiv (Cl-Cl) coupling
                                             (0, 2), # Na-Na
                                             (1, 3)  # Cl-Cl (same distance as self-pair above)
                                             ]
        self.isotopes = {'Na': 23, 'Cl': 35}
        self.reference_dipolar_couplings = [-0.139069188,
                                            -0.018222048,  # Cl to nearest equiv Cl (same as Cl-Cl pair)
                                            -0.132670595,
                                            -0.018222048]

    def test_dipolar_couplings(self):
        dipolar_couplings = get_pair_dipolar_couplings(self.atoms, self.pairs, self.isotopes)
        self.assertTrue(np.allclose(dipolar_couplings, self.reference_dipolar_couplings))

    def test_dipolar_couplings_no_isotopes(self):
        # Same as before since those were the default isotopes...
        dipolar_couplings = get_pair_dipolar_couplings(self.atoms, self.pairs)
        self.assertTrue(np.allclose(dipolar_couplings, self.reference_dipolar_couplings))

    def test_dipolar_couplings_zero_for_same_indices(self):
        """Self-pairs return coupling to nearest equivalent site (non-zero for periodic crystals)."""
        pairs_with_same_indices = [(0, 0), (1, 1), (2, 2)]
        dipolar_couplings = get_pair_dipolar_couplings(self.atoms, pairs_with_same_indices, self.isotopes)
        # All couplings should be non-zero (nearest equiv site distances are finite)
        self.assertTrue(all(abs(d) > 0 for d in dipolar_couplings))
        # Self-pair for Cl (atom 1) should equal the explicit nearest Cl-Cl pair (1,3)
        d_self = get_pair_dipolar_couplings(self.atoms, [(1, 1)], self.isotopes)[0]
        d_explicit = get_pair_dipolar_couplings(self.atoms, [(1, 3)], self.isotopes)[0]
        self.assertAlmostEqual(d_self, d_explicit, places=6)

    def test_dipolar_couplings_use_equiv_sites_false(self):
        """use_equiv_sites=False: self-pairs use self_coupling (nearest periodic image)."""
        # Non-self pairs must be identical regardless of the flag.
        non_self_pairs = [(0, 1), (0, 2)]
        d_on  = get_pair_dipolar_couplings(self.atoms, non_self_pairs, self.isotopes, use_equiv_sites=True)
        d_off = get_pair_dipolar_couplings(self.atoms, non_self_pairs, self.isotopes, use_equiv_sites=False)
        self.assertTrue(np.allclose(d_on, d_off))

        # For self-pairs the two modes can differ when there is a closer
        # equivalent atom in the cell than the nearest periodic image.
        # The default (use_equiv_sites=True) should yield a coupling at least
        # as strong (shorter or equal distance) as the self-image version.
        d_equiv = get_pair_dipolar_couplings(self.atoms, [(1, 1)], self.isotopes, use_equiv_sites=True)[0]
        d_self  = get_pair_dipolar_couplings(self.atoms, [(1, 1)], self.isotopes, use_equiv_sites=False)[0]
        # Both must be non-zero
        self.assertGreater(abs(d_equiv), 0)
        self.assertGreater(abs(d_self), 0)
        # equiv-site coupling >= self-coupling in magnitude (shorter distance → stronger coupling)
        self.assertGreaterEqual(abs(d_equiv), abs(d_self) - 1e-9)


class TestCalculateDistances(unittest.TestCase):

    def setUp(self):
        self.atoms = bulk('NaCl', 'rocksalt', a=5.64, cubic=True)
        # FCC nearest-neighbour distance
        self.a_over_sqrt2 = 5.64 / np.sqrt(2)

    def test_non_self_pairs_unchanged_by_flag(self):
        """use_equiv_sites flag must not affect non-self pairs."""
        pairs = [(0, 1), (0, 2), (1, 3)]
        d_on  = calculate_distances(pairs, self.atoms, use_equiv_sites=True)
        d_off = calculate_distances(pairs, self.atoms, use_equiv_sites=False)
        self.assertTrue(np.allclose(d_on, d_off))

    def test_self_pairs_equiv_sites_true(self):
        """Default mode: self-pair distance = nearest equivalent site."""
        pairs = [(1, 1)]
        dist = calculate_distances(pairs, self.atoms, use_equiv_sites=True)[0]
        self.assertAlmostEqual(dist, self.a_over_sqrt2, places=4)

    def test_self_pairs_equiv_sites_false(self):
        """use_equiv_sites=False: self-pair uses the nearest periodic image."""
        pairs = [(1, 1)]
        dist = calculate_distances(pairs, self.atoms, use_equiv_sites=False)[0]
        # For the rocksalt cubic cell the nearest Cl periodic image is also a_over_sqrt2
        # but this path must not raise and must return a positive finite distance.
        self.assertGreater(dist, 0)
        self.assertTrue(np.isfinite(dist))

if __name__ == '__main__':
    unittest.main()
