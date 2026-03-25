#!/usr/bin/env python
"""
Test code for NMR calculations - i.e. in the calculate/nmr directory
"""


import os
import unittest
from typing import List, Tuple

import numpy as np
from ase import io
from ase.build import bulk, molecule

from soprano.calculate.nmr.nmr import (
    NMRData2D,
    Peak2D,
    calculate_distances,
    get_pair_dipolar_couplings,
    merge_peaks,
)
from soprano.calculate.nmr.utils import _build_equiv_groups, _nearest_equiv_site, generate_peaks, get_atom_labels
from soprano.properties.nmr import DipolarRSSByAtom

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


class TestDipolarRSSMetric(unittest.TestCase):
    """Tests for the 'dipolar_rss' correlation_strength_metric in NMRData2D."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

    def _make_hc(self, **kwargs):
        """Helper: H–C 2D plot with dipolar_rss metric."""
        return NMRData2D(
            atoms=self.atoms,
            xelement="H",
            yelement="C",
            correlation_strength_metric="dipolar_rss",
            **kwargs,
        )

    def test_basic_produces_positive_strengths(self):
        """All dipolar_rss correlation strengths should be non-negative."""
        d = self._make_hc(rss_cutoff=5.0)
        self.assertGreater(len(d.peaks), 0)
        self.assertTrue(all(p.correlation_strength >= 0 for p in d.peaks))

    def test_peak_strength_matches_direct_rss(self):
        """Each peak's correlation_strength must equal DipolarRSSByAtom directly."""
        d = self._make_hc(rss_cutoff=10.0, rss_expand_j="periodic_images")
        for peak in d.peaks:
            expected = DipolarRSSByAtom.get(
                self.atoms,
                sel_i=[peak.idx_x],
                sel_j=[peak.idx_y],
                cutoff=10.0,
                expand_j="periodic_images",
            )[0]
            self.assertAlmostEqual(peak.correlation_strength, expected, places=6)

    def test_smaller_cutoff_gives_smaller_or_equal_rss(self):
        """RSS is monotonically non-decreasing with cutoff.

        This is a universal property: a larger cutoff includes all periodic
        images from the smaller cutoff plus potentially more, so adding further
        non-negative squared terms can only increase (or leave unchanged) the
        RSS. It does not depend on the specific structure under test.
        """
        d_large = self._make_hc(rss_cutoff=10.0, rss_expand_j="periodic_images")
        d_small = self._make_hc(rss_cutoff=3.0, rss_expand_j="periodic_images")

        large = {(p.idx_x, p.idx_y): p.correlation_strength for p in d_large.peaks}
        small = {(p.idx_x, p.idx_y): p.correlation_strength for p in d_small.peaks}

        # Monotonicity: every pair must satisfy rss(3 Å) ≤ rss(10 Å)
        for key in small:
            if key in large:
                self.assertLessEqual(small[key], large[key] + 1e-9)

        # Sanity check: the larger cutoff must be strictly larger for at least
        # one pair, confirming the cutoff is actually doing something.
        self.assertTrue(
            any(large[k] > small[k] + 1e-9 for k in small if k in large),
            "Expected at least one pair to have a larger RSS with cutoff=10 Å vs 3 Å"
        )

    def test_rss_metric_differs_from_dipolar_metric(self):
        """dipolar_rss and dipolar metrics should generally differ (RSS > single coupling)."""
        d_rss = self._make_hc(rss_cutoff=5.0)
        d_dip = NMRData2D(
            atoms=self.atoms,
            xelement="H",
            yelement="C",
            correlation_strength_metric="dipolar",
        )
        rss_map = {(p.idx_x, p.idx_y): p.correlation_strength for p in d_rss.peaks}
        dip_map = {(p.idx_x, p.idx_y): p.correlation_strength for p in d_dip.peaks}

        # For at least some pair, the RSS (summing all periodic images) should
        # be larger than the single-nearest-image dipolar coupling magnitude.
        common_keys = set(rss_map) & set(dip_map)
        self.assertTrue(
            any(rss_map[k] > abs(dip_map[k]) for k in common_keys),
            "Expected dipolar_rss > |dipolar| for at least one pair"
        )

    def test_expand_j_symmetry_expands_sel_j(self):
        """RSS with expand_j='symmetry' should be >= 'periodic_images' for same pair."""
        d_pi = self._make_hc(rss_cutoff=5.0, rss_expand_j="periodic_images")
        d_sym = self._make_hc(rss_cutoff=5.0, rss_expand_j="symmetry")

        pi_map = {(p.idx_x, p.idx_y): p.correlation_strength for p in d_pi.peaks}
        sym_map = {(p.idx_x, p.idx_y): p.correlation_strength for p in d_sym.peaks}

        # Symmetry expansion can only add neighbours, so RSS can only increase
        for key in pi_map:
            if key in sym_map:
                self.assertGreaterEqual(sym_map[key], pi_map[key] - 1e-9)

    def test_dipolar_rss_in_marker_info(self):
        """'dipolar_rss' should be registered in MARKER_INFO with expected fields."""
        from soprano.calculate.nmr.nmr import MARKER_INFO
        self.assertIn("dipolar_rss", MARKER_INFO)
        self.assertIn("label", MARKER_INFO["dipolar_rss"])
        self.assertIn("unit", MARKER_INFO["dipolar_rss"])
        self.assertEqual(MARKER_INFO["dipolar_rss"]["unit"], "kHz")

    def test_correlation_unit_and_label_set_on_instance(self):
        """NMRData2D should expose correct unit and label for dipolar_rss."""
        d = self._make_hc(rss_cutoff=5.0)
        self.assertEqual(d.correlation_unit, "kHz")
        self.assertIn("RSS", d.correlation_label)


class TestDQSQSpectrum(unittest.TestCase):
    """Tests for Double Quantum / Single Quantum (DQ/SQ) 2D spectra.

    DQ/SQ properties under test:
    - xelement == yelement (homonuclear)
    - yaxis_order='2Q': y-axis is the double quantum frequency σ_i + σ_j
    - x-axis is the single quantum frequency σ_i
    - Self-pairs (i, i) are removed: zero inter-nuclear distance is unphysical
    - Pair (i, j) and (j, i) form a "roof": two SQ peaks at different x-positions
      but the same DQ y-value, with identical ylabel labels
    """

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

    def _make_dqsq(self, **kwargs):
        """H–H DQ/SQ 2D spectrum (fixed correlation strength by default)."""
        return NMRData2D(
            atoms=self.atoms,
            xelement="H",
            yelement="H",
            yaxis_order="2Q",
            **kwargs,
        )

    def _unmerged_peaks(self, d):
        """Return the unmerged peak list for an already-constructed NMRData2D."""
        labels = get_atom_labels(d.atoms)
        return generate_peaks(
            d.data, d.pairs, labels, d.correlation_strengths, "2Q", "H", "H",
        )

    def test_dqsq_no_self_pairs(self):
        """Self-pairs (idx_x == idx_y) are unconditionally removed in DQ/SQ."""
        d = self._make_dqsq()
        for peak in d.peaks:
            self.assertNotEqual(
                peak.idx_x, peak.idx_y,
                f"Self-pair found: idx={peak.idx_x}",
            )

    def test_dqsq_y_equals_sum_of_shieldings(self):
        """Each peak y must equal σ_x + σ_y (the double quantum frequency)."""
        d = self._make_dqsq()
        for peak in self._unmerged_peaks(d):
            expected_y = d.data[peak.idx_x] + d.data[peak.idx_y]
            self.assertAlmostEqual(
                peak.y, expected_y, places=6,
                msg=(
                    f"y={peak.y:.4f} != "
                    f"σ[{peak.idx_x}]+σ[{peak.idx_y}]={expected_y:.4f}"
                ),
            )

    def test_dqsq_x_is_single_atom_shielding(self):
        """Each peak x must equal the SQ shielding of atom idx_x only."""
        d = self._make_dqsq()
        for peak in self._unmerged_peaks(d):
            self.assertAlmostEqual(
                peak.x, d.data[peak.idx_x], places=6,
                msg=f"x={peak.x:.4f} != σ[{peak.idx_x}]={d.data[peak.idx_x]:.4f}",
            )

    def test_dqsq_y_axis_label_contains_2q(self):
        """The auto-generated y-axis label must contain '2Q'."""
        d = self._make_dqsq()
        self.assertIn("2Q", d.y_axis_label)

    def test_dqsq_roof_y_values_match(self):
        """Roofed peaks (i,j) and (j,i) must share the same y-value.

        y = σ_i + σ_j is symmetric, so both orientations of a pair land on
        the same DQ frequency regardless of which atom is i or j.
        """
        from collections import defaultdict

        d = self._make_dqsq()
        y_per_pair = defaultdict(list)
        for peak in self._unmerged_peaks(d):
            key = (min(peak.idx_x, peak.idx_y), max(peak.idx_x, peak.idx_y))
            y_per_pair[key].append(peak.y)

        for key, ys in y_per_pair.items():
            if len(ys) == 2:
                self.assertAlmostEqual(
                    ys[0], ys[1], places=6,
                    msg=f"Roof y-mismatch for pair {key}: {ys[0]:.4f} vs {ys[1]:.4f}",
                )

    def test_dqsq_roof_ylabels_match(self):
        """Roofed peaks (i,j) and (j,i) must share the same ylabel.

        Homonuclear DQ/SQ uses a sorted label combination (e.g. 'H1 + H2')
        so that merge_peaks can recognise both roof peaks as the same DQ
        correlation and/or plot them with the correct shared label.
        """
        from collections import defaultdict

        d = self._make_dqsq()
        ylabels_per_pair = defaultdict(set)
        for peak in self._unmerged_peaks(d):
            key = (min(peak.idx_x, peak.idx_y), max(peak.idx_x, peak.idx_y))
            ylabels_per_pair[key].add(peak.ylabel)

        for key, label_set in ylabels_per_pair.items():
            self.assertEqual(
                len(label_set), 1,
                f"Pair {key} has non-canonical ylabels: {label_set}",
            )


class TestDQSQWithCIFLabels(unittest.TestCase):
    """DQ/SQ tests using EDIZUM.magres, which has CIF-style labels with Z=4.

    EDIZUM has 4 crystallographically equivalent copies of each unique site.
    This class verifies two things:

    1. The refactored _expand_sel_j_by_equiv helper (used inside
       DipolarRSSByAtom) gives the same result as the previous per-branch
       implementation — all equivalent pairs produce identical RSS.

    2. In a DQ/SQ NMRData2D spectrum, expand_j='cif_labels' correctly accounts
       for in-cell equivalent neighbours (not just periodic images), giving a
       larger RSS for intra-group pairs than periodic_images alone.
    """

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        labels = self.atoms.get_array("labels")
        syms = self.atoms.get_chemical_symbols()
        self.h1_idx = [i for i in range(len(self.atoms))
                       if syms[i] == "H" and labels[i] == "H1"]
        self.h2_idx = [i for i in range(len(self.atoms))
                       if syms[i] == "H" and labels[i] == "H2"]

    # ------------------------------------------------------------------
    # Direct DipolarRSSByAtom tests (exercising the refactored helper)
    # ------------------------------------------------------------------

    def test_equiv_h1h2_pairs_give_same_rss_cif(self):
        """All (H1_a, H2_b) pairs give identical RSS with expand_j='cif_labels'.

        When sel_j=[H2_b] is expanded to all H2 atoms and sel_i=H1_a is one
        of 4 equivalent H1 sites, every combination (H1_a, H2_b) sees the
        same neighbourhood by crystallographic symmetry.  The RSS must be
        the same for all 4×4 = 16 H1/H2 combinations.
        """
        values = [
            DipolarRSSByAtom.get(
                self.atoms,
                sel_i=[i], sel_j=[j],
                cutoff=5.0, expand_j="cif_labels",
            )[0]
            for i in self.h1_idx
            for j in self.h2_idx
        ]
        self.assertGreater(values[0], 0,
                           "Expected non-zero RSS for H1-H2 pair")
        for v in values[1:]:
            self.assertAlmostEqual(
                v, values[0], places=4,
                msg=f"H1-H2 RSS not equal across equivalent pairs: {v} vs {values[0]}",
            )

    def test_equiv_h1h1_pairs_give_same_rss_cif(self):
        """All non-self (H1_a, H1_b) pairs give identical RSS with cif_labels.

        The CIF expansion adds H1_0, H1_1, H1_2, H1_3 as neighbours for any
        seed H1_x, so every pair within the group sees the same environment.
        """
        values = [
            DipolarRSSByAtom.get(
                self.atoms,
                sel_i=[i], sel_j=[j],
                cutoff=6.0, expand_j="cif_labels",
            )[0]
            for i in self.h1_idx
            for j in self.h1_idx
            if i != j
        ]
        self.assertGreater(values[0], 0,
                           "Expected non-zero RSS for H1-H1 pair")
        for v in values[1:]:
            self.assertAlmostEqual(
                v, values[0], places=4,
                msg=f"H1-H1 RSS not equal across equivalent pairs: {v} vs {values[0]}",
            )

    def test_cif_expansion_gt_periodic_images_for_intragroup(self):
        """For an intra-group H1-H1 pair, cif_labels RSS > periodic_images RSS.

        With periodic_images only the single in-cell atom H1_1 (and its
        periodic copies) is counted.  With cif_labels, H1_0, H1_2, H1_3 are
        also added as in-cell neighbours, strictly increasing the RSS.
        """
        h1_a, h1_b = self.h1_idx[0], self.h1_idx[1]
        rss_cif = DipolarRSSByAtom.get(
            self.atoms, sel_i=[h1_a], sel_j=[h1_b],
            cutoff=6.0, expand_j="cif_labels",
        )[0]
        rss_pi = DipolarRSSByAtom.get(
            self.atoms, sel_i=[h1_a], sel_j=[h1_b],
            cutoff=6.0, expand_j="periodic_images",
        )[0]
        self.assertGreater(
            rss_cif, rss_pi,
            "Expected cif_labels RSS > periodic_images RSS for an intragroup pair",
        )

    # ------------------------------------------------------------------
    # NMRData2D integration: DQ/SQ with dipolar_rss + cif_labels
    # ------------------------------------------------------------------

    def test_dqsq_nmrdata2d_equiv_pairs_equal_strength(self):
        """NMRData2D DQ/SQ with dipolar_rss+cif_labels: equiv pairs get same strength.

        Passes explicit pairs so the test is fast (4 peaks only).  Verifies
        that (H1_0,H1_1) and (H1_2,H1_3) have identical correlation_strength
        and that (H1_0,H2_4) and (H1_1,H2_5) have identical strength.
        """
        h1_0, h1_1, h1_2, h1_3 = self.h1_idx[:4]
        h2_0, h2_1 = self.h2_idx[:2]
        explicit_pairs = [
            (h1_0, h1_1), (h1_2, h1_3),   # intra-group H1-H1 equiv pair
            (h1_0, h2_0), (h1_1, h2_1),   # inter-group H1-H2 equiv pair
        ]
        d = NMRData2D(
            atoms=self.atoms,
            xelement="H", yelement="H",
            yaxis_order="2Q",
            correlation_strength_metric="dipolar_rss",
            rss_expand_j="cif_labels",
            rss_cutoff=6.0,
            pairs=explicit_pairs,
        )
        # Use d.pairs / d.correlation_strengths (pre-merge) rather than d.peaks,
        # because equivalent pairs have the same (x, y) and get merged into one
        # peak when all H1 atoms are crystallographically equivalent.
        pair_strengths = dict(zip(d.pairs, d.correlation_strengths))
        self.assertAlmostEqual(
            pair_strengths[(h1_0, h1_1)], pair_strengths[(h1_2, h1_3)], places=4,
            msg="Equiv H1-H1 pairs should have the same DQ correlation strength",
        )
        self.assertAlmostEqual(
            pair_strengths[(h1_0, h2_0)], pair_strengths[(h1_1, h2_1)], places=4,
            msg="Equiv H1-H2 pairs should have the same DQ correlation strength",
        )


if __name__ == '__main__':
    unittest.main()
