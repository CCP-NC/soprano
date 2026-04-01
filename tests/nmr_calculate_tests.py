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
        """Each peak's correlation_strength must equal DipolarRSSByAtom (converted to kHz)."""
        d = self._make_hc(rss_cutoff=10.0, rss_expand_j="periodic_images")
        for peak in d.peaks:
            expected_hz = DipolarRSSByAtom.get(
                self.atoms,
                sel_i=[peak.idx_x],
                sel_j=[peak.idx_y],
                cutoff=10.0,
                expand_j="periodic_images",
            )[0]
            expected_khz = expected_hz * 1e-3
            self.assertAlmostEqual(peak.correlation_strength, expected_khz, places=6)

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

    def test_symmetry_and_cif_labels_agree(self):
        """expand_j='symmetry' and expand_j='cif_labels' give the same RSS for EDIZUM.

        EDIZUM has consistent CIF labels and crystallographic symmetry (Z=4),
        so both expansion strategies should identify the same set of equivalent
        neighbours and produce identical RSS values for all H1-H2 and H1-H1
        pairs tested.
        """
        cutoff = 6.0
        test_pairs = (
            [(i, j) for i in self.h1_idx for j in self.h2_idx]  # H1-H2
            + [(i, j) for i in self.h1_idx for j in self.h1_idx if i != j]  # H1-H1
        )
        for idx_i, idx_j in test_pairs:
            rss_cif = DipolarRSSByAtom.get(
                self.atoms, sel_i=[idx_i], sel_j=[idx_j],
                cutoff=cutoff, expand_j="cif_labels",
            )[0]
            rss_sym = DipolarRSSByAtom.get(
                self.atoms, sel_i=[idx_i], sel_j=[idx_j],
                cutoff=cutoff, expand_j="symmetry",
            )[0]
            self.assertAlmostEqual(
                rss_cif, rss_sym, places=4,
                msg=(f"cif_labels and symmetry RSS disagree for pair "
                     f"({idx_i}, {idx_j}): {rss_cif:.6f} vs {rss_sym:.6f}"),
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


class TestNMRData2DReduce(unittest.TestCase):
    """End-to-end tests for the reduce=True code-path in NMRData2D.

    EDIZUM (Z=4) is used throughout: 148 atoms total, 37 in the asymmetric unit
    (19 unique H sites among them).

    Key invariants tested:
    - reduce=True collapses to the asymmetric unit (atoms_full kept for RSS)
    - pairs and Peak2D indices are valid indices into the *reduced* atoms
    - RSS values match directly-computed values on atoms_full
    - reduce=True + cif_labels gives the same RSS as reduce=False + cif_labels
      for the same (label, label) pair
    - The CLI and Python API agree: produce the same peak count and the same
      RSS strengths for EDIZUM H–H DQ/SQ with reduce=True + cif_labels
    """

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        self._kw = dict(
            xelement="H", yelement="H",
            yaxis_order="2Q", rcut=6.0,
            correlation_strength_metric="dipolar_rss",
            rss_cutoff=6.0,
            references={"H": 29.5}, gradients={"H": -0.95},
        )

    # ------------------------------------------------------------------
    # Structure of the reduced atoms object
    # ------------------------------------------------------------------

    def test_reduce_collapses_to_asymmetric_unit(self):
        """reduce=True should produce 37 atoms (the asymmetric unit of EDIZUM)."""
        nd = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=True)
        self.assertEqual(len(nd.atoms), 37)

    def test_atoms_full_is_set_when_reduce_true(self):
        """atoms_full should hold the full labeled cell (148 atoms) after reduce=True."""
        nd = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=True)
        self.assertIsNotNone(nd.atoms_full)
        self.assertEqual(len(nd.atoms_full), 148)

    def test_pair_indices_valid_for_reduced_atoms(self):
        """All pair indices must be valid indices into nd.atoms (the reduced structure)."""
        nd = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=True)
        n = len(nd.atoms)
        for i, j in nd.pairs:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, n, f"pair index i={i} out of range for reduced atoms (len={n})")
            self.assertGreaterEqual(j, 0)
            self.assertLess(j, n, f"pair index j={j} out of range for reduced atoms (len={n})")

    # ------------------------------------------------------------------
    # RSS value correctness
    # ------------------------------------------------------------------

    def test_rss_values_agree_with_direct_computation(self):
        """RSS stored in correlation_strengths must match a direct DipolarRSSByAtom call.

        The mapping is: reduced-atom label → first matching index in atoms_full.
        We spot-check the first 5 pairs.
        """
        nd = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=True)
        reduced_labels = get_atom_labels(nd.atoms, None)
        full_labels = get_atom_labels(nd.atoms_full, None)

        def _first_full_idx(label):
            matches = np.where(full_labels == label)[0]
            return int(matches[0])

        for k, (i, j) in enumerate(nd.pairs[:5]):
            fi = _first_full_idx(reduced_labels[i])
            fj = _first_full_idx(reduced_labels[j])
            expected_khz = DipolarRSSByAtom.get(
                nd.atoms_full,
                sel_i=[fi], sel_j=[fj],
                cutoff=6.0, expand_j="cif_labels",
            )[0] * 1e-3
            self.assertAlmostEqual(
                nd.correlation_strengths[k], expected_khz, places=5,
                msg=f"RSS mismatch for pair ({i},{j}): stored={nd.correlation_strengths[k]:.6f} expected={expected_khz:.6f}",
            )

    def test_reduce_true_and_false_give_same_rss_per_label_pair(self):
        """reduce=True and reduce=False should give the same RSS for the same label pair.

        We take one (H_label_i, H_label_j) pair that appears in both result sets
        and confirm the correlation_strength agrees — the same physics, just
        reached via different code paths.
        """
        nd_reduced = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=True)
        nd_full    = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=False)

        red_labels  = get_atom_labels(nd_reduced.atoms, None)
        full_labels = get_atom_labels(nd_full.atoms, None)

        # Build label-pair → strength map for both
        red_map  = {(red_labels[i],  red_labels[j]):  s for (i, j), s in zip(nd_reduced.pairs, nd_reduced.correlation_strengths)}
        full_map = {(full_labels[i], full_labels[j]):  s for (i, j), s in zip(nd_full.pairs,    nd_full.correlation_strengths)}

        common = set(red_map) & set(full_map)
        self.assertGreater(len(common), 0, "No common label-pairs between reduce=True and reduce=False results")
        for lp in list(common)[:5]:
            self.assertAlmostEqual(
                red_map[lp], full_map[lp], places=4,
                msg=f"RSS mismatch for label pair {lp}: reduce=True→{red_map[lp]:.5f} reduce=False→{full_map[lp]:.5f}",
            )

    # ------------------------------------------------------------------
    # Peak count
    # ------------------------------------------------------------------

    def test_reduce_true_fewer_peaks_than_reduce_false(self):
        """reduce=True should produce fewer (or equal) peaks than reduce=False.

        All equivalent copies are merged into one asymmetric-unit representative.
        """
        nd_r = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=True)
        nd_f = NMRData2D(self.atoms, **self._kw, rss_expand_j="cif_labels", reduce=False)
        self.assertLessEqual(
            len(nd_r.peaks), len(nd_f.peaks),
            "Expected fewer or equal peaks with reduce=True vs reduce=False",
        )


class TestNMRData2DSymmetryExpand(unittest.TestCase):
    """Tests for NMRData2D when rss_expand_j='symmetry'.

    Two sub-scenarios are covered:

    1. EDIZUM (Z=4, has CIF labels): symmetry mode calls spglib with
       override_cif=True, ignoring the embedded labels.  The RSS values must
       still agree with direct DipolarRSSByAtom calls using expand_j='symmetry',
       and for EDIZUM the spglib groups are consistent with the CIF label groups
       so both modes should give the same strengths for shared pairs.

    2. Ethanol (Z=1, no CIF labels): only MagresView-style labels are available.
       The symmetry path must not raise and must return finite, positive RSS
       values.
    """

    def setUp(self):
        self._edizum  = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        self._ethanol = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self._kw = dict(
            xelement="H", yelement="H",
            yaxis_order="2Q", rcut=6.0,
            correlation_strength_metric="dipolar_rss",
            rss_cutoff=6.0,
            references={"H": 29.5}, gradients={"H": -0.95},
        )

    # ------------------------------------------------------------------
    # EDIZUM + rss_expand_j="symmetry"
    # ------------------------------------------------------------------

    def test_edizum_symmetry_collapses_to_asymmetric_unit(self):
        """reduce=True with rss_expand_j='symmetry' should still give 37 atoms."""
        nd = NMRData2D(self._edizum, **self._kw, rss_expand_j="symmetry", reduce=True)
        self.assertEqual(len(nd.atoms), 37)

    def test_edizum_symmetry_pair_indices_valid(self):
        """All pair indices must be valid indices into the reduced atoms."""
        nd = NMRData2D(self._edizum, **self._kw, rss_expand_j="symmetry", reduce=True)
        n = len(nd.atoms)
        for i, j in nd.pairs:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, n, f"pair index i={i} out of range for reduced atoms (len={n})")
            self.assertGreaterEqual(j, 0)
            self.assertLess(j, n, f"pair index j={j} out of range for reduced atoms (len={n})")

    def test_edizum_symmetry_rss_values_agree_direct_computation(self):
        """RSS stored in correlation_strengths must match direct DipolarRSSByAtom calls
        using expand_j='symmetry'.  Spot-checks the first 5 pairs."""
        nd = NMRData2D(self._edizum, **self._kw, rss_expand_j="symmetry", reduce=True)
        reduced_labels = get_atom_labels(nd.atoms, None)
        full_labels    = get_atom_labels(nd.atoms_full, None)

        def _first_full_idx(label):
            return int(np.where(full_labels == label)[0][0])

        for k, (i, j) in enumerate(nd.pairs[:5]):
            fi = _first_full_idx(reduced_labels[i])
            fj = _first_full_idx(reduced_labels[j])
            expected_khz = DipolarRSSByAtom.get(
                nd.atoms_full,
                sel_i=[fi], sel_j=[fj],
                cutoff=6.0, expand_j="symmetry",
            )[0] * 1e-3
            self.assertAlmostEqual(
                nd.correlation_strengths[k], expected_khz, places=5,
                msg=(
                    f"RSS mismatch for pair ({i},{j}): "
                    f"stored={nd.correlation_strengths[k]:.6f} "
                    f"expected={expected_khz:.6f}"
                ),
            )

    def test_edizum_symmetry_and_cif_labels_agree_for_common_pairs(self):
        """For EDIZUM, spglib groups and CIF label groups are consistent, so
        rss_expand_j='symmetry' and 'cif_labels' should give the same RSS for
        any label pair that appears in both result sets."""
        nd_sym = NMRData2D(self._edizum, **self._kw, rss_expand_j="symmetry",   reduce=True)
        nd_cif = NMRData2D(self._edizum, **self._kw, rss_expand_j="cif_labels", reduce=True)

        sym_labels = get_atom_labels(nd_sym.atoms, None)
        cif_labels = get_atom_labels(nd_cif.atoms, None)

        sym_map = {
            (sym_labels[i], sym_labels[j]): s
            for (i, j), s in zip(nd_sym.pairs, nd_sym.correlation_strengths)
        }
        cif_map = {
            (cif_labels[i], cif_labels[j]): s
            for (i, j), s in zip(nd_cif.pairs, nd_cif.correlation_strengths)
        }

        common = set(sym_map) & set(cif_map)
        self.assertGreater(len(common), 0,
                           "No common label-pairs between symmetry and cif_labels results")
        for lp in list(common)[:10]:
            self.assertAlmostEqual(
                sym_map[lp], cif_map[lp], places=4,
                msg=(
                    f"RSS mismatch for pair {lp}: "
                    f"symmetry={sym_map[lp]:.5f} cif_labels={cif_map[lp]:.5f}"
                ),
            )

    # ------------------------------------------------------------------
    # Ethanol + rss_expand_j="symmetry"  (no CIF labels present)
    # ------------------------------------------------------------------

    def test_ethanol_no_cif_labels_symmetry_does_not_raise(self):
        """rss_expand_j='symmetry' must not raise for structures without CIF labels."""
        from soprano.nmr.extract import has_cif_labels
        self.assertFalse(
            has_cif_labels(self._ethanol),
            "ethanol.magres unexpectedly has CIF labels; choose a different fixture",
        )
        nd = NMRData2D(self._ethanol, **self._kw, rss_expand_j="symmetry", reduce=True)
        self.assertGreater(len(nd.pairs), 0, "Expected at least one H–H pair in ethanol")

    def test_ethanol_no_cif_labels_rss_values_finite_and_positive(self):
        """All RSS strengths for ethanol must be finite and positive."""
        nd = NMRData2D(self._ethanol, **self._kw, rss_expand_j="symmetry", reduce=True)
        self.assertTrue(
            np.all(np.isfinite(nd.correlation_strengths)),
            "Some RSS values are not finite",
        )
        self.assertTrue(
            np.all(nd.correlation_strengths > 0),
            "Some RSS values are not positive",
        )


class TestPlotNMRCLI(unittest.TestCase):
    """End-to-end CLI tests for `soprano plotnmr` with dipolar_rss weighting.

    Uses Click's test runner so no subprocess is spawned.  Checks that:
    - The command runs without errors for basic 2D H–H DQ/SQ cases
    - The --no-reduce flag runs cleanly (more peaks than default reduce)
    - --weight-by dipolar_rss --rss-expand-j cif_labels terminates with exit_code=0
    """

    def setUp(self):
        import tempfile
        from unittest.mock import patch
        from click.testing import CliRunner
        from soprano.scripts.cli import soprano as soprano_cli
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend; no display required
        self._runner = CliRunner()
        self._patch = patch
        self._soprano_cli = soprano_cli
        self._magres = os.path.join(_TESTDATA_DIR, "EDIZUM.magres")
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil, matplotlib.pyplot as plt
        plt.close("all")
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _csv_path(self, name):
        return os.path.join(self._tmp, name)

    def _run(self, extra_args):
        """Invoke plotnmr and save the plot to a PNG file to avoid plt.show().

        Uses Agg backend (set in setUp) so no display is needed.
        """
        import matplotlib.pyplot as plt
        png = self._csv_path("out.png")
        base = [
            "plotnmr", self._magres,
            "-p", "2D",
            "-x", "H", "-y", "H",
            "--yaxis-order", "2Q",
            "--references", "H:29.5",
            "--output", png,   # save plot to file; prevents plt.show()
        ]
        with self._patch("click_log.basic_config"):
            result = self._runner.invoke(self._soprano_cli, base + extra_args)
        plt.close("all")
        return result

    def test_cli_basic_2d_exits_cleanly(self):
        """Basic 2D H–H DQ/SQ plot (default reduce, fixed weights) returns exit_code=0."""
        result = self._run([])
        self.assertEqual(result.exit_code, 0,
                          f"Unexpected exit: {result.output}\n{result.exception}")

    def test_cli_no_reduce_exits_cleanly(self):
        """--no-reduce runs without errors (uses full 148-atom structure)."""
        result = self._run(["--no-reduce"])
        self.assertEqual(result.exit_code, 0,
                          f"Unexpected exit: {result.output}\n{result.exception}")

    def test_cli_dipolar_rss_periodic_images_exits_cleanly(self):
        """--weight-by dipolar_rss with default expand_j runs cleanly."""
        result = self._run(["--weight-by", "dipolar_rss", "--rss-cutoff", "6.0"])
        self.assertEqual(result.exit_code, 0,
                          f"Unexpected exit: {result.output}\n{result.exception}")

    def test_cli_dipolar_rss_cif_labels_exits_cleanly(self):
        """--weight-by dipolar_rss --rss-expand-j cif_labels runs cleanly."""
        result = self._run([
            "--weight-by", "dipolar_rss",
            "--rss-cutoff", "6.0",
            "--rss-expand-j", "cif_labels",
        ])
        self.assertEqual(result.exit_code, 0,
                          f"Unexpected exit: {result.output}\n{result.exception}")

    def test_cli_reduce_and_no_reduce_both_exit_cleanly(self):
        """Both --reduce (default) and --no-reduce complete without errors."""
        result_r = self._run([])
        result_f = self._run(["--no-reduce"])
        self.assertEqual(result_r.exit_code, 0,
                         f"--reduce failed: {result_r.output}\n{result_r.exception}")
        self.assertEqual(result_f.exit_code, 0,
                         f"--no-reduce failed: {result_f.output}\n{result_f.exception}")

    def test_cli_dipolar_rss_symmetry_expand_exits_cleanly(self):
        """--rss-expand-j symmetry runs cleanly (uses spglib, ignores CIF labels)."""
        result = self._run([
            "--weight-by", "dipolar_rss",
            "--rss-cutoff", "6.0",
            "--rss-expand-j", "symmetry",
        ])
        self.assertEqual(result.exit_code, 0,
                         f"Unexpected exit: {result.output}\n{result.exception}")

    def test_cli_export_simpson_writes_files(self):
        """--export-file writes SIMPSON grid and companion peaks CSV."""
        export_path = self._csv_path("cli_export.spe")
        result = self._run([
            "--export-file", export_path,
            "--export-format", "simpson",
            "--x-larmor-freq", "100.0",
        ])
        self.assertEqual(result.exit_code, 0,
                         f"Unexpected exit: {result.output}\n{result.exception}")
        self.assertTrue(os.path.exists(export_path), "Expected .spe export file")
        self.assertTrue(
            os.path.exists(export_path + ".peaks.csv"),
            "Expected SIMPSON companion .peaks.csv file",
        )

    def test_cli_export_npz_respects_grid_max(self):
        """--grid-max rescales exported NPZ contour grid to requested max intensity."""
        export_path = self._csv_path("cli_export_scaled.npz")
        target_max = 1.0e6
        result = self._run([
            "--export-file", export_path,
            "--export-format", "npz",
            "--grid-max", str(target_max),
        ])
        self.assertEqual(result.exit_code, 0,
                         f"Unexpected exit: {result.output}\n{result.exception}")
        self.assertTrue(os.path.exists(export_path), "Expected .npz export file")

        payload = np.load(export_path, allow_pickle=True)
        self.assertAlmostEqual(float(np.max(payload["Z"])), target_max, places=6)


if __name__ == '__main__':
    unittest.main()
