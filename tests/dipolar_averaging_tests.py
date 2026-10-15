#!/usr/bin/env python
"""Tests for averaged_dipolar_coupling, the fast-exchange motional average.

The static reference is the closed form d = -mu0 hbar g_i g_j / (8 pi^2 r^3),
evaluated independently of the code under test.
"""

import os
import sys
import unittest
import warnings

import numpy as np
from ase import Atoms, io
from scipy import constants as cnst

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from soprano.data.nmr import _get_isotope_data  # noqa: E402
from soprano.properties.nmr.dipolar import (  # noqa: E402
    DipolarCouplingList,
    averaged_dipolar_coupling,
    averaged_dipolar_coupling_list,
)
from soprano.properties.nmr.isc import JCouplingList, averaged_j_coupling_list  # noqa: E402

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
_GAMMA_H = _get_isotope_data(["H"], "gamma", {}, None)[0]


def _static_hh(r_ang):
    """Static H-H dipolar coupling in Hz at distance r (Angstrom)."""
    r = r_ang * 1e-10
    return -(cnst.mu_0 * cnst.hbar * _GAMMA_H**2) / (8 * np.pi**2 * r**3)


class TestSinglePair(unittest.TestCase):
    def test_single_pair_equals_static_coupling(self):
        r = 1.5
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, r]])
        d_eff, D_avg = averaged_dipolar_coupling(atoms, [0], [1])
        self.assertAlmostEqual(d_eff / _static_hh(r), 1.0, places=10)
        # One pair is not averaged, so the tensor is the static one: (-d, -d, 2d)
        evals = np.sort(np.linalg.eigvalsh(D_avg))
        d = _static_hh(r)
        np.testing.assert_allclose(evals, sorted([-d, -d, 2 * d]), rtol=1e-10)

    def test_single_pair_is_symmetric_in_groups(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.8]])
        self.assertAlmostEqual(
            averaged_dipolar_coupling(atoms, [0], [1])[0],
            averaged_dipolar_coupling(atoms, [1], [0])[0],
        )

    def test_coupling_is_negative_for_protons(self):
        """gamma_i gamma_j > 0 for two protons, so d is negative."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.0]])
        self.assertLess(averaged_dipolar_coupling(atoms, [0], [1])[0], 0.0)


class TestMethylGroup(unittest.TestCase):
    """Three-site methyl hop on the ethanol test structure."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.methyl = [0, 1, 2]  # the three H on C6
        self.ch2 = [3, 4, 5]     # the non-methyl H on C7

    def test_methyl_hydrogens_are_bonded_to_c6(self):
        """Sanity check on the index choice used below."""
        symbols = self.atoms.get_chemical_symbols()
        pos = self.atoms.get_positions()
        for i in self.methyl:
            self.assertEqual(symbols[i], "H")
            self.assertLess(np.linalg.norm(pos[i] - pos[6]), 1.2)

    def test_intra_methyl_is_minus_half_static(self):
        pos = self.atoms.get_positions()
        static = [
            _static_hh(np.linalg.norm(pos[i] - pos[j]))
            for i in self.methyl for j in self.methyl if i < j
        ]
        d_eff, _ = averaged_dipolar_coupling(self.atoms, self.methyl, self.methyl)
        self.assertAlmostEqual(d_eff / (-0.5 * np.mean(static)), 1.0, delta=0.02)

    def test_intra_methyl_tensor_is_axial(self):
        """The three edges lie in a plane, so the averaged tensor is axial:
        eigenvalues (d/2, d/2, -d) with d the mean static constant."""
        pos = self.atoms.get_positions()
        static = np.mean([
            _static_hh(np.linalg.norm(pos[i] - pos[j]))
            for i in self.methyl for j in self.methyl if i < j
        ])
        _, D_avg = averaged_dipolar_coupling(self.atoms, self.methyl, self.methyl)
        evals = np.sort(np.linalg.eigvalsh(D_avg))
        np.testing.assert_allclose(
            evals, np.sort([static / 2, static / 2, -static]), rtol=0.02
        )

    def test_external_proton_bounded_by_mean_static_magnitude(self):
        """|d_eff| never exceeds the mean of the static magnitudes."""
        pos = self.atoms.get_positions()
        for k in self.ch2:
            static = [
                _static_hh(np.linalg.norm(pos[k] - pos[i])) for i in self.methyl
            ]
            d_eff, _ = averaged_dipolar_coupling(self.atoms, [k], self.methyl)
            self.assertLessEqual(abs(d_eff), np.mean(np.abs(static)) * (1 + 1e-10))
            self.assertLess(d_eff, 0.0)

    def test_doc_example_values(self):
        """Numbers quoted in docs/dipolar-averaging.md."""
        d_ext, _ = averaged_dipolar_coupling(self.atoms, [3], self.methyl)
        d_intra, _ = averaged_dipolar_coupling(self.atoms, self.methyl, self.methyl)
        self.assertAlmostEqual(d_ext, -5091, delta=1.0)
        self.assertAlmostEqual(d_intra, 10766, delta=1.0)

    def test_doc_table_static_couplings(self):
        """Static couplings (kHz) in the docs table, to the printed precision.

        Distances are minimum-image, as in DipolarCoupling. H4-H1 crosses the
        cell boundary, so its raw distance would give a different value.
        """
        table = {
            3: [-7.60, -7.52, -4.16],
            4: [-7.57, -4.17, -7.69],
            5: [-2.52, -7.50, -4.57],
        }
        for k, row in table.items():
            for i, expected in zip(self.methyl, row):
                r = self.atoms.get_distance(k, i, mic=True)
                self.assertAlmostEqual(_static_hh(r) / 1e3, expected, delta=0.005)

    def test_doc_table_tensor_averages(self):
        """Tensor-averaged magnitudes (kHz) in the docs table."""
        expected = {3: 5.09, 4: 5.06, 5: 4.17}
        for k, value in expected.items():
            d_eff, _ = averaged_dipolar_coupling(self.atoms, [k], self.methyl)
            self.assertAlmostEqual(abs(d_eff) / 1e3, value, delta=0.005)


class TestEmptySelections(unittest.TestCase):
    def test_empty_sel_i(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        d_eff, D_avg = averaged_dipolar_coupling(atoms, [], [1])
        self.assertEqual(d_eff, 0.0)
        np.testing.assert_array_equal(D_avg, np.zeros((3, 3)))

    def test_empty_sel_j(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        d_eff, D_avg = averaged_dipolar_coupling(atoms, [0], [])
        self.assertEqual(d_eff, 0.0)
        np.testing.assert_array_equal(D_avg, np.zeros((3, 3)))

    def test_only_self_pairs(self):
        """i == j pairs are skipped, leaving no valid pairs."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        d_eff, D_avg = averaged_dipolar_coupling(atoms, [0], [0])
        self.assertEqual(d_eff, 0.0)
        np.testing.assert_array_equal(D_avg, np.zeros((3, 3)))


class TestPeriodicImages(unittest.TestCase):
    def test_minimum_image_across_boundary(self):
        """Atoms 9 Å apart in a 10 Å box are 1 Å apart through the boundary."""
        atoms = Atoms(
            "H2",
            positions=[[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        d_eff, _ = averaged_dipolar_coupling(atoms, [0], [1])
        self.assertAlmostEqual(d_eff / _static_hh(1.0), 1.0, places=10)

    def test_zero_cell_does_not_warn(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            d_eff, _ = averaged_dipolar_coupling(atoms, [0], [1])
        self.assertAlmostEqual(d_eff / _static_hh(1.5), 1.0, places=10)


class TestOverlappingAtoms(unittest.TestCase):
    def test_coincident_atoms_raise(self):
        """Zero separation would give NaN; it must be an explicit error."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            averaged_dipolar_coupling(atoms, [0], [1])


class TestRSSByAtomHelpers(unittest.TestCase):
    """Edge cases of DipolarRSSByAtom and its symmetry-tag cache in dipolar.py."""

    def test_empty_sel_j_warns_like_no_neighbour(self):
        from soprano.properties.nmr.dipolar import DipolarRSSByAtom

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        with self.assertWarnsRegex(UserWarning, "sel_j is empty"):
            rss = DipolarRSSByAtom.get(atoms, sel_i=[0], sel_j=[], cutoff=5.0)
        np.testing.assert_array_equal(rss, [0.0])

    def test_zero_cell_rss_matches_static_and_does_not_warn(self):
        from soprano.properties.nmr.dipolar import DipolarRSSByAtom

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            rss = DipolarRSSByAtom.get(atoms, sel_i=[0], sel_j=[1], cutoff=5.0)
        np.testing.assert_allclose(rss, [abs(_static_hh(1.5))], rtol=1e-10)

    def test_expand_j_none_raises(self):
        from soprano.properties.nmr.dipolar import DipolarRSSByAtom

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])
        with self.assertRaises(ValueError):
            DipolarRSSByAtom.get(atoms, sel_i=[0], sel_j=[1], expand_j=None)

    def test_symmetry_tags_follow_structure_changes(self):
        """Cached tags must not survive a change to the structure."""
        from soprano.properties.nmr.dipolar import _symmetry_tags

        atoms = Atoms("H3", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.75]],
                      cell=[5, 5, 5], pbc=True)
        first = _symmetry_tags(atoms, 1e-4, override_cif=False)
        np.testing.assert_array_equal(first, [0, 0, 1])  # first two are equivalent
        np.testing.assert_array_equal(
            _symmetry_tags(atoms, 1e-4, override_cif=False), first
        )

        atoms.positions[2] += [0.7, 0.0, 0.0]  # break the equivalence
        second = _symmetry_tags(atoms, 1e-4, override_cif=False)
        np.testing.assert_array_equal(second, [0, 1, 2])


class TestAveragedCouplingList(unittest.TestCase):
    """Spin-system couplings averaged over collapsed groups."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.methyl = [0, 1, 2]
        self.c6 = 6  # the methyl carbon

    def test_single_atom_sites_match_static_list(self):
        sets = [[i] for i in range(len(self.atoms))]
        new = {(c.site_i, c.site_j): c for c in averaged_dipolar_coupling_list(self.atoms, sets)}
        old = {(c.site_i, c.site_j): c for c in DipolarCouplingList.get(self.atoms)}
        self.assertEqual(new.keys(), old.keys())
        for ij, c in old.items():
            self.assertAlmostEqual(new[ij].coupling_constant, c.coupling_constant, places=6)

    def test_collapsed_methyl_to_its_carbon_is_minus_a_third(self):
        """The C-H vectors sit at ~109.5 deg to C3, so P2 = -1/3."""
        sets = [self.methyl, [self.c6]]
        (coupling,) = averaged_dipolar_coupling_list(self.atoms, sets)
        pos = self.atoms.get_positions()
        static = np.mean([
            DipolarCouplingList.get(self.atoms, sel_i=[i], sel_j=[self.c6])[0].coupling_constant
            for i in self.methyl
        ])
        self.assertGreater(coupling.coupling_constant, 0)
        self.assertLess(static, 0)
        self.assertAlmostEqual(coupling.coupling_constant / static, -1 / 3, delta=0.03)
        # and not what the first hydrogen alone would give
        first_h = DipolarCouplingList.get(self.atoms, sel_i=[0], sel_j=[self.c6])[0]
        self.assertLess(first_h.coupling_constant, 0)

    def test_sites_sharing_a_group_get_the_intra_group_average(self):
        """Expanded group members couple through the -1/2 residual."""
        (coupling,) = averaged_dipolar_coupling_list(self.atoms, [self.methyl, self.methyl])
        d_eff, _ = averaged_dipolar_coupling(self.atoms, self.methyl, self.methyl)
        self.assertAlmostEqual(coupling.coupling_constant, d_eff, places=6)
        self.assertGreater(coupling.coupling_constant, 0)

    def test_selection_and_isonuclear(self):
        sets = [self.methyl, [3], [self.c6], [7]]
        all_pairs = averaged_dipolar_coupling_list(self.atoms, sets)
        self.assertEqual(len(all_pairs), 6)
        c_h = averaged_dipolar_coupling_list(self.atoms, sets, sel_i=[2, 3], sel_j=[0, 1])
        self.assertEqual({(c.site_i, c.site_j) for c in c_h}, {(0, 2), (0, 3), (1, 2), (1, 3)})
        hh = averaged_dipolar_coupling_list(self.atoms, sets, isonuclear=True)
        self.assertEqual({(c.site_i, c.site_j) for c in hh}, {(0, 1), (2, 3)})

    def test_mixed_species_in_one_site_rejected(self):
        with self.assertRaises(ValueError):
            averaged_dipolar_coupling_list(self.atoms, [[0, self.c6], [3]])


class TestAveragedJCouplingList(unittest.TestCase):
    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

    def test_single_atom_sites_match_static_list(self):
        sets = [[i] for i in range(len(self.atoms))]
        new = {(c.site_i, c.site_j): c for c in averaged_j_coupling_list(self.atoms, sets)}
        old = {(c.site_i, c.site_j): c for c in JCouplingList.get(self.atoms)}
        self.assertEqual(new.keys(), old.keys())
        for ij, c in old.items():
            np.testing.assert_allclose(new[ij].tensor.data, c.tensor.data)

    def test_group_tensor_is_the_mean_over_members(self):
        (coupling,) = averaged_j_coupling_list(self.atoms, [[0, 1, 2], [6]])
        static = [c for c in JCouplingList.get(self.atoms) if 6 in (c.site_i, c.site_j)
                  and min(c.site_i, c.site_j) in (0, 1, 2)]
        self.assertEqual(len(static), 3)
        np.testing.assert_allclose(
            coupling.tensor.data, np.mean([c.tensor.data for c in static], axis=0)
        )


if __name__ == "__main__":
    unittest.main()
