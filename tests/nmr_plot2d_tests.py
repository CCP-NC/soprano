#!/usr/bin/env python
"""Tests for the 2D NMR plotting / contour-grid pipeline.

These target behaviours that were previously untested:
- signed contour level generation (``use_signed``)
- degenerate / zero-intensity grids (``_resolve_levels`` guard)
- Gaussian vs Lorentzian broadening line widths (FWHM correctness)
- signed vs unsigned contour grids (negative correlation strengths)
- heteronuclear 1Q spectra (default ``yaxis_order``) and the heteronuclear
  2Q (DQ) warning
- empty-peaks edge cases (raise on contour, no crash on markers)
- matplotlib rendering smoke tests (figure/axes, labels, diagonal)
"""

import logging
import os
import unittest

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import numpy as np
from ase import io

from soprano.calculate.nmr.backends import (
    PLOTLY_AVAILABLE,
    MatplotlibBackend,
    PlotlyBackend,
    _resolve_levels,
)
from soprano.calculate.nmr.config import PlotSettings
from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.nmr import Peak2D
from soprano.calculate.nmr.plot2d import NMRPlot2D
from soprano.calculate.nmr.utils import generate_contour_map, generate_peaks, merge_peaks

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def _fwhm_of_profile(x: np.ndarray, z: np.ndarray) -> float:
    """Measure the full width at half maximum of a 1D peak profile."""
    zmax = z.max()
    half = zmax / 2.0
    above = np.where(z >= half)[0]
    # Linear-interpolate the crossing points for sub-sample accuracy.
    li, ri = above[0], above[-1]
    # left crossing between li-1 and li
    if li > 0:
        x_left = np.interp(half, [z[li - 1], z[li]], [x[li - 1], x[li]])
    else:
        x_left = x[li]
    if ri < len(x) - 1:
        x_right = np.interp(half, [z[ri + 1], z[ri]], [x[ri + 1], x[ri]])
    else:
        x_right = x[ri]
    return abs(x_right - x_left)


class TestResolveLevels(unittest.TestCase):
    """Contour level generation, including signed and degenerate grids."""

    def test_positive_levels_span_contour_range(self):
        Z = np.array([[0.0, 0.5], [1.0, 2.0]])
        levels = _resolve_levels(Z, 5, (10.0, 100.0))
        self.assertEqual(len(levels), 5)
        # Peak magnitude is 2.0; 10%..100% -> 0.2 .. 2.0
        self.assertAlmostEqual(levels[0], 0.2, places=6)
        self.assertAlmostEqual(levels[-1], 2.0, places=6)
        self.assertTrue(np.all(levels >= 0))

    def test_signed_levels_include_negative(self):
        """use_signed must produce mirrored negative levels for signed grids."""
        Z = np.array([[-2.0, -0.5], [0.5, 2.0]])
        levels = _resolve_levels(Z, 4, (10.0, 100.0), use_signed=True)
        self.assertTrue(np.any(levels < 0), "Expected negative contour levels")
        self.assertTrue(np.any(levels > 0), "Expected positive contour levels")
        # Symmetric about zero.
        self.assertAlmostEqual(levels.min(), -levels.max(), places=6)

    def test_signed_flag_but_all_positive_stays_positive(self):
        """use_signed on a purely positive grid yields only positive levels."""
        Z = np.array([[0.0, 1.0], [1.5, 2.0]])
        levels = _resolve_levels(Z, 4, (10.0, 100.0), use_signed=True)
        self.assertTrue(np.all(levels >= 0))

    def test_zero_grid_raises(self):
        """An all-zero grid has no contour structure: fail with a clear error."""
        Z = np.zeros((4, 4))
        with self.assertRaisesRegex(ValueError, "all zero"):
            _resolve_levels(Z, 10, (10.0, 100.0))

    def test_all_nan_grid_raises(self):
        Z = np.full((4, 4), np.nan)
        with self.assertRaisesRegex(ValueError, "no finite values"):
            _resolve_levels(Z, 10, (10.0, 100.0))

    def test_numpy_integer_levels_are_counts(self):
        Z = np.array([[0.0, 0.5], [1.0, 2.0]])
        levels = _resolve_levels(Z, np.int64(5), (10.0, 100.0))
        self.assertEqual(len(levels), 5)


class TestBroadening(unittest.TestCase):
    """Gaussian vs Lorentzian broadening produce the requested FWHM."""

    def _single_peak(self):
        return [Peak2D(x=50.0, y=50.0, correlation_strength=1.0,
                       xlabel="A", ylabel="B", idx_x=0, idx_y=1)]

    def test_gaussian_fwhm_matches_broadening(self):
        fwhm = 2.0
        X, Y, Z = generate_contour_map(
            self._single_peak(), grid_size=801,
            broadening="gaussian", x_broadening=fwhm, y_broadening=fwhm,
        )
        # Central horizontal cross-section through the peak row.
        row = np.argmin(np.abs(Y[:, 0] - 50.0))
        measured = _fwhm_of_profile(X[row, :], Z[row, :])
        self.assertAlmostEqual(measured, fwhm, delta=0.1 * fwhm)

    def test_lorentzian_fwhm_matches_broadening(self):
        fwhm = 2.0
        X, Y, Z = generate_contour_map(
            self._single_peak(), grid_size=2001,
            broadening="lorentzian", x_broadening=fwhm, y_broadening=fwhm,
        )
        row = np.argmin(np.abs(Y[:, 0] - 50.0))
        measured = _fwhm_of_profile(X[row, :], Z[row, :])
        self.assertAlmostEqual(measured, fwhm, delta=0.1 * fwhm)

    def test_lorentzian_has_longer_tails_than_gaussian(self):
        """At several FWHM out, the Lorentzian retains far more intensity."""
        peak = self._single_peak()
        fwhm = 2.0
        # Evaluate both profiles on the same explicit grid via the raw functions.
        from soprano.calculate.nmr.utils import gaussian, lorentzian
        x = np.array([50.0 + 5 * fwhm])
        y = np.array([50.0])
        X, Yg = np.meshgrid(x, y)
        g = gaussian(X, 50.0, Yg, 50.0, fwhm, fwhm, normalise=False)
        lo = lorentzian(X, 50.0, Yg, 50.0, fwhm, fwhm, normalise=False)
        self.assertGreater(float(lo.ravel()[0]), float(g.ravel()[0]))


class TestSeparableLorentzian(unittest.TestCase):
    """The 2D Lorentzian is the product of two 1D Lorentzians."""

    def test_fwhm_along_x_is_the_same_at_any_y(self):
        from soprano.calculate.nmr.utils import lorentzian

        fwhm = 2.0
        x = np.linspace(40.0, 60.0, 4001)
        y = np.array([50.0, 50.0 + 3.0 * fwhm])
        X, Y = np.meshgrid(x, y)
        Z = lorentzian(X, 50.0, Y, 50.0, fwhm, fwhm, normalise=False)
        widths = [_fwhm_of_profile(x, Z[row]) for row in range(len(y))]
        self.assertAlmostEqual(widths[0], fwhm, places=3)
        self.assertAlmostEqual(widths[1], widths[0], places=6)

    def test_normalised_integral_is_one(self):
        from soprano.calculate.nmr.utils import lorentzian

        fwhm = 1.0
        # ±200 FWHM window; the Lorentzian tail outside it is ~0.3 % per axis.
        x = np.linspace(-200.0, 200.0, 1601)
        X, Y = np.meshgrid(x, x)
        Z = lorentzian(X, 0.0, Y, 0.0, fwhm, fwhm, normalise=True)
        # Uniform grid, so the Riemann sum is accurate to well under 1 %.
        area = float(Z.sum()) * (x[1] - x[0]) ** 2
        self.assertAlmostEqual(area, 1.0, delta=0.02)

    def test_unnormalised_peak_maximum_is_one(self):
        from soprano.calculate.nmr.utils import lorentzian

        X, Y = np.meshgrid(np.linspace(-5.0, 5.0, 101), np.linspace(-5.0, 5.0, 101))
        Z = lorentzian(X, 0.0, Y, 0.0, 1.0, 2.0, normalise=False)
        self.assertAlmostEqual(float(Z.max()), 1.0, places=6)


class TestContourGridCoordinates(unittest.TestCase):
    """The coordinate grid must survive the broadening loop untouched.

    NumPy's temporary-elision optimisation can rewrite ``X - x0`` into an
    in-place subtraction when it mistakes a caller-owned array for a
    temporary, which shifts the whole grid by -x0 once per peak and silently
    moves the contour/heatmap away from the peaks it is meant to describe.
    """

    def _peaks(self):
        return [
            Peak2D(x=10.0 + 3.0 * i, y=20.0 + 5.0 * i, correlation_strength=1.0 + i,
                   xlabel=f"X{i}", ylabel=f"Y{i}", idx_x=i, idx_y=i + 1)
            for i in range(8)
        ]

    def test_grid_spans_the_requested_limits(self):
        X, Y, _ = generate_contour_map(
            self._peaks(), grid_size=64, broadening="lorentzian",
            x_broadening=1.0, y_broadening=1.0,
            xlims=(-5.0, 60.0), ylims=(-10.0, 90.0),
        )
        # Explicit limits are covered exactly; padding only ever widens.
        self.assertLessEqual(float(X.min()), -5.0)
        self.assertGreaterEqual(float(X.max()), 60.0)
        self.assertLessEqual(float(Y.min()), -10.0)
        self.assertGreaterEqual(float(Y.max()), 90.0)
        # ...and the peaks must lie inside the grid, not off its edge.
        for p in self._peaks():
            self.assertTrue(X.min() <= p.x <= X.max(), f"peak x={p.x} outside grid")
            self.assertTrue(Y.min() <= p.y <= Y.max(), f"peak y={p.y} outside grid")

    def test_grid_is_a_regular_ascending_mesh(self):
        X, Y, _ = generate_contour_map(
            self._peaks(), grid_size=32, broadening="gaussian",
            x_broadening=2.0, y_broadening=2.0,
        )
        np.testing.assert_allclose(
            X, np.tile(np.linspace(X[0, 0], X[0, -1], 32), (32, 1))
        )
        np.testing.assert_allclose(
            Y, np.tile(np.linspace(Y[0, 0], Y[-1, 0], 32), (32, 1)).T
        )

    def test_broadening_functions_do_not_touch_their_inputs(self):
        """lorentzian/gaussian must treat the coordinate grids as read-only."""
        from soprano.calculate.nmr.utils import gaussian, lorentzian

        X, Y = np.meshgrid(np.linspace(0.0, 10.0, 17), np.linspace(0.0, 10.0, 17))
        X.flags.writeable = False
        Y.flags.writeable = False
        for fn in (lorentzian, gaussian):
            with self.subTest(fn=fn.__name__):
                # Raises ValueError if the implementation writes into X or Y.
                out = fn(X, 4.0, Y, 6.0, 1.0, 1.0)
                self.assertEqual(out.shape, X.shape)
                self.assertTrue(np.all(np.isfinite(out)))


class TestSignedContourGrid(unittest.TestCase):
    """use_signed controls whether the grid keeps negative correlations."""

    def _signed_peaks(self):
        return [
            Peak2D(x=10.0, y=10.0, correlation_strength=1.0,
                   xlabel="P", ylabel="P", idx_x=0, idx_y=1),
            Peak2D(x=40.0, y=40.0, correlation_strength=-1.0,
                   xlabel="N", ylabel="N", idx_x=2, idx_y=3),
        ]

    def test_unsigned_grid_is_nonnegative(self):
        d = NMRData2D(peaks=self._signed_peaks(), xelement="H", is_shift=True)
        cd = d.get_contour_data(grid_size=60, use_signed=False)
        self.assertGreaterEqual(float(cd.Z.min()), 0.0)

    def test_signed_grid_retains_negative_lobe(self):
        d = NMRData2D(peaks=self._signed_peaks(), xelement="H", is_shift=True)
        cd = d.get_contour_data(grid_size=60, use_signed=True)
        self.assertLess(float(cd.Z.min()), 0.0)
        self.assertGreater(float(cd.Z.max()), 0.0)


class TestHeteronuclear1Q(unittest.TestCase):
    """Default (1Q) heteronuclear spectra keep x and y as separate shifts."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))

    def test_hc_1q_y_is_single_shielding_not_sum(self):
        d = NMRData2D(
            atoms=self.atoms, xelement="H", yelement="C",
            correlation_strength_metric="dipolar",
        )  # default yaxis_order == '1Q'
        self.assertEqual(d.yaxis_order, "1Q")
        for peak in d.peaks:
            # y is the C shielding of idx_y alone, NOT σ_x + σ_y
            self.assertAlmostEqual(peak.y, d.data[peak.idx_y], places=6)

    def test_hc_1q_axis_label_has_no_2q(self):
        d = NMRData2D(
            atoms=self.atoms, xelement="H", yelement="C",
            correlation_strength_metric="dipolar",
        )
        self.assertNotIn("2Q", d.y_axis_label)


class TestHeteronuclearDQWarning(unittest.TestCase):
    """Heteronuclear 2Q summing must warn about the ill-defined ppm axis."""

    def test_heteronuclear_2q_warns(self):
        data = [5.0, 100.0]
        pairs = [(0, 1)]
        labels = ["H1", "C1"]
        with self.assertLogs(
            "soprano.calculate.nmr.utils", level="WARNING"
        ) as cm:
            generate_peaks(data, pairs, labels, 1.0, "2Q", "H", "C")
        self.assertTrue(any("Heteronuclear 2Q" in m for m in cm.output))

    def test_homonuclear_2q_does_not_warn(self):
        data = [5.0, 7.0]
        pairs = [(0, 1)]
        labels = ["H1", "H2"]
        logger = logging.getLogger("soprano.calculate.nmr.utils")
        with self.assertNoLogs(logger, level="WARNING"):
            generate_peaks(data, pairs, labels, 1.0, "2Q", "H", "H")


class TestEmptyPeaks(unittest.TestCase):
    """Empty peak lists must fail loudly for grids and not crash markers."""

    def test_contour_raises_on_empty(self):
        d = NMRData2D(peaks=[], xelement="H", is_shift=True)
        with self.assertRaises(ValueError):
            d.get_contour_data(grid_size=20)

    def test_plot_with_empty_peaks_does_not_crash(self):
        d = NMRData2D(peaks=[], xelement="H", is_shift=True)
        settings = PlotSettings(
            show_heatmap=False, show_contour=False,
            show_markers=True, show_labels=False,
            plot_filename=None,
        )
        plot = NMRPlot2D(d, plot_settings=settings)
        fig, ax = plot.plot()
        self.assertIsNotNone(fig)
        plt.close("all")


class TestPlot2DRendering(unittest.TestCase):
    """Smoke tests for the matplotlib rendering path."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        self.d = NMRData2D(
            atoms=self.atoms, xelement="H", yelement="H",
            yaxis_order="2Q", references={"H": 29.5},
            correlation_strength_metric="fixed",
        )

    def tearDown(self):
        plt.close("all")

    def test_plot_returns_fig_and_axes(self):
        plot = NMRPlot2D(self.d, plot_settings=PlotSettings(plot_filename=None))
        fig, ax = plot.plot()
        self.assertIsInstance(ax, plt.Axes)
        self.assertGreater(len(ax.collections) + len(ax.lines), 0)

    def test_axis_labels_are_set(self):
        plot = NMRPlot2D(self.d, plot_settings=PlotSettings(plot_filename=None))
        fig, ax = plot.plot()
        self.assertTrue(ax.get_xlabel())
        self.assertTrue(ax.get_ylabel())
        self.assertIn("2Q", ax.get_ylabel())

    def test_shift_axes_are_inverted(self):
        """Both ppm axes should be inverted (descending) for shift data."""
        plot = NMRPlot2D(self.d, plot_settings=PlotSettings(plot_filename=None))
        fig, ax = plot.plot()
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        self.assertGreater(xlo, xhi, "x-axis should be inverted for shift data")
        self.assertGreater(ylo, yhi, "y-axis should be inverted for shift data")

    def test_diagonal_uses_settings_limits(self):
        """The 2Q diagonal (y=2x) must span the requested x-limits."""
        backend = MatplotlibBackend()
        backend.set_axis_properties("x", "y", (0.0, 10.0), (0.0, 20.0), False)
        settings = PlotSettings(yaxis_order="2Q")
        backend.plot_diagonal(settings)
        line = backend.ax.lines[-1]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # y = 2x across the stored xlim (0, 10) -> (0, 20)
        self.assertAlmostEqual(xdata[0], 0.0, places=6)
        self.assertAlmostEqual(xdata[1], 10.0, places=6)
        self.assertAlmostEqual(ydata[0], 0.0, places=6)
        self.assertAlmostEqual(ydata[1], 20.0, places=6)
        plt.close("all")

    def test_identity_diagonal_is_y_equals_x(self):
        backend = MatplotlibBackend()
        backend.set_axis_properties("x", "y", (0.0, 10.0), (0.0, 10.0), False)
        settings = PlotSettings(yaxis_order="1Q")
        backend.plot_diagonal(settings)
        line = backend.ax.lines[-1]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        self.assertAlmostEqual(xdata[0], ydata[0], places=6)
        self.assertAlmostEqual(xdata[1], ydata[1], places=6)
        plt.close("all")

    def test_lorentzian_heatmap_grid_size_contours(self):
        """Explicit heatmap_grid_size with Lorentzian broadening should generate valid contours."""
        settings = PlotSettings(
            show_contour=True,
            show_heatmap=True,
            broadening_type="lorentzian",
            heatmap_grid_size=600,
            xlim=(6, -2),
            ylim=(12, -4),
        )
        plot = NMRPlot2D(self.d, plot_settings=settings)
        fig, ax = plot.plot()
        num_paths = sum(len(c.get_paths()) for c in ax.collections)
        self.assertGreater(num_paths, 0, "Contours should be drawn when heatmap_grid_size is explicitly set")
        plt.close("all")


class TestAverageGroup(unittest.TestCase):
    """Functional-group averaging must see the groups' heavy atoms.

    A pattern like CH3 or NH3 is anchored on its heavy atom, and both of the
    steps that derive a structure can remove it: ``xelement``/``yelement``
    filtering strips carbon from a homonuclear H-H plot, and ``reduce`` can
    merge the group members away. Detection therefore runs against the source
    structure and is translated through the site map.
    """

    def setUp(self):
        self.alanine = io.read(os.path.join(_TESTDATA_DIR, "alanine_manual_labels.magres"))
        self.ethanol = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.edizum = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))

    _HH = dict(
        xelement="H",
        yelement="H",
        references={"H": 29.5},
        rcut=6.0,
        correlation_strength_metric="dipolar",
    )

    def test_homonuclear_hh_finds_ch3_and_nh3(self):
        """H-H plot: CH3/NH3 groups are found even though C and N are filtered out."""
        plain = NMRData2D(self.alanine, **self._HH)
        grouped = NMRData2D(self.alanine, average_group="CH3,NH3", **self._HH)

        self.assertLess(
            len(grouped.peaks),
            len(plain.peaks),
            "Averaging CH3 and NH3 should merge peaks in an H-H spectrum. "
            "An equal count means group detection silently found nothing.",
        )

    def test_homonuclear_hh_with_reduce(self):
        """Reduction merges group members away, so detection cannot use it either."""
        for atoms in (self.alanine, self.edizum):
            with self.subTest(formula=atoms.get_chemical_formula()):
                plain = NMRData2D(atoms.copy(), reduce=True, **self._HH)
                grouped = NMRData2D(
                    atoms.copy(), reduce=True, average_group="CH3", **self._HH
                )
                self.assertLess(
                    len(grouped.peaks),
                    len(plain.peaks),
                    "average_group must still merge when reduce=True",
                )

    def test_site_map_spans_source_to_final_sites(self):
        """The composed map relates the user's input to the peaks' index space."""
        d = NMRData2D(self.alanine, reduce=True, **self._HH)
        self.assertEqual(d.site_map.n_atoms, len(self.alanine))
        self.assertEqual(d.site_map.n_sites, len(d.atoms))
        d.site_map.validate(d.source)
        # Only hydrogen survives, and every surviving site has at least one atom.
        for k in range(d.site_map.n_sites):
            members = d.site_map.members(k)
            self.assertGreater(len(members), 0)
            self.assertTrue(all(d.source[i].symbol == "H" for i in members))

    def test_no_warning_when_groups_are_present(self):
        """The 'matched no groups' warning must not fire for a structure that has them."""
        with self.assertLogs("soprano.calculate.nmr.data2d", level=logging.WARNING) as ctx:
            NMRData2D(
                self.alanine,
                xelement="H",
                yelement="H",
                references={"H": 29.5},
                rcut=6.0,
                correlation_strength_metric="dipolar",
                average_group="CH3,NH3",
            )
            # assertLogs fails on an empty context, so emit a sentinel to keep it happy.
            logging.getLogger("soprano.calculate.nmr.data2d").warning("sentinel")
        self.assertFalse(
            [r for r in ctx.records if "matched no groups" in r.getMessage()],
            "CH3/NH3 groups exist in alanine; no 'matched no groups' warning expected",
        )

    def test_heteronuclear_grouping_unchanged(self):
        """C-H plot: group members survive the filter, so behaviour is as before."""
        common = dict(
            xelement="C",
            yelement="H",
            references={"C": 175, "H": 30},
            correlation_strength_metric="dipolar",
        )
        plain = NMRData2D(self.ethanol, **common)
        grouped = NMRData2D(self.ethanol, average_group="CH3", **common)
        self.assertLess(len(grouped.peaks), len(plain.peaks))
        self.assertIn(3, {p.multiplicity for p in grouped.peaks})

    def test_dipolar_strengths_are_tensor_averaged(self):
        """Merged dipolar strengths must come from the averaged tensor.

        Fast methyl rotation averages the coupling *tensors* over the three
        H sites, and orientational cancellation makes the residual coupling
        smaller than the arithmetic mean of the static constants.  The old
        multiplicity-weighted mean therefore systematically overestimated
        merged dipolar strengths.
        """
        from soprano.properties.nmr import averaged_dipolar_coupling

        grouped = NMRData2D(self.ethanol, average_group="CH3", **self._HH)

        methyl = [0, 1, 2]  # CH3 hydrogens in ethanol.magres
        merged = [p for p in grouped.peaks if p.multiplicity >= 3]
        self.assertTrue(merged, "expected at least one merged methyl peak")

        # Every merged strength must equal a tensor-averaged coupling of the
        # methyl group with one of the external protons (3, 4, 5) or with
        # itself (the intra-methyl diagonal peak).
        expected = {
            round(abs(averaged_dipolar_coupling(self.ethanol, [k], methyl)[0])
                  * 1e-3, 6)
            for k in (3, 4, 5)
        }
        expected.add(
            round(abs(averaged_dipolar_coupling(self.ethanol, methyl, methyl)[0])
                  * 1e-3, 6)
        )
        for peak in merged:
            self.assertIn(
                round(abs(peak.correlation_strength), 6),
                expected,
                f"merged peak {peak.xlabel}/{peak.ylabel} does not carry a "
                "tensor-averaged coupling",
            )

        # And the cancellation property: the external-H couplings must lie
        # below the arithmetic mean of their static member couplings.
        from soprano.properties.nmr import DipolarCoupling
        for k in (3, 4, 5):
            statics = [
                abs(list(DipolarCoupling.get(
                    self.ethanol, sel_i=[k], sel_j=[h]).values())[0][0])
                for h in methyl
            ]
            d_eff, _ = averaged_dipolar_coupling(self.ethanol, [k], methyl)
            self.assertLess(abs(d_eff), np.mean(statics))

    def test_intra_methyl_coupling_scaled_by_minus_half(self):
        """Intra-methyl H-H pairs must show the -1/2 residual scaling.

        The three H-H vectors are perpendicular to the C3 axis, so the
        rotationally averaged tensor is axial along the axis with
        d_eff = -d_static/2.  The tensor average over the three edges
        reproduces this without any special-casing.
        """
        from soprano.properties.nmr import (
            DipolarCoupling,
            averaged_dipolar_coupling,
        )

        methyl = [0, 1, 2]
        statics = [
            list(DipolarCoupling.get(self.ethanol, sel_i=[i], sel_j=[j]).values())[0][0]
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        d_eff, _ = averaged_dipolar_coupling(self.ethanol, methyl, methyl)

        # Static couplings are negative (like gammas); the residual is
        # positive and close to half the mean static magnitude.  The methyl
        # in a real crystal is not perfectly equilateral, so allow 5%.
        mean_static = np.mean(np.abs(statics))
        self.assertGreater(d_eff, 0.0)
        self.assertAlmostEqual(
            d_eff / (mean_static / 2.0), 1.0, delta=0.05,
        )

    def test_averaged_coupling_empty_and_self_pairs(self):
        """No valid pairs (same single atom on both sides) must yield zero."""
        from soprano.properties.nmr import averaged_dipolar_coupling

        d_eff, D = averaged_dipolar_coupling(self.ethanol, [0], [0])
        self.assertEqual(d_eff, 0.0)
        self.assertTrue(np.all(D == 0.0))

    def test_unmatched_pattern_still_warns(self):
        """A pattern with no match anywhere in the structure keeps warning."""
        with self.assertLogs("soprano.calculate.nmr.data2d", level=logging.WARNING) as ctx:
            d = NMRData2D(
                self.ethanol,
                xelement="C",
                yelement="H",
                references={"C": 175, "H": 30},
                correlation_strength_metric="dipolar",
                average_group="NH3",
            )
        self.assertTrue(any("matched no groups" in r.getMessage() for r in ctx.records))
        self.assertEqual(len(d.peaks), len(NMRData2D(
            self.ethanol,
            xelement="C",
            yelement="H",
            references={"C": 175, "H": 30},
            correlation_strength_metric="dipolar",
        ).peaks))


class TestMergePeaksLabels(unittest.TestCase):
    """Merged peaks pool labels only from peaks that share the merge key."""

    def test_labels_are_pooled_per_merge_key(self):
        peaks = [
            Peak2D(x=1.0, y=2.0, xlabel="C1", ylabel="C2", correlation_strength=1.0),
            Peak2D(x=3.0, y=4.0, xlabel="C1", ylabel="C2", correlation_strength=2.0),
            Peak2D(x=1.0, y=2.0, xlabel="C5", ylabel="C2", correlation_strength=1.0),
        ]
        merged = {(p.x, p.y): p for p in merge_peaks(peaks)}
        self.assertEqual(merged[(3.0, 4.0)].xlabel, "C1")
        self.assertEqual(merged[(1.0, 2.0)].xlabel, "C1/C5")
        self.assertEqual(merged[(1.0, 2.0)].multiplicity, 2)


class TestCorrelationStrengthInput(unittest.TestCase):
    """Strengths can be given as a list or an array, alone or with a metric."""

    def setUp(self):
        self.ethanol = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.pairs = [(0, 3), (1, 4)]

    def test_numpy_array_strengths_are_used(self):
        d = NMRData2D(
            atoms=self.ethanol, xelement="H", yelement="H", pairs=self.pairs,
            correlation_strengths=np.array([1.5, 2.5]),
        )
        self.assertEqual(sorted(p.correlation_strength for p in d.peaks), [1.5, 2.5])

    def test_strengths_with_metric_warn_and_use_strengths(self):
        with self.assertLogs("soprano.calculate.nmr.data2d", level="WARNING") as ctx:
            d = NMRData2D(
                atoms=self.ethanol, xelement="H", yelement="H", pairs=self.pairs,
                correlation_strengths=[1.5, 2.5],
                correlation_strength_metric="distance",
            )
        self.assertTrue(any("Using correlation_strengths" in m for m in ctx.output))
        self.assertEqual(sorted(p.correlation_strength for p in d.peaks), [1.5, 2.5])


class TestDQReducedSelfPairs(unittest.TestCase):
    """DQ self-pairs of merged sites stand for their distinct-atom pairs."""

    def setUp(self):
        self.atoms = io.read(os.path.join(_TESTDATA_DIR, "alanine_manual_labels.magres"))
        self.kw = dict(xelement="H", yelement="H", yaxis_order="2Q",
                       references={"H": 29.5})

    def test_total_weight_matches_unreduced(self):
        """Each merged site contributes m(m-1) self-pair weight, so the total matches."""
        unreduced = NMRData2D(self.atoms.copy(), reduce=False,
                              correlation_strength_metric="fixed", **self.kw)
        reduced = NMRData2D(self.atoms.copy(), reduce=True,
                            correlation_strength_metric="fixed", **self.kw)
        total_unreduced = sum(p.multiplicity for p in unreduced.peaks)
        total_reduced = sum(p.multiplicity for p in reduced.peaks)
        self.assertEqual(total_reduced, total_unreduced)

    def test_diagonal_peaks_exist_when_reduced(self):
        d = NMRData2D(self.atoms.copy(), reduce=True,
                      correlation_strength_metric="distance", **self.kw)
        diagonal = [p for p in d.peaks if abs(p.y - 2 * p.x) < 1e-6]
        self.assertGreater(len(diagonal), 0)

    def test_merged_self_pair_distance_is_nearest_distinct_atom(self):
        """The self-pair of a merged site uses the closest distinct equivalent atom."""
        d = NMRData2D(self.atoms.copy(), reduce=True,
                      correlation_strength_metric="distance", **self.kw)
        mult = d.atoms.get_array("multiplicity")
        for k, (i, j) in enumerate(d.pairs):
            if i != j:
                continue
            self.assertGreater(mult[i], 1)
            members = d.site_map.members(int(i)).tolist()
            nearest_distinct = min(
                d.source.get_distance(members[0], m, mic=True)
                for m in members[1:]
            )
            self.assertLessEqual(d.pair_distances[k], nearest_distinct + 1e-8)


class TestPlotSettingsNotMutated(unittest.TestCase):
    """Plotting fills in limits on a copy; the caller's settings stay as given."""

    def test_settings_unchanged_after_plot(self):
        atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        d = NMRData2D(atoms=atoms, xelement="H", yelement="H", yaxis_order="2Q",
                      references={"H": 29.5}, correlation_strength_metric="fixed")
        settings = PlotSettings(plot_filename=None)
        before = settings.model_dump()
        NMRPlot2D(d, plot_settings=settings).plot()
        plt.close("all")
        self.assertEqual(settings.model_dump(), before)


class TestPlotlyAnnotations(unittest.TestCase):
    """Labels are placed in paper coordinates so inverted axes keep them on the edges."""

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly not installed")
    def test_labels_use_paper_coordinates_for_the_edges(self):
        backend = PlotlyBackend()
        backend.set_axis_properties("x", "y", (0.0, 10.0), (0.0, 10.0), True)
        backend.plot_annotations(
            np.array([2.0, 5.0]), np.array([3.0, 7.0]),
            ["C1", "C2"], ["H1", "H2"], PlotSettings(),
        )
        anns = list(backend.fig.layout.annotations)
        top = [a for a in anns if a.text in ("C1", "C2")]
        right = [a for a in anns if a.text in ("H1", "H2")]
        self.assertEqual(len(top), 2)
        self.assertEqual(len(right), 2)
        for a in top:
            self.assertEqual(a.yref, "paper")
            self.assertEqual(a.y, 1.0)
        for a in right:
            self.assertEqual(a.xref, "paper")
            self.assertEqual(a.x, 1.0)


class TestPairValidation(unittest.TestCase):
    """Explicit pairs must match the requested elements."""

    def test_pair_outside_element_sets_raises(self):
        from soprano.calculate.nmr.utils import process_pairs

        idx_x, idx_y = np.array([0, 1]), np.array([2, 3])
        with self.assertRaisesRegex(ValueError, "not made of the requested elements"):
            process_pairs(idx_x, idx_y, [(0, 9)])

    def test_matching_pair_is_kept(self):
        from soprano.calculate.nmr.utils import process_pairs

        idx_x, idx_y = np.array([0, 1]), np.array([2, 3])
        pairs, pairs_el_idx, _, _ = process_pairs(idx_x, idx_y, [(1, 3)])
        self.assertEqual(pairs, [(1, 3)])
        self.assertEqual(pairs_el_idx, [(1, 1)])


class TestBroadeningValidation(unittest.TestCase):
    """Broadening widths must be positive when given."""

    def test_nonpositive_broadening_rejected(self):
        for field in ("x_broadening", "y_broadening"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    PlotSettings(**{field: 0.0})
                with self.assertRaises(ValueError):
                    PlotSettings(**{field: -1.0})

    def test_unset_and_positive_broadening_accepted(self):
        self.assertIsNone(PlotSettings().x_broadening)
        self.assertEqual(PlotSettings(x_broadening=0.5).x_broadening, 0.5)


if __name__ == "__main__":
    unittest.main()
