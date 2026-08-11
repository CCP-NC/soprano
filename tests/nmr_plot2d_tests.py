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

from soprano.calculate.nmr.backends import MatplotlibBackend, _resolve_levels
from soprano.calculate.nmr.config import PlotSettings
from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.nmr import Peak2D
from soprano.calculate.nmr.plot2d import NMRPlot2D
from soprano.calculate.nmr.utils import generate_contour_map, generate_peaks

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

    def test_zero_grid_does_not_crash(self):
        """A grid whose values are all zero returns a benign single level."""
        Z = np.zeros((4, 4))
        levels = _resolve_levels(Z, 10, (10.0, 100.0))
        self.assertEqual(list(levels), [0.0])

    def test_all_nan_grid_does_not_crash(self):
        Z = np.full((4, 4), np.nan)
        levels = _resolve_levels(Z, 10, (10.0, 100.0))
        self.assertEqual(list(levels), [0.0])
        self.assertTrue(np.all(np.isfinite(levels)))


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


if __name__ == "__main__":
    unittest.main()
