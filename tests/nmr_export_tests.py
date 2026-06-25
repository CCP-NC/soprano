#!/usr/bin/env python
"""Focused tests for the standalone 2D NMR export module."""

import csv
import json
import os
import tempfile
import unittest

import numpy as np
from ase import io

from soprano.calculate.nmr.config import PlotSettings
from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.export import export_contour_data

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestNMRExportPublicAPI(unittest.TestCase):
    def test_package_level_import(self):
        from soprano.calculate.nmr import export_contour_data as package_export

        self.assertIs(package_export, export_contour_data)

    def test_new_symbols_exported(self):
        from soprano.calculate.nmr import ExportConfig, compute_larmor_frequency, guess_format_from_path

        self.assertTrue(callable(compute_larmor_frequency))
        self.assertTrue(callable(guess_format_from_path))
        cfg = ExportConfig()
        self.assertIsNone(cfg.x_larmor_freq_mhz)


class TestExportConfig(unittest.TestCase):
    def test_defaults(self):
        from soprano.calculate.nmr.export import ExportConfig

        cfg = ExportConfig()
        self.assertIsNone(cfg.x_larmor_freq_mhz)
        self.assertEqual(cfg.grid_size, 500)
        self.assertEqual(cfg.broadening_type, "lorentzian")
        self.assertFalse(cfg.use_signed)

    def test_custom_values(self):
        from soprano.calculate.nmr.export import ExportConfig

        cfg = ExportConfig(grid_size=200, x_larmor_freq_mhz=400.0, use_signed=True)
        self.assertEqual(cfg.grid_size, 200)
        self.assertEqual(cfg.x_larmor_freq_mhz, 400.0)
        self.assertTrue(cfg.use_signed)


class TestLarmorFrequencyHelpers(unittest.TestCase):
    def test_compute_larmor_hydrogen_at_9_4_tesla(self):
        from soprano.calculate.nmr.export import compute_larmor_frequency

        freq = compute_larmor_frequency("H", b0_tesla=9.4)
        # 1H @ 9.4 T ≈ 400 MHz
        self.assertAlmostEqual(freq, 400.13, delta=1.0)

    def test_compute_larmor_carbon_at_same_field(self):
        from soprano.calculate.nmr.export import compute_larmor_frequency

        freq = compute_larmor_frequency("C", b0_tesla=9.4)
        # 13C @ 9.4 T ≈ 100 MHz
        self.assertAlmostEqual(freq, 100.61, delta=1.0)

    def test_compute_larman_scales_linearly_with_field(self):
        from soprano.calculate.nmr.export import compute_larmor_frequency

        f1 = compute_larmor_frequency("H", b0_tesla=9.4)
        f2 = compute_larmor_frequency("H", b0_tesla=18.8)
        self.assertAlmostEqual(f2, 2 * f1, places=1)

    def test_compute_larman_unknown_element_raises(self):
        from soprano.calculate.nmr.export import compute_larmor_frequency

        with self.assertRaises(ValueError):
            compute_larmor_frequency("Xx", b0_tesla=9.4)

    def test_compute_b0_from_spectrometer_freq_roundtrip(self):
        from soprano.calculate.nmr.export import compute_b0_from_spectrometer_freq, compute_larmor_frequency

        b0 = compute_b0_from_spectrometer_freq(600.0)
        freq_back = compute_larmor_frequency("H", b0_tesla=b0)
        self.assertAlmostEqual(freq_back, 600.0, places=3)

    def test_spectrometer_freq_produces_correct_b0_for_600mhz(self):
        from soprano.calculate.nmr.export import compute_b0_from_spectrometer_freq

        b0 = compute_b0_from_spectrometer_freq(600.0)
        # 600 MHz ¹H → ~14.09 T
        self.assertAlmostEqual(b0, 14.09, delta=0.05)



class TestFormatInference(unittest.TestCase):
    def test_known_extensions(self):
        from soprano.calculate.nmr.export import guess_format_from_path

        self.assertEqual(guess_format_from_path("spectrum.spe"), "simpson")
        self.assertEqual(guess_format_from_path("spectrum.npz"), "npz")
        self.assertEqual(guess_format_from_path("spectrum.csv"), "csv")
        self.assertEqual(guess_format_from_path("spectrum.json"), "json")
        self.assertEqual(guess_format_from_path("spectrum.ssnake"), "json")
        self.assertEqual(guess_format_from_path("spectrum.txt"), "plain")

    def test_unknown_extension_raises(self):
        from soprano.calculate.nmr.export import guess_format_from_path

        with self.assertRaises(ValueError):
            guess_format_from_path("spectrum.xyz")


class TestPlotSettingsRanges(unittest.TestCase):
    def test_shared_intensity_range_with_optional_overrides(self):
        default_shared = PlotSettings()
        self.assertEqual(default_shared.intensity_range, (10.0, 100.0))
        self.assertEqual(default_shared.contour_range, (10.0, 100.0))
        self.assertEqual(default_shared.heatmap_range, (10.0, 100.0))

        shared_custom = PlotSettings(intensity_range=(5.0, 80.0))
        self.assertEqual(shared_custom.contour_range, (5.0, 80.0))
        self.assertEqual(shared_custom.heatmap_range, (5.0, 80.0))

        split_layers = PlotSettings(
            intensity_range=(5.0, 80.0),
            contour_range=(20.0, 90.0),
            heatmap_range=(2.0, 50.0),
        )
        self.assertEqual(split_layers.contour_range, (20.0, 90.0))
        self.assertEqual(split_layers.heatmap_range, (2.0, 50.0))


class TestNMRExportModule(unittest.TestCase):
    def setUp(self):
        atoms = io.read(os.path.join(_TESTDATA_DIR, "EDIZUM.magres"))
        if isinstance(atoms, list):
            self.fail("Expected a single Atoms object from EDIZUM.magres")
        self.atoms = atoms
        self.nmr_data = NMRData2D(
            atoms=self.atoms,
            xelement="H",
            yelement="H",
            yaxis_order="2Q",
            references={"H": 29.5},
            correlation_strength_metric="fixed",
        )
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.tmpdir, name)

    def test_export_csv_grid(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.csv")
        export_contour_data(self.nmr_data, out, fmt="csv", config=ExportConfig(grid_size=60))
        self.assertTrue(os.path.exists(out))

        with open(out, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["x_ppm", "y_ppm", "intensity"])
        self.assertGreater(len(rows), 10)

    def test_export_npz_contains_expected_arrays(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.npz")
        export_contour_data(self.nmr_data, out, fmt="npz", config=ExportConfig(grid_size=40))
        self.assertTrue(os.path.exists(out))

        data = np.load(out, allow_pickle=True)
        for key in ("X", "Y", "Z", "peak_x", "peak_y", "xlims", "ylims"):
            self.assertIn(key, data.files)

    def test_export_npz_omits_peaks_when_disabled(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("no_peaks.npz")
        export_contour_data(
            self.nmr_data, out, fmt="npz",
            config=ExportConfig(grid_size=40, include_peaks=False),
        )
        data = np.load(out, allow_pickle=True)
        for key in ("X", "Y", "Z"):
            self.assertIn(key, data.files)
        for key in ("peak_x", "peak_y"):
            self.assertNotIn(key, data.files)

    def test_export_npz_respects_grid_max_scaling(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("scaled.npz")
        target_max = 1.0e6
        export_contour_data(
            self.nmr_data,
            out,
            fmt="npz",
            config=ExportConfig(grid_size=40, grid_max=target_max),
        )
        self.assertTrue(os.path.exists(out))

        data = np.load(out, allow_pickle=True)
        zmax = float(np.max(data["Z"]))
        self.assertAlmostEqual(zmax, target_max, places=6)

    def test_export_json_requires_larmor_when_auto_fails(self):
        """JSON export raises ValueError when Larmor freq can't be determined."""
        # Temporarily override element to something without gamma data
        original_xelement = self.nmr_data.xelement
        self.nmr_data.xelement = "Xx"
        try:
            from soprano.calculate.nmr.export import ExportConfig
            out = self._path("out.json")
            with self.assertRaises(ValueError):
                export_contour_data(self.nmr_data, out, fmt="json", config=ExportConfig(grid_size=40))
        finally:
            self.nmr_data.xelement = original_xelement

    def test_export_json_auto_computes_larmor(self):
        """JSON export auto-computes Larmor frequency from element."""
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.json")
        export_contour_data(
            self.nmr_data,
            out,
            fmt="json",
            config=ExportConfig(grid_size=40, b0_field_tesla=9.4),
        )
        self.assertTrue(os.path.exists(out))

        with open(out) as f:
            payload = json.load(f)

        self.assertIn("metaData", payload)
        self.assertIn("x_larmor_MHz", payload["metaData"])
        # 1H @ 9.4 T ≈ 400 MHz
        self.assertAlmostEqual(payload["metaData"]["x_larmor_MHz"], 400.13, delta=1.0)

    def test_export_json_includes_peaks(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.json")
        export_contour_data(
            self.nmr_data,
            out,
            fmt="json",
            config=ExportConfig(grid_size=40, x_larmor_freq_mhz=100.0, y_larmor_freq_mhz=100.0),
        )
        self.assertTrue(os.path.exists(out))

        with open(out) as f:
            payload = json.load(f)

        for key in ("dataReal", "dataImag", "freq", "sw", "ref", "xaxArray", "metaData", "peaks"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["peaks"], list)
        self.assertGreater(len(payload["peaks"]), 0)
        self.assertIn("x", payload["peaks"][0])
        self.assertIn("correlation_strength", payload["peaks"][0])

    def test_export_json_writes_ssnake_fields(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.json")
        export_contour_data(
            self.nmr_data,
            out,
            fmt="json",
            config=ExportConfig(grid_size=40, x_larmor_freq_mhz=100.0, y_larmor_freq_mhz=100.0),
        )
        self.assertTrue(os.path.exists(out))

        with open(out) as f:
            payload = json.load(f)

        for key in ("dataReal", "dataImag", "freq", "sw", "ref", "xaxArray", "metaData"):
            self.assertIn(key, payload)

    def test_export_simpson_writes_companion_peak_csv(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.spe")
        export_contour_data(
            self.nmr_data,
            out,
            fmt="simpson",
            config=ExportConfig(grid_size=40, x_larmor_freq_mhz=100.0),
        )
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(out + ".peaks.csv"))

    def test_export_auto_detects_format_from_extension(self):
        """Format should be inferred from file extension when fmt=None."""
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("auto.npz")
        export_contour_data(self.nmr_data, out, config=ExportConfig(grid_size=40))
        self.assertTrue(os.path.exists(out))

        data = np.load(out, allow_pickle=True)
        self.assertIn("Z", data.files)

    def test_export_plain_text_format(self):
        """Plain text export should produce space-separated columns and a peaks companion."""
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.txt")
        export_contour_data(self.nmr_data, out, fmt="plain", config=ExportConfig(grid_size=40))
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(out + ".peaks.csv"))

        with open(out) as f:
            lines = f.readlines()

        header_lines = [l for l in lines if l.startswith("#")]
        self.assertGreater(len(header_lines), 0)

        data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
        self.assertGreater(len(data_lines), 0)
        self.assertEqual(len(data_lines[0].strip().split()), 3)

    def test_export_plain_omits_peaks_when_disabled(self):
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("no_peaks.txt")
        export_contour_data(
            self.nmr_data, out, fmt="plain",
            config=ExportConfig(grid_size=40, include_peaks=False),
        )
        self.assertTrue(os.path.exists(out))
        self.assertFalse(os.path.exists(out + ".peaks.csv"))

    def test_export_csv_includes_peaks(self):
        """CSV export should produce a companion peaks file."""
        from soprano.calculate.nmr.export import ExportConfig
        out = self._path("out.csv")
        export_contour_data(self.nmr_data, out, fmt="csv", config=ExportConfig(grid_size=40))
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(out + ".peaks.csv"))

    def test_export_config_spectrometer_freq_computes_larmor(self):
        """ExportConfig.spectrometer_freq_mhz should auto-compute Larmor freqs."""
        from soprano.calculate.nmr.export import ExportConfig

        cfg = ExportConfig(spectrometer_freq_mhz=400.0)
        x, _ = cfg.resolve_larmor_freqs(self.nmr_data)
        # H @ 400 MHz spectrometer → x_larmor ≈ 400 MHz
        self.assertIsNotNone(x)
        self.assertAlmostEqual(x, 400.0, delta=1.0)

    def test_export_config_both_b0_and_spectrometer_freq_raises(self):
        """Specifying both b0_field_tesla and spectrometer_freq_mhz should raise."""
        from soprano.calculate.nmr.export import ExportConfig

        with self.assertRaises(ValueError):
            ExportConfig(b0_field_tesla=9.4, spectrometer_freq_mhz=600.0)

    def test_export_config_object(self):
        """Using ExportConfig should work equivalently to kwargs."""
        from soprano.calculate.nmr.export import ExportConfig

        out = self._path("config.npz")
        cfg = ExportConfig(grid_size=40, grid_max=1e6)
        export_contour_data(self.nmr_data, out, config=cfg)
        self.assertTrue(os.path.exists(out))

        data = np.load(out, allow_pickle=True)
        zmax = float(np.max(data["Z"]))
        self.assertAlmostEqual(zmax, 1e6, places=6)

    def test_export_reuses_cached_contour_data(self):
        """NMRData2D.export_contour_data reuses the cached grid when no rendering overrides are given."""
        from unittest.mock import patch

        # Prime the cache with a specific grid
        self.nmr_data.get_contour_data(grid_size=77)
        cached = self.nmr_data._contour_data
        self.assertEqual(cached.Z.shape[0], 77)

        out = self._path("reuse.npz")
        # Call the method with no rendering overrides — must reuse the cached grid
        with patch.object(self.nmr_data, 'get_contour_data', wraps=self.nmr_data.get_contour_data) as mock_cd:
            self.nmr_data.export_contour_data(out)
            mock_cd.assert_not_called()

        data = np.load(out, allow_pickle=True)
        # Grid shape should match the cached grid, not the ExportConfig default (500)
        self.assertEqual(data["Z"].shape[0], 77)

    def test_export_recomputes_when_rendering_override_given(self):
        """export_contour_data recomputes when a rendering param is explicitly overridden."""
        from soprano.calculate.nmr.export import ExportConfig

        # Prime the cache with grid_size=77
        self.nmr_data.get_contour_data(grid_size=77)

        out = self._path("override.npz")
        # Explicit grid_size override must trigger recomputation
        export_contour_data(self.nmr_data, out, config=ExportConfig(grid_size=50))

        data = np.load(out, allow_pickle=True)
        self.assertEqual(data["Z"].shape[0], 50)


if __name__ == "__main__":
    unittest.main()
