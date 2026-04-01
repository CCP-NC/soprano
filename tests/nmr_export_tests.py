#!/usr/bin/env python
"""Focused tests for the standalone 2D NMR export module."""

import csv
import json
import os
import tempfile
import unittest

import numpy as np
from ase import io

from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.export import export_contour_data

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestNMRExportPublicAPI(unittest.TestCase):
    def test_package_level_import(self):
        from soprano.calculate.nmr import export_contour_data as package_export

        self.assertIs(package_export, export_contour_data)


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
        out = self._path("out.csv")
        export_contour_data(self.nmr_data, out, fmt="csv", grid_size=60)
        self.assertTrue(os.path.exists(out))

        with open(out, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["x_ppm", "y_ppm", "intensity"])
        self.assertGreater(len(rows), 10)

    def test_export_npz_contains_expected_arrays(self):
        out = self._path("out.npz")
        export_contour_data(self.nmr_data, out, fmt="npz", grid_size=40)
        self.assertTrue(os.path.exists(out))

        data = np.load(out, allow_pickle=True)
        for key in ("X", "Y", "Z", "peak_x", "peak_y", "xlims", "ylims"):
            self.assertIn(key, data.files)

    def test_export_json_requires_larmor(self):
        out = self._path("out.json")
        with self.assertRaises(ValueError):
            export_contour_data(self.nmr_data, out, fmt="json", grid_size=40)

    def test_export_json_writes_ssnake_fields(self):
        out = self._path("out.json")
        export_contour_data(
            self.nmr_data,
            out,
            fmt="json",
            grid_size=40,
            x_larmor_freq_mhz=100.0,
            y_larmor_freq_mhz=100.0,
        )
        self.assertTrue(os.path.exists(out))

        with open(out) as f:
            payload = json.load(f)

        for key in ("dataReal", "dataImag", "freq", "sw", "ref", "xaxArray", "metaData"):
            self.assertIn(key, payload)

    def test_export_simpson_writes_companion_peak_csv(self):
        out = self._path("out.spe")
        export_contour_data(
            self.nmr_data,
            out,
            fmt="simpson",
            grid_size=40,
            x_larmor_freq_mhz=100.0,
        )
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(out + ".peaks.csv"))


if __name__ == "__main__":
    unittest.main()
