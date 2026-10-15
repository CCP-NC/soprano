#!/usr/bin/env python
"""
Regression tests for NMRCalculator: isotope symbols, spectrum_1d defaults and
windows, and negative-gamma nuclei with MHz axes.
"""


from pathlib import Path
import unittest

import numpy as np
from ase import io

from soprano.calculate.nmr.nmr import NMRCalculator, NMRFlags
from soprano.calculate.powder.triavg import TriAvg

_TESTDATA_DIR = Path(__file__).parent.resolve() / "test_data"


class SetElementIsotopeTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eth = io.read(_TESTDATA_DIR / "ethanol.magres")

    def setUp(self):
        self.calc = NMRCalculator(self.eth)

    def test_integer_isotope(self):
        self.calc.set_element_isotope("O", 17)
        self.assertEqual(self.calc._isos[8], "17")

    def test_symbol_isotope(self):
        self.calc.set_element_isotope("O", "17O")
        self.assertEqual(self.calc._isos[8], "17")

    def test_other_elements_unchanged(self):
        self.calc.set_element_isotope("O", "17O")
        self.assertEqual(list(self.calc._isos[:6]), ["1"] * 6)
        self.assertEqual(list(self.calc._isos[6:8]), ["13", "13"])

    def test_mismatched_symbol_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid element"):
            self.calc.set_element_isotope("O", "13C")

    def test_invalid_integer_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid isotope 99"):
            self.calc.set_element_isotope("O", 99)

    def test_set_isotopes_accepts_symbol(self):
        isos = [None] * len(self.eth)
        isos[8] = "17O"
        self.calc.set_isotopes(isos)
        self.assertEqual(self.calc._isos[8], "17")


class SpectrumDefaultsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eth = io.read(_TESTDATA_DIR / "ethanol.magres")

    def test_default_is_powder(self):
        calc = NMRCalculator(self.eth)
        expected = TriAvg("hemisphere").get_orient_points(8)[0]
        self.assertEqual(calc._orients[0].shape, expected.shape)
        self.assertGreater(len(calc._orients[2]), 0)

    def test_isotropic_spectrum_without_orientation_setup(self):
        calc = NMRCalculator(self.eth)
        spec, freq = calc.spectrum_1d("C", min_freq=-50, max_freq=250, bins=301)
        ref = NMRCalculator(self.eth)
        ref.set_powder(N=8, mode="hemisphere")
        spec_ref, _ = ref.spectrum_1d("C", min_freq=-50, max_freq=250, bins=301)
        np.testing.assert_allclose(spec, spec_ref)

    def test_reference_window_is_on_shift_axis(self):
        calc = NMRCalculator(self.eth)
        calc.set_reference(170.0, "C")
        spec, freq = calc.spectrum_1d(
            "C", min_freq=-20, max_freq=200, bins=221, freq_broad=1.0,
            effects=NMRFlags.CS_ISO,
        )
        np.testing.assert_allclose(freq, np.linspace(-20, 200, 221))
        # Shifts [-20, 200] are shieldings [-30, 190] reversed
        shielding = NMRCalculator(self.eth)
        spec_sig, _ = shielding.spectrum_1d(
            "C", min_freq=-30, max_freq=190, bins=221, freq_broad=1.0,
            effects=NMRFlags.CS_ISO,
        )
        np.testing.assert_allclose(spec, spec_sig[::-1], rtol=1e-6, atol=1e-9)

    def test_reference_window_powder_broadening(self):
        calc = NMRCalculator(self.eth)
        calc.set_reference(170.0, "C")
        spec, freq = calc.spectrum_1d(
            "C", min_freq=-20, max_freq=200, bins=221, freq_broad=1.0,
            effects=NMRFlags.CS,
        )
        shielding = NMRCalculator(self.eth)
        spec_sig, _ = shielding.spectrum_1d(
            "C", min_freq=-30, max_freq=190, bins=221, freq_broad=1.0,
            effects=NMRFlags.CS,
        )
        np.testing.assert_allclose(freq, np.linspace(-20, 200, 221))
        np.testing.assert_allclose(spec, spec_sig[::-1], rtol=1e-6, atol=1e-9)

    def test_freq_broad_follows_mhz_units(self):
        calc = NMRCalculator(self.eth)
        larm = calc.get_larmor_frequency("13C")
        u = 1e6 / larm
        spec_mhz, freq_mhz = calc.spectrum_1d(
            "C", min_freq=-0.03, max_freq=0.03, bins=1201,
            freq_broad=3.0 / abs(u), freq_units="MHz", effects=NMRFlags.CS_ISO,
        )
        spec_ppm, freq_ppm = calc.spectrum_1d(
            "C", min_freq=-0.03 * u, max_freq=0.03 * u, bins=1201,
            freq_broad=3.0, freq_units="ppm", effects=NMRFlags.CS_ISO,
        )
        np.testing.assert_allclose(freq_mhz * u, freq_ppm)
        np.testing.assert_allclose(spec_mhz, spec_ppm, rtol=1e-6, atol=1e-9)


class NegativeGammaTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eth = io.read(_TESTDATA_DIR / "ethanol.magres")

    def setUp(self):
        self.calc = NMRCalculator(self.eth)
        self.calc.set_element_isotope("O", 17)

    def test_mhz_matches_ppm_for_negative_gamma(self):
        larm = self.calc.get_larmor_frequency("17O")
        self.assertLess(larm, 0)
        u = 1e6 / larm
        half = 2000 / abs(u)
        spec_mhz, freq_mhz = self.calc.spectrum_1d(
            "O", min_freq=-half, max_freq=half, bins=401,
            freq_broad=20.0 / abs(u), freq_units="MHz",
            effects=NMRFlags.Q_1_ORIENT,
        )
        spec_ppm, freq_ppm = self.calc.spectrum_1d(
            "O", min_freq=-half * u, max_freq=half * u, bins=401,
            freq_broad=20.0, freq_units="ppm",
            effects=NMRFlags.Q_1_ORIENT,
        )
        self.assertTrue(np.all(np.diff(freq_mhz) > 0))
        self.assertGreater(spec_mhz.max(), 0)
        np.testing.assert_allclose(freq_mhz * u, freq_ppm)
        np.testing.assert_allclose(spec_mhz, spec_ppm, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
