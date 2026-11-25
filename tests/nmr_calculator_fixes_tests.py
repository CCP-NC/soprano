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
        calc.set_reference(170.0, "C", gradient=-1.0)  # exactly shift = ref - ms
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
        calc.set_reference(170.0, "C", gradient=-1.0)  # exactly shift = ref - ms
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

    def test_positive_shift_is_positive_mhz_offset(self):
        """MHz offsets follow the shift for negative gamma too (IUPAC, |nu0|)."""
        from soprano.properties.nmr import MSShift

        self.calc.set_reference(300.0, "17O")
        o = self.eth.get_chemical_symbols().index("O")
        delta = MSShift.get(self.eth[[o]], references={"O": 300.0})[0]
        self.assertGreater(delta, 0)
        larm = abs(self.calc.get_larmor_frequency("17O"))
        spec, freq = self.calc.spectrum_1d(
            "O", min_freq=0.0, max_freq=2 * delta * larm * 1e-6, bins=2001,
            freq_broad=0.02 * delta * larm * 1e-6, freq_units="MHz",
        )
        self.assertAlmostEqual(freq[np.argmax(spec)], delta * larm * 1e-6, delta=2e-3 * delta * larm * 1e-6)


def _peak(spec, freq):
    return freq[np.argmax(spec)]


class SpectrumReferencingTestCase(unittest.TestCase):
    """spectrum_1d places peaks where MSShift and the quadrupolar theory put them."""

    @classmethod
    def setUpClass(cls):
        cls.eth = io.read(_TESTDATA_DIR / "ethanol.magres")
        cls.o = cls.eth.get_chemical_symbols().index("O")

    def _single(self, symbol):
        idx = self.eth.get_chemical_symbols().index(symbol)
        return self.eth[[idx]]

    def test_default_reference_uses_full_formula(self):
        from soprano.properties.nmr import MSShift

        c = self._single("C")
        delta = MSShift.get(c, references={"C": 170.0})[0]
        calc = NMRCalculator(c)
        calc.set_reference(170.0, "C")
        spec, freq = calc.spectrum_1d("C", min_freq=delta - 1, max_freq=delta + 1,
                                      bins=20001, freq_broad=0.01)
        self.assertAlmostEqual(_peak(spec, freq), delta, places=3)

    def test_gradient_is_applied(self):
        from soprano.properties.nmr import MSShift

        c = self._single("C")
        delta = MSShift.get(c, references={"C": 170.0}, gradients={"C": -0.9})[0]
        calc = NMRCalculator(c)
        calc.set_reference(170.0, "C", gradient=-0.9)
        spec, freq = calc.spectrum_1d("C", min_freq=delta - 1, max_freq=delta + 1,
                                      bins=20001, freq_broad=0.01)
        self.assertAlmostEqual(_peak(spec, freq), delta, places=3)

    def test_multiplicity_weights_sites(self):
        h = [i for i, s in enumerate(self.eth.get_chemical_symbols()) if s == "H"]
        sub = self.eth[h[:2]]
        sub.set_array("multiplicity", np.array([3, 1]))
        calc = NMRCalculator(sub)
        spec, freq = calc.spectrum_1d("H", min_freq=20, max_freq=40, bins=4001, freq_broad=0.02)
        from soprano.properties.nmr import MSIsotropy

        iso = MSIsotropy.get(sub)
        if abs(iso[0] - iso[1]) > 0.5:
            heights = [spec[np.argmin(abs(freq - v))] for v in iso]
            self.assertAlmostEqual(heights[0] / heights[1], 3.0, places=2)
        self.assertAlmostEqual(spec.sum() / len(spec), 4.0)

    def test_transitions_weighted_by_transition_probability(self):
        """1Q transitions m -> m+1 carry I(I+1) - m(m+1): 5:8:9:8:5 for I = 5/2."""
        from scipy.signal import find_peaks

        o = self.eth[[self.o]]
        calc = NMRCalculator(o)
        calc.set_element_isotope("O", 17)
        calc.set_single_crystal(0.3, 0.2)  # first order: five separate lines
        spec, _ = calc.spectrum_1d(
            "17O", effects=NMRFlags.Q_1_ORIENT, min_freq=-4, max_freq=4, bins=8001,
            freq_broad=0.01, freq_units="MHz", use_reference=True,
        )
        idx, _ = find_peaks(spec, height=0.2 * spec.max())
        np.testing.assert_allclose(spec[idx] / spec[idx].min() * 5, [5, 8, 9, 8, 5], rtol=1e-6)

    def test_second_order_quadrupolar_shift_is_upfield(self):
        """The central-transition shift is -(3/40) (CQ/(I(2I-1)))² (I(I+1)-3/4)(1+η²/3)/ν0."""
        from soprano.properties.nmr import EFGAsymmetry, EFGQuadrupolarConstant, MSShift

        o = self.eth[[self.o]]
        calc = NMRCalculator(o)
        calc.set_element_isotope("O", 17)
        calc.set_reference(300.0, "17O")
        cq = EFGQuadrupolarConstant.get(o, isotopes={"O": 17})[0]
        eta = EFGAsymmetry.get(o)[0]
        I = 2.5
        nu0 = abs(calc.get_larmor_frequency("17O")) * 1e6
        nu_qis = -(3 / 40) * (cq / (I * (2 * I - 1))) ** 2 * (I * (I + 1) - 0.75) * (1 + eta ** 2 / 3) / nu0
        expected = MSShift.get(o, references={"O": 300.0})[0] + nu_qis / nu0 * 1e6
        self.assertLess(nu_qis, 0)

        spec, freq = calc.spectrum_1d(
            "O", min_freq=expected - 50, max_freq=expected + 50, bins=10001,
            freq_broad=0.1, effects=NMRFlags.CS_ISO | NMRFlags.Q_2_SHIFT, use_central=True,
        )
        self.assertAlmostEqual(_peak(spec, freq), expected, delta=0.05)


if __name__ == "__main__":
    unittest.main()
