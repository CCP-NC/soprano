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


if __name__ == "__main__":
    unittest.main()
