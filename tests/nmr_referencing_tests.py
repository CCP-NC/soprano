#!/usr/bin/env python
"""
Tests for chemical shift referencing: full formula versus linear fit,
chosen per element.
"""


from pathlib import Path
import logging
import unittest
import warnings

import numpy as np
from ase import io

from soprano.nmr.utils import _gradients_to_list
from soprano.properties.nmr.ms import MSShielding, MSShift, MSTensor
from soprano.properties.nmr.spinsystem import NMRSpinSystem

_TESTDATA_DIR = Path(__file__).parent.resolve() / "test_data"

REFS = {"H": 31.0, "C": 170.0, "O": 20.0}


def _full(sigma, ref):
    return (ref - sigma) / (1 - ref * 1e-6)


def _linear(sigma, ref, grad):
    return ref + grad * sigma


class ReferencingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eth = io.read(_TESTDATA_DIR / "ethanol.magres")
        cls.symbols = cls.eth.get_chemical_symbols()
        cls.sigma = np.array(MSShielding.get(cls.eth, save_array=False))

    def expected(self, refs, grads):
        """Independent per-site reference values from the isotropic shieldings."""
        out = []
        for el, sig in zip(self.symbols, self.sigma):
            grad = None if grads is None else grads.get(el)
            if grad is None:
                out.append(_full(sig, refs[el]))
            else:
                out.append(_linear(sig, refs[el], grad))
        return np.array(out)

    def shift(self, gradients=None, refs=REFS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return MSShift.get(self.eth, references=refs, gradients=gradients, save_array=False)


class TestFormulas(ReferencingTestCase):

    def test_full_formula_value(self):
        delta = self.shift(None)
        np.testing.assert_allclose(delta, self.expected(REFS, None), rtol=1e-12)
        # sanity: the denominator is not negligible for a 170 ppm reference
        ref_c = self.sigma[self.symbols.index("C")]
        self.assertGreater(abs(_full(ref_c, 170.0) - _linear(ref_c, 170.0, -1.0)), 1e-3)

    def test_linear_value(self):
        grads = {"H": -0.9, "C": -0.98, "O": -0.95}
        delta = self.shift(grads)
        np.testing.assert_allclose(delta, self.expected(REFS, grads), rtol=1e-12)

    def test_none_and_empty_dict_identical(self):
        np.testing.assert_array_equal(self.shift(None), self.shift({}))

    def test_gradients_to_list_empty_dict_is_none(self):
        self.assertIsNone(_gradients_to_list({}, self.symbols))
        self.assertIsNone(_gradients_to_list(None, self.symbols))

    def test_gradients_to_list_partial_dict_has_none_entries(self):
        grads = _gradients_to_list({"C": -0.98}, self.symbols)
        for el, grad in zip(self.symbols, grads):
            if el == "C":
                self.assertEqual(grad, -0.98)
            else:
                self.assertIsNone(grad)


class TestMixedMode(ReferencingTestCase):

    def test_partial_dict_per_element(self):
        grads = {"C": -0.98}
        with self.assertWarns(UserWarning) as cm:
            delta = MSShift.get(self.eth, references=REFS, gradients=grads, save_array=False)
        np.testing.assert_allclose(delta, self.expected(REFS, grads), rtol=1e-12)
        message = str(cm.warning)
        self.assertIn("H", message)
        self.assertIn("O", message)
        self.assertNotIn("C,", message)

    def test_partial_dict_warning_names_only_fallback_elements(self):
        grads = {"C": -0.98, "H": -0.9}
        with self.assertWarns(UserWarning) as cm:
            MSShift.get(self.eth, references=REFS, gradients=grads, save_array=False)
        self.assertIn("O", str(cm.warning))
        self.assertNotIn("No gradient given for C", str(cm.warning))
        self.assertNotIn("No gradient given for H", str(cm.warning))

    def test_full_and_none_gradients_no_fallback_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.shift(None)
            self.shift({})
        self.assertEqual([w for w in caught if "No gradient" in str(w.message)], [])

    def test_info_log_states_each_mode(self):
        with self.assertLogs("soprano.properties.nmr.ms", level=logging.INFO) as logs:
            self.shift({"C": -0.98})
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn("H=full formula", message)
        self.assertIn("C=linear fit (m=-0.98)", message)
        self.assertIn("O=full formula", message)

    def test_mstensor_shift_consistent_with_msshift_mixed(self):
        grads = {"C": -0.98}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensors = MSTensor.get(self.eth, references=REFS, gradients=grads)
            shifts = MSShift.get(self.eth, references=REFS, gradients=grads, save_array=False)
        tensor_shifts = np.array([t.shift for t in tensors])
        np.testing.assert_allclose(tensor_shifts, shifts, rtol=1e-10)
        np.testing.assert_allclose(tensor_shifts, self.expected(REFS, grads), rtol=1e-10)


class TestGradientValues(ReferencingTestCase):

    def test_exact_minus_one_is_linear_and_warns(self):
        with self.assertWarns(UserWarning) as cm:
            delta = MSShift.get(self.eth, references=REFS, gradients=-1.0, save_array=False)
        np.testing.assert_allclose(delta, self.expected(REFS, {"H": -1.0, "C": -1.0, "O": -1.0}), rtol=1e-12)
        message = str(cm.warning)
        self.assertIn("exactly -1", message)
        self.assertIn("C", message)

    def test_int_gradient_accepted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            as_int = MSShift.get(self.eth, references=REFS, gradients={"C": -1}, save_array=False)
            as_float = MSShift.get(self.eth, references=REFS, gradients={"C": -1.0}, save_array=False)
        np.testing.assert_array_equal(as_int, as_float)

    def test_gradients_to_list_accepts_numbers_not_bool(self):
        self.assertEqual(_gradients_to_list(-1, self.symbols), [-1.0] * len(self.symbols))
        self.assertEqual(_gradients_to_list(np.float64(-0.9), self.symbols)[0], -0.9)
        with self.assertRaises(ValueError):
            _gradients_to_list(True, self.symbols)

    def test_gradient_outside_range_warns(self):
        with self.assertWarns(UserWarning) as cm:
            MSShift.get(self.eth, references=REFS, gradients=-2.0, save_array=False)
        self.assertIn("outside the range", str(cm.warning))

    def test_large_reference_no_longer_warns_in_linear_mode(self):
        refs = {"H": 31.0, "C": 800.0, "O": 20.0}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MSShift.get(self.eth, references=refs, gradients={"C": -0.98, "H": -0.9, "O": -0.95}, save_array=False)
        self.assertEqual([w for w in caught if "denominator" in str(w.message)], [])


class TestSpinSystemDefault(ReferencingTestCase):

    def test_default_gradient_is_full_formula(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spin_system = NMRSpinSystem.get(self.eth, references=REFS)
        for site, el, sig in zip(spin_system.sites, self.symbols, self.sigma):
            if el not in REFS:
                continue
            self.assertAlmostEqual(site.ms.shift, _full(sig, REFS[el]), places=8)


if __name__ == "__main__":
    unittest.main()
