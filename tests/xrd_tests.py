#!/usr/bin/env python
"""
Test code for the calculate.xrd module
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import unittest
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa
# from soprano.calculate import xrd
# from soprano.calculate.xrd.xrd import XraySpectrum, XraySpectrumData

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestXRDCalculator(unittest.TestCase):
    def test_func_interface(self):

        from soprano.calculate import xrd

        xr = xrd.XRDCalculator()
        # Test that functions and function arguments are properly
        # accepted/rejected
        # Case 1:
        # Wrong kind of function

        def bad_func(x):
            return x

        try:
            assertRegex = self.assertRaisesRegex
        except AttributeError:
            # Python 3, how I loathe your arbitrary changes...
            assertRegex = self.assertRaisesRegexp

        assertRegex(
            ValueError,
            "Invalid peak_func passed to set_peak_func",
            xr.set_peak_func,
            bad_func,
        )
        # Case 2:
        # Right function, wrong arguments

        def good_func(x, w, a, b, c=0.2):
            return x * w * a * b * c

        bad_args = [0]
        assertRegex(
            ValueError,
            """Invalid number of peak_f_args passed to
                                    set_peak_func""",
            xr.set_peak_func,
            good_func,
            bad_args,
        )
        # Case 3:
        # All good
        good_args = [0, 0]
        try:
            xr.set_peak_func(peak_func=good_func, peak_f_args=good_args)
        except Exception:
            self.fail("Good function not accepted")

    def test_powder_peaks(self):

        from soprano.calculate import xrd

        xr = xrd.XRDCalculator()

        abc = [[3, 5, 10], [np.pi / 2, np.pi / 2, np.pi / 2]]
        peaks_nosym = xr.powder_peaks(latt_abc=abc)
        peaks_sym = xr.powder_peaks(latt_abc=abc, n=230, o=1)

        # A very crude test for now
        self.assertTrue(len(peaks_nosym.theta2) >= len(peaks_sym.theta2))

    def test_lebail_fit(self):

        from soprano.calculate import xrd
        from soprano.calculate.xrd.xrd import XraySpectrum

        xr = xrd.XRDCalculator()

        # Define a fake experimental spectrum to fit
        peak_n = 3
        th2_axis = np.linspace(0, np.pi, 1000)
        rng = np.random.default_rng(0)

        xpeaks = XraySpectrum(
            rng.random(peak_n) * np.pi,
            [],
            [],
            [],
            rng.random(peak_n) * 3.0 + 1.0,
            xr.lambdax,
        )

        # Build a simulated spectrum
        xpeaks_exp_mock, simul_peaks = xr.spec_simul(xpeaks, th2_axis)
        # Now clear the intensities
        np.copyto(xpeaks.intensity, np.ones(peak_n))
        # And carry out the leBail fit
        xpeaks, simul_spec, simul_peaks, rwp = xr.lebail_fit(xpeaks, xpeaks_exp_mock)

        self.assertAlmostEqual(rwp, 0.0, places=5)


class TestXRDRules(unittest.TestCase):
    def test_sel_rules(self):

        from soprano.calculate import xrd

        # Load the data from Colan's reference file
        ref_file_ends = ["mono"]

        for e in ref_file_ends:
            fname = os.path.join(_TESTDATA_DIR, "xrd_sel_test_{0}.txt".format(e))
            refdata = np.loadtxt(fname)

            n_o_pair = (0, 0)
            sel_rule = None

            for case in refdata:
                n, o, h, k, l, val = case
                n = int(n)
                o = int(o)
                if (n, o) != n_o_pair:
                    sel_rule = xrd.get_sel_rule_from_international(n, o)
                    n_o_pair = (n, o)
                self.assertEqual(sel_rule((h, k, l)), val)


if __name__ == "__main__":
    unittest.main()
