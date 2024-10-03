#!/usr/bin/env python
"""
Test code for NMR calculations - i.e. in the calculate/nmr directory
"""


import os
import unittest
from typing import List, Tuple

import numpy as np
from ase.build import bulk

from soprano.calculate.nmr.nmr import Peak2D, get_pair_dipolar_couplings, merge_peaks

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")




class TestMergePeaks(unittest.TestCase):

    def setUp(self):
        # Create some sample peaks for testing
        self.peaks: List[Peak2D] = [
            Peak2D(x=1.0, y=2.0000001, correlation_strength=0.5, xlabel='H1', ylabel='H2', idx_x=0, idx_y=1, color='red'),
            Peak2D(x=1.0, y=2.0000001, correlation_strength=0.5, xlabel='H1', ylabel='H2', idx_x=0, idx_y=1, color='red'),
            Peak2D(x=1.0, y=2.0999999, correlation_strength=0.5, xlabel='H1', ylabel='H3', idx_x=0, idx_y=2, color='blue'),
            Peak2D(x=1.2, y=2.2000003, correlation_strength=0.5, xlabel='H2', ylabel='H3', idx_x=1, idx_y=0, color='green'),
            Peak2D(x=1.2, y=2.2000004, correlation_strength=0.7, xlabel='H2', ylabel='H3', idx_x=2, idx_y=3, color='yellow')
        ]

    def test_merge_identical_peaks(self):
        merged_peaks = merge_peaks(self.peaks, xtol=1e-5, ytol=1e-5, corr_tol=1e-5, ignore_correlation_strength=False)
        self.assertEqual(len(merged_peaks), 4)
        self.assertEqual(merged_peaks[0].xlabel, 'H1')
        self.assertEqual(merged_peaks[0].ylabel, 'H2')

    def test_merge_with_tolerance(self):
        # peaks 0, 1, 2 should be merged
        merged_peaks = merge_peaks(self.peaks, xtol=0.1, ytol=0.1, corr_tol=0.1, ignore_correlation_strength=False)
        print(merged_peaks)
        self.assertEqual(len(merged_peaks), 3)
        self.assertEqual(merged_peaks[0].xlabel, 'H1')
        self.assertEqual(merged_peaks[0].ylabel, 'H2/H3')

    def test_merge_ignore_correlation_strength(self):
        # peaks 0, 1 should be merged and 3 and 4 should be merged
        merged_peaks = merge_peaks(self.peaks, xtol=1e-5, ytol=1e-5, corr_tol=1e-5, ignore_correlation_strength=True)
        self.assertEqual(len(merged_peaks), 3)
        self.assertEqual(merged_peaks[0].xlabel, 'H1')
        self.assertEqual(merged_peaks[0].ylabel, 'H2')
        self.assertEqual(merged_peaks[2].xlabel, 'H2')
        self.assertEqual(merged_peaks[2].ylabel, 'H3')



class DipolarCoupling:
    @staticmethod
    def get(atoms, sel_i, sel_j, isotopes):
        # Mock implementation of DipolarCoupling.get
        return {0: {0: 1.0}}

class TestGetPairDipolarCouplings(unittest.TestCase):

    def setUp(self):
        # Create a mock Atoms object
        self.atoms = bulk('NaCl', 'rocksalt', a=5.64, cubic=True)
        self.pairs: List[Tuple[int, int]] = [(0, 1), # Na-Cl
                                             (1, 1), # Same atom -> should return 0
                                             (0, 2), # Na-Na
                                             (1, 3)  # Cl-Cl
                                             ]
        self.isotopes = {'Na': 23, 'Cl': 35}
        self.reference_dipolar_couplings = [-0.139069,
                                             0.000000,
                                            -0.132671,
                                            -0.018222]

    def test_dipolar_couplings(self):
        dipolar_couplings = get_pair_dipolar_couplings(self.atoms, self.pairs, self.isotopes)
        self.assertTrue(np.allclose(dipolar_couplings, self.reference_dipolar_couplings))

    def test_dipolar_couplings_no_isotopes(self):
        # Same as before since those were the default isotopes...
        dipolar_couplings = get_pair_dipolar_couplings(self.atoms, self.pairs)
        self.assertTrue(np.allclose(dipolar_couplings, self.reference_dipolar_couplings))

    def test_dipolar_couplings_zero_for_same_indices(self):
        pairs_with_same_indices = [(0, 0), (1, 1), (2, 2)]
        dipolar_couplings = get_pair_dipolar_couplings(self.atoms, pairs_with_same_indices, self.isotopes)
        self.assertEqual(dipolar_couplings, [0, 0, 0])

if __name__ == '__main__':
    unittest.main()
