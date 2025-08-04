import os
import unittest
from ase.io import read
from soprano.calculate.nmr.simpson import write_spinsys
from soprano.selection import AtomSelection
import warnings

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestWriteSpinSys(unittest.TestCase):

    def setUp(self):
        # Use ethanol from test data magres file
        self.atoms = read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
    
    def test_deprecation_warning(self):
        """Test that a deprecation warning is raised for the old function name"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_spinsys(self.atoms)
            self.assertTrue(any("write_spinsys" in str(warn.message) for warn in w))

    def test_basic_output(self):
        """Test basic functionality with minimal parameters"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms)
        
        self.assertIn("spinsys", output)
        self.assertIn("channels", output)
        self.assertIn("nuclei", output)

        # Should contain 1H and 13C by default
        self.assertIn("1H", output)
        self.assertIn("13C", output)

    def test_custom_isotopes(self):
        """Test with custom isotope list"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, isotope_list=[2]*6+[13]*2 + [17])
        
        self.assertIn("2H", output)
        self.assertIn("13C", output)
        self.assertIn("17O", output)

    def test_magnetic_shielding(self):
        """Test magnetic shielding output"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, use_ms=True)
        
        self.assertIn("shift", output)

    def test_magnetic_shielding_isotropic(self):
        """Test isotropic magnetic shielding"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, use_ms=True, ms_iso=True)
        
        # Should have shift lines with zero anisotropy, zero asymmetry and zero angles
        self.assertIn("shift", output)
        self.assertIn("0.0p 0.0 0.0 0.0 0.0", output)

    def test_quadrupolar(self):
        """Test quadrupolar coupling output"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, q_order=2)
        # The 17O nucleus should be present with quadrupolar coupling
        self.assertIn("quadrupole 9 2", output)

    def test_dipolar(self):
        """Test dipolar coupling output"""
        sel = AtomSelection.from_element(self.atoms, 'H')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, dip_sel=sel)
        self.assertIn("dipole", output)

    def test_reference_values(self):
        """Test reference values for chemical shifts"""
        refs = {'H': 30.0, 'C': 170.0, 'O': 300.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, use_ms=True, ref=refs)
        
        self.assertIn("shift", output)

    def test_observed_nucleus(self):
        """Test observed nucleus specification"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = write_spinsys(self.atoms, obs_nuc='1H')
        
        # 1H should appear first in channels list
        channels_line = [line for line in output.split('\n') if 'channels' in line][0]
        self.assertTrue(channels_line.split()[1] == '1H')

    def test_invalid_quadrupolar_order(self):
        """Test error handling for invalid quadrupolar order"""
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_spinsys(self.atoms, q_order=3)

    def test_file_output(self):
        """Test writing to file"""
        test_file = "test_spinsys.txt"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_spinsys(self.atoms, path=test_file)
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, 'r') as f:
                content = f.read()
            self.assertIn("spinsys", content)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_invalid_observed_nucleus(self):
        """Test error handling for invalid observed nucleus"""
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_spinsys(self.atoms, obs_nuc='99Zz')

if __name__ == '__main__':
    unittest.main()