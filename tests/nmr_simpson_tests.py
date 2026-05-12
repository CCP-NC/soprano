import os
import unittest
import warnings
import pytest
from ase.io import read
from soprano.calculate.nmr.simpson import write_spinsys
from soprano.selection import AtomSelection

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


@pytest.mark.filterwarnings(
    "ignore:The Euler angles do not give a consistent rotation.*:UserWarning"
)
class TestWriteSpinSys(unittest.TestCase):

    def setUp(self):
        self.atoms = read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))

    def test_basic_output(self):
        """Basic spinsys structure is produced"""
        output = write_spinsys(self.atoms)
        self.assertIn("spinsys", output)
        self.assertIn("channels", output)
        self.assertIn("nuclei", output)
        self.assertIn("1H", output)
        self.assertIn("13C", output)

    def test_custom_isotopes(self):
        """Custom isotope list is applied correctly"""
        output = write_spinsys(self.atoms, isotope_list=[2]*6 + [13]*2 + [17])
        self.assertIn("2H", output)
        self.assertIn("13C", output)
        self.assertIn("17O", output)

    def test_magnetic_shielding(self):
        """use_ms=True produces shift lines"""
        refs = {'H': 30.0, 'C': 170.0, 'O': 300.0}
        output = write_spinsys(self.atoms, use_ms=True, ref=refs)
        self.assertIn("shift", output)

    @pytest.mark.filterwarnings("ignore:Gimbal lock detected.*:UserWarning")
    def test_magnetic_shielding_isotropic(self):
        """ms_iso=True produces shift lines with zero anisotropy and angles"""
        refs = {'H': 30.0, 'C': 170.0, 'O': 300.0}
        output = write_spinsys(self.atoms, use_ms=True, ms_iso=True, ref=refs)
        self.assertIn("shift", output)
        self.assertIn("0.0p 0.0 0.0 0.0 0.0", output)

    def test_quadrupolar(self):
        """q_order=2 produces quadrupole lines"""
        output = write_spinsys(self.atoms, q_order=2)
        self.assertIn("quadrupole", output)

    def test_dipolar(self):
        """dip_sel produces dipole lines"""
        sel = AtomSelection.from_element(self.atoms, 'H')
        output = write_spinsys(self.atoms, dip_sel=sel)
        self.assertIn("dipole", output)

    def test_observed_nucleus(self):
        """obs_nuc appears first in channels"""
        output = write_spinsys(self.atoms, obs_nuc='1H')
        channels_line = [line for line in output.split('\n') if 'channels' in line][0]
        self.assertEqual(channels_line.split()[1], '1H')

    def test_invalid_quadrupolar_order(self):
        """q_order > 2 raises ValueError"""
        with self.assertRaises(ValueError):
            write_spinsys(self.atoms, q_order=3)

    def test_q_order_type_error(self):
        """Non-integer q_order raises TypeError"""
        with self.assertRaises(TypeError):
            write_spinsys(self.atoms, q_order=1.5)

    def test_invalid_observed_nucleus(self):
        """Invalid obs_nuc raises ValueError"""
        with self.assertRaises(ValueError):
            write_spinsys(self.atoms, obs_nuc='99Zz')

    def test_file_output(self):
        """path= argument writes a valid file"""
        test_file = "test_spinsys.txt"
        try:
            write_spinsys(self.atoms, path=test_file)
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, 'r') as f:
                content = f.read()
            self.assertIn("spinsys", content)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_custom_ms_tag(self):
        """Custom ms_tag reads shielding from the correct array"""
        atoms = read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        atoms.new_array("ms_custom", atoms.get_array("ms"))
        refs = {'H': 30.0, 'C': 170.0, 'O': 300.0}
        output = write_spinsys(atoms, use_ms=True, ms_tag="ms_custom", ref=refs)
        self.assertIn("shift", output)

    def test_custom_efg_tag(self):
        """Custom efg_tag reads EFG from the correct array"""
        atoms = read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        atoms.new_array("efg_custom", atoms.get_array("efg"))
        output = write_spinsys(atoms, q_order=2, efg_tag="efg_custom")
        self.assertIn("quadrupole", output)

    def test_legacy_backend_deprecation(self):
        """backend='legacy' raises a DeprecationWarning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_spinsys(self.atoms, backend='legacy')
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_invalid_backend(self):
        """Unknown backend raises ValueError"""
        with self.assertRaises(ValueError):
            write_spinsys(self.atoms, backend='unknown')

    def test_include_cross_terms_false_suppresses_cross_terms(self):
        """Regression test: include_cross_terms must be threaded
        through write_spinsys -> _write_spinsys_spinsys -> SpinSystem.to_simpson.

        Before the fix, write_spinsys had no include_cross_terms parameter and
        always used the default (True) inside to_simpson.
        """
        # Verify the parameter is accepted without TypeError
        output_with = write_spinsys(self.atoms, q_order=2, include_cross_terms=True)
        output_without = write_spinsys(self.atoms, q_order=2, include_cross_terms=False)

        # Both outputs must still contain valid spinsys structure
        for output in [output_with, output_without]:
            self.assertIn("spinsys", output)

        # When cross-terms are disabled, neither cross-term keyword must appear
        self.assertNotIn("quadrupole_x_dipole", output_without)
        self.assertNotIn("quadrupole_x_shift", output_without)


if __name__ == '__main__':
    unittest.main()

