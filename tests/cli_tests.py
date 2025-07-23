#!/usr/bin/env python
"""
Test code for the Command Line Interface for Soprano
"""


import io
import logging
import os
import sys
import unittest
from logging.handlers import MemoryHandler
from tempfile import NamedTemporaryFile
from unittest.mock import patch
from pathlib import Path  # Add this import

import pandas as pd
from ase.io import read
from click.testing import CliRunner

from soprano.scripts.cli import soprano

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
class TestCLI(unittest.TestCase):
    def setUp(self):
        # Use Path for cross-platform compatibility
        self._TEST_DIR = Path(__file__).parent
        self._TESTDATA_DIR = self._TEST_DIR / "test_data"
        self._TESTSAVE_DIR = self._TEST_DIR / "test_save"
        # Create an in-memory logging handler
        self.log_stream = io.StringIO()
        self.handler = MemoryHandler(1024*10, target=logging.StreamHandler(self.log_stream))
        logger = logging.getLogger('cli')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.handler)

    def tearDown(self):
        # Remove the in-memory logging handler
        logger = logging.getLogger('cli')
        logger.removeHandler(self.handler)
        self.handler.close()
        
        # Clean up any test files
        for file in self._TESTSAVE_DIR.glob('*.cif'):
            file.unlink(missing_ok=True)



    def test_read_valid_ms(self):
        # Load the data calculated with MagresView
        fname_data = self._TESTDATA_DIR / "ethanol_ms.dat"

        # Handle Windows line endings in the data file
        ref_df = pd.read_csv(
            fname_data,
            sep=r"\s+|\t",
            lineterminator=None,  # Let pandas detect line endings
            skiprows=7,
            engine="python"
        )
        nsites = len(ref_df)

        # get the same data use CLI
        runner = CliRunner()
        with patch('click_log.basic_config'):
            with NamedTemporaryFile(delete=False) as temp_csv:
                fname_mag = self._TESTDATA_DIR / "ethanol.magres"
                option_flags = [
                    "-p", "ms",
                    "--precision", "9",
                    "-o", str(temp_csv.name),  # Convert Path to string
                    "--output-format", "csv",
                    "-v",
                ]
                result = runner.invoke(
                    soprano, ["nmr", str(fname_mag)] + option_flags,  # Convert Path to string
                    prog_name="nmr"
                )
                # all went smoothly?
                self.assertEqual(result.exit_code, 0)

                # test to see that we parsed the correct file
                output = result.output.strip().split("\n")
                # Convert both paths to strings for comparison
                self.assertEqual(output[3], str(fname_mag))

                # and extracted the right number of sites' results
                df = pd.read_csv(temp_csv.name)
                self.assertEqual(len(df), nsites)
                # make sure the labels are right
                ref_labels = ref_df["Atom"]
                cli_labels = df["labels"]
                pd.testing.assert_series_equal(ref_labels, cli_labels, check_names=False)

                # make sure numbers are correct
                ref_iso = ref_df["s_iso(ppm)"]
                cli_iso = df["MS_shielding/ppm"]
                pd.testing.assert_series_equal(ref_iso, cli_iso, check_names=False)

    def test_reduce_by_cif_labels(self):
        """
        The file EDIZUM.magres in the test_data directory has four identical molecules in the unit cell
        with cif-style labels. We should be able to reduce this to a single molecule with the correct
        labels and multiplicities.
        """
        runner = CliRunner()
        with patch('click_log.basic_config'):
            with NamedTemporaryFile() as temp_csv:
                fname_mag = self._TESTDATA_DIR / "EDIZUM.magres"  # Use Path
                option_flags = [
                    "-o",
                    temp_csv.name,
                    "--output-format",
                    "csv",
                ]
                result = runner.invoke(
                    soprano, ["nmr", str(fname_mag)] + option_flags, prog_name="nmr"
                )
                # all went smoothly?
                self.assertEqual(result.exit_code, 0)

                # test to see that we parsed the correct file
                output = result.output.strip().split("\n")
                self.assertEqual(output[3], str(fname_mag))

                # and extracted the right number of sites' results
                df = pd.read_csv(temp_csv.name)
                self.assertEqual(len(df), 37)

                # make sure the multiplicity for each site is 4
                self.assertEqual(df["multiplicity"].nunique(), 1)
                self.assertEqual(df["multiplicity"].unique()[0], 4)

    def test_reduce_by_symmetry(self):
        """
        The file nacl.magres in the test_data directory is NaCl in a cubic cell -> 4 units of NaCl.
        I ran the Castep NMR calculation with symmetry on so we should get the same isotropy at each equivalent site.

        I did not use cif-style labels, so in this case we rely on the symmetry-finding algorithm to identify
        equivalent sites.
        """
        runner = CliRunner()
        with patch('click_log.basic_config'):
            with NamedTemporaryFile() as temp_csv:
                fname_mag = self._TESTDATA_DIR / "nacl.magres"
                option_flags = [
                    "-o",
                    temp_csv.name,
                    "--output-format",
                    "csv",
                ]
                result = runner.invoke(
                    soprano, ["nmr", str(fname_mag)] + option_flags, prog_name="nmr"
                )
                # all went smoothly?
                self.assertEqual(result.exit_code, 0)

                # test to see that we parsed the correct file
                output = result.output.strip().split("\n")
                self.assertEqual(output[3], str(fname_mag))

                # and extracted the right number of sites' results
                df = pd.read_csv(temp_csv.name)
                self.assertEqual(len(df), 2)

                # make sure the isotropy for each site is the same
                self.assertEqual(df["MS_shielding/ppm"].nunique(), 2)
                # Na_1 should have MS_shielding of 433.819
                self.assertAlmostEqual(df[df["labels"] == "Na_1"]["MS_shielding/ppm"].values[0], 433.819, places=3)
                # Cl_1 should have MS_shielding of 999.085
                self.assertAlmostEqual(df[df["labels"] == "Cl_1"]["MS_shielding/ppm"].values[0], 999.085, places=3)

                # check the multiplicity
                self.assertEqual(df["multiplicity"].nunique(), 1)
                self.assertEqual(df["multiplicity"].unique()[0], 4)

    def test_file_not_found(self):
        # # Can't find file
        runner = CliRunner()
        result = runner.invoke(soprano, ["nmr", "nothinghere"], prog_name="nmr")
        no_file_error = (
            "Error: Invalid value for 'FILES...': Path 'nothinghere' does not exist."
        )
        self.assertEqual(result.output.split("\n")[-2], no_file_error)
        self.assertEqual(result.exit_code, 2)

    # def test_invalid_option(self):
    #     # # Can't find file
    #     runner = CliRunner()
    #     result = runner.invoke(soprano, ['nmr', '-x', 'nothinghere'], prog_name='nmr')
    #     invalid_option_error = "Error: no such option: -x"
    #     self.assertEqual(result.output.split('\n')[-2], invalid_option_error)
    #     self.assertEqual(result.exit_code, 2)

    def test_molecules_zeolite(self):
        # Test that the splitmols command works on a zeolite with a single molecule
        # in the pore
        # - i.e. it should split into two atoms objects, one for the framework and
        # one for the molecule
        
        # First, ensure test_save directory exists and is writable
        self._TESTSAVE_DIR.mkdir(exist_ok=True)
        
        framework_file = self._TESTSAVE_DIR / "ZSM-5_withH2O_0.cif"
        molecule_file = self._TESTSAVE_DIR / "ZSM-5_withH2O_1.cif"
        try:
            # Run the actual test
            runner = CliRunner()
            with patch('click_log.basic_config'):
                # Check that source file exists
                fname = self._TESTDATA_DIR / "ZSM-5_withH2O.cif"
                self.assertTrue(fname.exists(), f"Test data file not found: {fname}")
                
                # Remove any existing output files from previous test runs
                framework_file.unlink(missing_ok=True)
                molecule_file.unlink(missing_ok=True)
                
                # this will generate two output files
                option_flags = [
                    "-o", 
                    str(self._TESTSAVE_DIR),  # Convert Path to string
                    "-f", 
                    "cif", 
                    "--vdw-scale", 
                    "1.3"
                ]
                
                # Run the CLI command
                result = runner.invoke(
                    soprano, ["splitmols", str(fname)] + option_flags, prog_name="splitmols"
                )
                
                # Check CLI command execution was successful
                self.assertEqual(result.exit_code, 0, f"CLI command failed with output: {result.output}")

                # there should be a warning about the lack of CH bonds:
                self.assertEqual(
                    result.output.split("\n")[0],
                    "warning: No C-H bonds found in the structure. Are you sure this is a molecular crystal?",
                )

                # Verify output files were created
                self.assertTrue(framework_file.exists(), f"Output file not created: {framework_file}")
                self.assertTrue(molecule_file.exists(), f"Output file not created: {molecule_file}")
                
                # Read in expected files
                try:
                    framework = read(framework_file)
                    molecule = read(molecule_file)
                except FileNotFoundError as e:
                    self.fail(f"Failed to read output file: {e}")
                
                # check the number of atoms in each
                self.assertEqual(len(framework), 288, "Framework structure has incorrect atom count")
                self.assertEqual(len(molecule), 3, "Molecule structure has incorrect atom count")
        finally:
            # Clean up: always remove the output files at the end of the test
            framework_file.unlink(missing_ok=True)
            molecule_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
