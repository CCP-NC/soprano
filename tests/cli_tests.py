#!/usr/bin/env python
"""
Test code for the Command Line Interface for Soprano
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
import unittest
import re
import numpy as np
import pandas as pd

from ase import io, Atoms
from click.testing import CliRunner
from soprano.scripts.cli import soprano, nmr
from tempfile import NamedTemporaryFile

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # noqa
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTDATA_DIR = os.path.join(_TEST_DIR, "test_data")
_TESTSAVE_DIR = os.path.join(_TEST_DIR, "test_save")


class TestCLI(unittest.TestCase):
    def test_read_valid_ms(self):
        # Load the data calculated with MagresView
        fname_data = os.path.join(_TESTDATA_DIR, "ethanol_ms.dat")

        ref_df = pd.read_csv(
            fname_data, sep="\s+|\t", lineterminator="\n", skiprows=7, engine="python"
        )
        nsites = len(ref_df)

        # get the same data use CLI
        runner = CliRunner()
        with NamedTemporaryFile() as temp_csv:
            fname_mag = os.path.join(_TESTDATA_DIR, "ethanol.magres")
            option_flags = [
                "-p",
                "ms",
                "--precision",
                "9",
                "-o",
                temp_csv.name,
                "--output-format",
                "csv",
                "-v",
            ]
            result = runner.invoke(
                soprano, ["nmr", fname_mag] + option_flags, prog_name="nmr"
            )
            # all went smoothly?
            self.assertEqual(result.exit_code, 0)

            # test to see that we parsed the correct file
            output = result.output.strip().split("\n")
            self.assertEqual(output[3], fname_mag)

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

            # np.testing.assert_almost_equal(ref_iso, cli_iso)

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
        runner = CliRunner()
        fname = os.path.join(_TESTDATA_DIR, "ZSM-5_withH2O.cif")
        # this will generate two output files, one for the framework and one for the molecule

        option_flags = ["-o", _TESTSAVE_DIR, "-f", "cif", "--vdw-scale", "1.3"]
        result = runner.invoke(
            soprano, ["splitmols", fname] + option_flags, prog_name="splitmols"
        )
        # all went smoothly?
        self.assertEqual(result.exit_code, 0)

        # there should be a warning about the lack of CH bonds:
        self.assertEqual(
            result.output.split("\n")[0],
            "warning: No C-H bonds found in the structure. Are you sure this is a molecular crystal?",
        )

        # read in expected files:
        framework = io.read(os.path.join(_TESTSAVE_DIR, "ZSM-5_withH2O_0.cif"))
        molecule = io.read(os.path.join(_TESTSAVE_DIR, "ZSM-5_withH2O_1.cif"))
        # check the number of atoms in each
        self.assertEqual(len(framework), 288)
        self.assertEqual(len(molecule), 3)

        # remove the files if they exist
        os.remove(os.path.join(_TESTSAVE_DIR, "ZSM-5_withH2O_0.cif"))
        os.remove(os.path.join(_TESTSAVE_DIR, "ZSM-5_withH2O_1.cif"))


if __name__ == "__main__":
    unittest.main()
