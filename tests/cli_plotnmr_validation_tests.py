"""CLI validation tests for plotnmr and the nmr command.

Invalid option combinations must give a click usage error (exit code 2) and
no traceback.
"""

import os

import pytest
from click.testing import CliRunner

from soprano.scripts.cli import soprano
from soprano.scripts.cli_utils import NMR_OPTIONS, mean_merge

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
ETHANOL = os.path.join(_TESTDATA_DIR, "ethanol.magres")
REFS_CH = ["--references", "C:170,H:30"]


def _invoke(args):
    return CliRunner().invoke(soprano, args, catch_exceptions=False)


def _assert_usage_error(result, message):
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert "Traceback" not in result.output


def test_b0_and_spectrometer_freq_are_exclusive():
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "C", "-y", "H",
         "--b0-field-tesla", "14.1", "--spectrometer-freq", "600"]
    )
    _assert_usage_error(result, "mutually exclusive")


def test_json_export_needs_larmor_frequency(tmp_path):
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "C", "-y", "H", *REFS_CH,
         "--export-file", str(tmp_path / "spec.json")]
    )
    _assert_usage_error(result, "needs a Larmor frequency")


def test_bruker_export_needs_larmor_frequency(tmp_path):
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "C", "-y", "H", *REFS_CH,
         "--export-file", str(tmp_path / "bruker"), "--export-format", "bruker"]
    )
    _assert_usage_error(result, "needs a Larmor frequency")


def test_unknown_export_extension(tmp_path):
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "C", "-y", "H", *REFS_CH,
         "--export-file", str(tmp_path / "spec.xyz")]
    )
    _assert_usage_error(result, "Cannot infer export format")


def test_cif_labels_rss_needs_cif_labels():
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "H", "-y", "H", "--yaxis-order", "2Q",
         "--weight-by", "dipolar_rss", "--rss-expand-j", "cif_labels",
         "--references", "H:30"]
    )
    _assert_usage_error(result, "needs a .magres file with CIF labels")


def test_2d_missing_reference_for_plotted_element():
    result = _invoke(["plotnmr", ETHANOL, "-x", "C", "-y", "H", "--references", "C:170"])
    _assert_usage_error(result, "No reference found for element(s) H")


def test_1d_missing_reference_for_plotted_element():
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "H", "--plot-type", "1D", "--references", "C:170"]
    )
    _assert_usage_error(result, "No reference found for element(s) H")


def test_export_options_without_export_file_warn(tmp_path):
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "C", "-y", "H", *REFS_CH, "--no-ticklabels",
         "--x-larmor-freq", "150", "-o", str(tmp_path / "plot.png")]
    )
    assert result.exit_code == 0, result.output
    assert "--x-larmor-freq has no effect without --export-file" in result.output


def test_rss_options_without_dipolar_rss_warn(tmp_path):
    result = _invoke(
        ["plotnmr", ETHANOL, "-x", "H", "-y", "H", "--yaxis-order", "2Q",
         "--rss-cutoff", "4.0", "--references", "H:30", "--no-ticklabels",
         "-o", str(tmp_path / "plot.png")]
    )
    assert result.exit_code == 0, result.output
    assert "--rss-cutoff only applies with --weight-by dipolar_rss" in result.output


def test_bruker_reexport_needs_overwrite(tmp_path):
    pytest.importorskip("nmrglue")
    args = ["plotnmr", ETHANOL, "-x", "C", "-y", "H", *REFS_CH, "--no-ticklabels",
            "--b0-field-tesla", "14.1", "--export-format", "bruker",
            "--export-file", str(tmp_path / "ds" / "1"), "-o", str(tmp_path / "p.png")]
    assert _invoke(args).exit_code == 0
    # Second run without --overwrite must refuse; with it, must succeed.
    _assert_usage_error(_invoke(args), "already exists")
    result = _invoke([*args, "--overwrite"])
    assert result.exit_code == 0, result.output


def test_nmr_missing_reference_is_usage_error():
    result = _invoke(["nmr", ETHANOL, "--references", "C:170,H:30"])
    _assert_usage_error(result, "No reference shielding given for element(s) O")


def test_nmr_mean_merge_option_works():
    assert mean_merge in NMR_OPTIONS
    result = _invoke(["nmr", "--mean-merge", ETHANOL])
    assert result.exit_code == 0, result.output
    assert "O_1" in result.output
