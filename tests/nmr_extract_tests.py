"""Tests for the NMR extraction helpers in soprano.nmr.extract."""

import os

import numpy as np
import pytest
import scipy.constants as cnst

from soprano.nmr.extract import nmr_extract_multi

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
ETHANOL = os.path.join(_TESTDATA_DIR, "ethanol.magres")

# Atomic unit of EFG in V/m^2 (Hartree / (e * a0^2))
AU_EFG_SI = cnst.physical_constants["atomic unit of electric field gradient"][0]


def _rows(df, species):
    return df[df["species"] == species]


def test_default_include_runs():
    dfs, images = nmr_extract_multi([ETHANOL])
    assert len(dfs) == 1
    assert len(images) == 1
    assert "MS_shielding/ppm" in dfs[0].columns
    assert "EFG_Vzz/Vm^-2" in dfs[0].columns


def test_efg_vzz_is_in_si_units():
    dfs, _ = nmr_extract_multi([ETHANOL], properties=["efg"])
    oxygen = _rows(dfs[0], "17O")
    assert len(oxygen) == 1
    # Vzz of the ethanol oxygen is about -1.86 au, so about -1.81e22 V/m^2
    vzz = oxygen["EFG_Vzz/Vm^-2"].iloc[0]
    assert -1.87 < vzz / AU_EFG_SI < -1.85
    assert vzz == pytest.approx(-1.81e22, rel=1e-2)


def test_partial_references_raise():
    with pytest.raises(ValueError, match="H, O"):
        nmr_extract_multi([ETHANOL], references={"C": 170.0})


def test_missing_gradients_use_full_formula():
    references = {"C": 170.0, "H": 30.0, "O": 100.0}
    dfs, _ = nmr_extract_multi(
        [ETHANOL], references=references, gradients={"C": -0.98}, properties=["ms"]
    )
    df = dfs[0]

    hydrogens = _rows(df, "1H")
    sigma_h = hydrogens["MS_shielding/ppm"].to_numpy()
    full_h = (30.0 - sigma_h) / (1 - 30.0 * 1e-6)
    np.testing.assert_allclose(hydrogens["MS_shift/ppm"].to_numpy(), full_h)

    carbons = _rows(df, "13C")
    sigma_c = carbons["MS_shielding/ppm"].to_numpy()
    np.testing.assert_allclose(carbons["MS_shift/ppm"].to_numpy(), 170.0 - 0.98 * sigma_c)


def test_unreadable_file_is_skipped():
    dfs, images = nmr_extract_multi(
        [os.path.join(_TESTDATA_DIR, "does_not_exist.magres"), ETHANOL]
    )
    assert len(dfs) == 1
    assert len(images) == 1


@pytest.mark.parametrize("seed", [0, 2, 3, 4])
def test_reduced_sites_come_from_one_molecule(seed):
    """Representatives form one molecule whatever the input atom order.

    Picking the first atom of each orbit independently put C2 and H4 on
    different molecules for these shuffles, turning the one-bond coupling
    into an intermolecular one.
    """
    from ase import io

    from soprano.nmr.extract import nmr_extract_atoms
    from soprano.properties.nmr import DipolarCoupling

    atoms = io.read(os.path.join(_TESTDATA_DIR, "alanine_manual_labels.magres"))
    shuffled = atoms[np.random.default_rng(seed).permutation(len(atoms))]
    reduced, index_map = nmr_extract_atoms(shuffled.copy(), reduce=True, return_index_map=True)

    labels = list(reduced.get_array("labels"))
    i, j = labels.index("C2"), labels.index("H4")
    assert reduced.get_distance(i, j, mic=True) == pytest.approx(1.0913, abs=1e-3)
    d = list(DipolarCoupling.get(reduced, sel_i=[i], sel_j=[j]).values())[0][0]
    assert d == pytest.approx(-23243.6, abs=1.0)

    # The index map still refers to the input order
    np.testing.assert_array_equal(
        reduced.get_array("labels")[index_map], shuffled.get_array("labels")
    )
