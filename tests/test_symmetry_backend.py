#!/usr/bin/env python
"""
Tests for soprano.properties.symmetry.backend – parametrised over both the
spglib and moyo symmetry backends.
"""

import os
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BACKENDS = ["spglib", "moyo"]


def backend_available(backend: str) -> bool:
    try:
        from soprano.properties.symmetry.backend import resolve_backend

        resolve_backend(backend)
        return True
    except ImportError:
        return False


def skip_if_unavailable(backend: str):
    return pytest.mark.skipif(
        not backend_available(backend),
        reason=f"{backend} is not installed",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def triclinic_atoms():
    pos = np.zeros((6, 3))
    pos[0] = [0, 0.1, 0.2]
    pos[1] = [0, 0.3, 0.8]
    pos[2] = [0.4, 0.2, 0.6]
    pos[3:] = -pos[:3]
    return Atoms(["C"] * 6, positions=pos, cell=[5] * 3, pbc=[True] * 3)


@pytest.fixture
def silicon():
    return bulk("Si")


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------


def test_resolve_auto_returns_valid_backend():
    from soprano.optional import has_symmetry_backend
    from soprano.properties.symmetry.backend import BACKENDS, resolve_backend

    if not has_symmetry_backend():
        pytest.skip("No symmetry backend installed")

    result = resolve_backend("auto")
    assert result in BACKENDS and result != "auto"


def test_resolve_invalid_raises():
    from soprano.properties.symmetry.backend import resolve_backend

    with pytest.raises(ValueError, match="backend must be one of"):
        resolve_backend("unknown")


# ---------------------------------------------------------------------------
# get_symmetry_dataset – per backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_dataset_international(backend, triclinic_atoms):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry.backend import get_symmetry_dataset

    ds = get_symmetry_dataset(triclinic_atoms, backend=backend)
    # triclinic P-1 (no spaces either way)
    assert ds.international.replace(" ", "") == "P-1"


@pytest.mark.parametrize("backend", BACKENDS)
def test_dataset_hall_number_positive(backend, triclinic_atoms):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry.backend import get_symmetry_dataset

    ds = get_symmetry_dataset(triclinic_atoms, backend=backend)
    assert isinstance(ds.hall_number, int)
    assert 1 <= ds.hall_number <= 530


@pytest.mark.parametrize("backend", BACKENDS)
def test_dataset_shapes(backend, silicon):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry.backend import get_symmetry_dataset

    ds = get_symmetry_dataset(silicon, backend=backend)
    assert ds.std_lattice.shape == (3, 3)
    assert ds.transformation_matrix.shape == (3, 3)
    assert ds.origin_shift.shape == (3,)
    n_ops = ds.rotations.shape[0]
    assert ds.rotations.shape == (n_ops, 3, 3)
    assert ds.translations.shape == (n_ops, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dataset_silicon_spacegroup(backend, silicon):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry.backend import get_symmetry_dataset

    ds = get_symmetry_dataset(silicon, backend=backend)
    # Si has Fd-3m, international number 227, hall_number 523 or 525
    assert ds.international.replace(" ", "") in ("Fd-3m", "Fd3m", "Fd-3m:2", "F41/d-3m")
    assert ds.hall_number in range(1, 531)


# ---------------------------------------------------------------------------
# get_symmetry_ops_from_hall – per backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ops_from_hall_shapes(backend, silicon):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry.backend import (
        get_symmetry_dataset,
        get_symmetry_ops_from_hall,
    )

    ds = get_symmetry_dataset(silicon, backend=backend)
    rots, transls = get_symmetry_ops_from_hall(ds.hall_number, backend=backend)
    assert rots.ndim == 3 and rots.shape[1:] == (3, 3)
    assert transls.ndim == 2 and transls.shape[1] == 3
    assert rots.shape[0] == transls.shape[0]


# ---------------------------------------------------------------------------
# Cross-backend consistency check
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (backend_available("spglib") and backend_available("moyo")),
    reason="Both spglib and moyopy must be installed for cross-backend tests",
)
def test_both_backends_agree_on_spacegroup(silicon):
    from soprano.properties.symmetry.backend import get_symmetry_dataset

    ds_spg = get_symmetry_dataset(silicon, backend="spglib")
    ds_moyo = get_symmetry_dataset(silicon, backend="moyo")

    # Space-group number should agree (use hall_number as a shared identifier)
    # Hall numbers may differ by spglib/moyo setting choice, but the ITA
    # number encoded in it should match.
    from moyopy import HallSymbolEntry

    n_spg = HallSymbolEntry(ds_spg.hall_number).number
    n_moyo = HallSymbolEntry(ds_moyo.hall_number).number
    assert n_spg == n_moyo


# ---------------------------------------------------------------------------
# SymmetryDataset / WyckoffPoints property classes – backend param threading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_atoms_property_backend_param(backend, triclinic_atoms):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry import SymmetryDataset

    ds = SymmetryDataset.get(triclinic_atoms, backend=backend)
    assert ds.international.replace(" ", "") == "P-1"


@pytest.mark.parametrize("backend", BACKENDS)
def test_wyckoff_points_backend_param(backend, silicon):
    pytest.importorskip(backend if backend != "moyo" else "moyopy")
    from soprano.properties.symmetry import WyckoffPoints

    wpoints = WyckoffPoints.get(silicon, backend=backend)
    # Si has two Wyckoff positions in the primitive cell
    assert len(wpoints) >= 1
    for wp in wpoints:
        assert wp.fpos.shape == (3,)


# ---------------------------------------------------------------------------
# has_* helpers in optional
# ---------------------------------------------------------------------------


def test_optional_helpers():
    from soprano.optional import has_moyopy, has_spglib, has_symmetry_backend

    # Return types should always be bool
    assert isinstance(has_spglib(), bool)
    assert isinstance(has_moyopy(), bool)
    assert isinstance(has_symmetry_backend(), bool)
    # consistency: has_symmetry_backend() == has_spglib() or has_moyopy()
    assert has_symmetry_backend() == (has_spglib() or has_moyopy())
