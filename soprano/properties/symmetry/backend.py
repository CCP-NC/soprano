# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
backend.py

Backend-agnostic symmetry analysis for ase.Atoms structures.

Two backends are supported: ``spglib`` (the original dependency) and ``moyo``
(moyopy, a faster Rust-based successor to spglib).  The active backend is
chosen with the ``backend`` keyword argument accepted by every public function:

* ``"auto"``   – use moyopy if installed, else spglib (default)
* ``"moyo"``   – always use moyopy (raises ImportError if not installed)
* ``"spglib"`` – always use spglib (raises ImportError if not installed)

The two public entry points are:

* :func:`get_symmetry_dataset` – returns a :class:`SpacegroupDataset`
* :func:`get_symmetry_ops_from_hall` – returns ``(rotations, translations)``
  numpy arrays for a given Hall number
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "BACKENDS",
    "SpacegroupDataset",
    "resolve_backend",
    "get_symmetry_dataset",
    "get_symmetry_ops_from_hall",
]

BACKENDS = ("auto", "spglib", "moyo")
BackendLiteral = Literal["auto", "spglib", "moyo"]


@dataclass
class SpacegroupDataset:
    """Backend-agnostic symmetry dataset for a periodic structure.

    All fields mirror the subset of the spglib symmetry dataset that is used
    inside Soprano.  The ``_raw`` attribute holds the native backend object
    (``spglib.SpglibDataset`` or ``moyopy.MoyoDataset``) for callers that need
    backend-specific fields.

    Parameters
    ----------
    international:
        Hermann-Mauguin space-group symbol (no spaces), e.g. ``"P-1"``.
    hall_number:
        Hall symbol number (1–530).
    std_lattice:
        Lattice vectors of the standardised conventional cell, shape ``(3, 3)``,
        each *row* is a lattice vector (same convention as spglib).
    transformation_matrix:
        Linear part of the transformation from the input cell to the
        standardised cell (spglib ``transformation_matrix`` /
        moyopy ``std_linear``), shape ``(3, 3)``.
    origin_shift:
        Origin shift of the transformation, shape ``(3,)``.
    rotations:
        Rotation matrices of all symmetry operations in the input cell,
        shape ``(N, 3, 3)``.
    translations:
        Translation vectors of all symmetry operations in the input cell,
        shape ``(N, 3)``.
    _raw:
        The raw dataset object returned by the backend.  Not included in
        ``repr``.
    """

    international: str
    hall_number: int
    std_lattice: np.ndarray
    transformation_matrix: np.ndarray
    origin_shift: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    equivalent_atoms: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    _raw: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def resolve_backend(backend: BackendLiteral) -> Literal["spglib", "moyo"]:
    """Resolve ``"auto"`` to the best available concrete backend.

    Parameters
    ----------
    backend:
        One of ``"auto"``, ``"spglib"``, ``"moyo"``.

    Returns
    -------
    str
        Either ``"spglib"`` or ``"moyo"``.

    Raises
    ------
    ValueError
        If *backend* is not one of the recognised values.
    ImportError
        If the requested backend (or any backend, when ``"auto"``) is not
        installed.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"backend must be one of {BACKENDS!r}, got {backend!r}"
        )
    if backend == "auto":
        try:
            import moyopy  # noqa: F401

            return "moyo"
        except ImportError:
            pass
        try:
            import spglib  # noqa: F401

            return "spglib"
        except ImportError:
            raise ImportError(
                "No symmetry backend is available.  Install moyopy with\n"
                "    pip install moyopy\n"
                "or spglib with\n"
                "    pip install 'spglib>=2.4'"
            )
    if backend == "moyo":
        try:
            import moyopy  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'moyo' backend requires moyopy to be installed:\n"
                "    pip install moyopy"
            )
        return "moyo"
    # backend == "spglib"
    try:
        import spglib  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'spglib' backend requires spglib to be installed:\n"
            "    pip install 'spglib>=2.4'"
        )
    return "spglib"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_symmetry_dataset(
    atoms,
    *,
    symprec: float = 1e-5,
    backend: BackendLiteral = "auto",
) -> SpacegroupDataset:
    """Return a :class:`SpacegroupDataset` for *atoms*.

    Parameters
    ----------
    atoms:
        An :class:`ase.Atoms` object representing a periodic structure.
    symprec:
        Distance tolerance in Ångströms for symmetry search.
    backend:
        Symmetry backend to use: ``"auto"`` (default), ``"spglib"``, or
        ``"moyo"``.

    Returns
    -------
    SpacegroupDataset
    """
    resolved = resolve_backend(backend)
    if resolved == "moyo":
        return _dataset_moyo(atoms, symprec)
    return _dataset_spglib(atoms, symprec)


def get_symmetry_ops_from_hall(
    hall_no: int,
    *,
    backend: BackendLiteral = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rotations, translations)`` for a given Hall number.

    The operations are those of the conventional cell for the space group
    identified by *hall_no*.

    Parameters
    ----------
    hall_no:
        Hall symbol number (1–530).
    backend:
        Symmetry backend to use: ``"auto"`` (default), ``"spglib"``, or
        ``"moyo"``.

    Returns
    -------
    rotations : np.ndarray, shape ``(N, 3, 3)``
    translations : np.ndarray, shape ``(N, 3)``
    """
    resolved = resolve_backend(backend)
    if resolved == "moyo":
        return _ops_moyo(hall_no)
    return _ops_spglib(hall_no)


# ---------------------------------------------------------------------------
# moyopy implementations
# ---------------------------------------------------------------------------


def _dataset_moyo(atoms, symprec: float) -> SpacegroupDataset:
    import moyopy

    # Build moyopy.Cell directly from ASE atoms to avoid the pymatgen dependency
    # that moyopy.interface.MoyoAdapter.from_atoms() introduces.
    cell = moyopy.Cell(
        atoms.get_cell().tolist(),
        atoms.get_scaled_positions().tolist(),
        atoms.get_atomic_numbers().tolist(),
    )
    ds = moyopy.MoyoDataset(cell, symprec=symprec)
    entry = moyopy.HallSymbolEntry(ds.hall_number)
    # moyopy hm_short uses spaces, e.g. "P -1"; strip them to match spglib
    international = entry.hm_short.replace(" ", "")
    # ds.orbits uses minimum-index representative (same convention as spglib);
    # convert to compact 0-based labels
    _, equiv = np.unique(np.array(ds.orbits), return_inverse=True)
    return SpacegroupDataset(
        international=international,
        hall_number=ds.hall_number,
        std_lattice=np.array(ds.std_cell.basis),
        transformation_matrix=np.array(ds.std_linear),
        origin_shift=np.array(ds.std_origin_shift),
        rotations=np.array(ds.operations.rotations, dtype=float),
        translations=np.array(ds.operations.translations, dtype=float),
        equivalent_atoms=equiv,
        _raw=ds,
    )


def _ops_moyo(hall_no: int) -> tuple[np.ndarray, np.ndarray]:
    import moyopy

    entry = moyopy.HallSymbolEntry(hall_no)
    ops = moyopy.operations_from_number(
        entry.number,
        setting=moyopy.Setting.hall_number(hall_no),
    )
    return (
        np.array(ops.rotations, dtype=float),
        np.array(ops.translations, dtype=float),
    )


# ---------------------------------------------------------------------------
# spglib implementations
# ---------------------------------------------------------------------------


def _dataset_spglib(atoms, symprec: float) -> SpacegroupDataset:
    import spglib
    from ase.utils import atoms_to_spglib_cell

    raw = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms), symprec=symprec)
    if raw is None:
        raise RuntimeError(
            "spglib failed to find symmetry for the given structure. "
            "Try increasing symprec."
        )
    # spglib equivalent_atoms maps each atom to the minimum-index representative;
    # convert to compact 0-based labels
    _, equiv = np.unique(np.array(raw.equivalent_atoms), return_inverse=True)
    return SpacegroupDataset(
        international=raw.international,
        hall_number=int(raw.hall_number),
        std_lattice=np.array(raw.std_lattice),
        transformation_matrix=np.array(raw.transformation_matrix),
        origin_shift=np.array(raw.origin_shift),
        rotations=np.array(raw.rotations, dtype=float),
        translations=np.array(raw.translations, dtype=float),
        equivalent_atoms=equiv,
        _raw=raw,
    )


def _ops_spglib(hall_no: int) -> tuple[np.ndarray, np.ndarray]:
    import spglib

    raw = spglib.get_symmetry_from_database(hall_no)
    return np.array(raw["rotations"], dtype=float), np.array(raw["translations"], dtype=float)
