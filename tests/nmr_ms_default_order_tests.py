"""MSTensor defaults to Haeberlen order, as MSEuler, MSQuaternion and MagneticShielding do."""

from pathlib import Path

import numpy as np
from ase import io

from soprano.nmr import MagneticShielding
from soprano.properties.nmr.ms import MSEuler, MSTensor

_TESTDATA_DIR = Path(__file__).parent.resolve() / "test_data"


def _ethanol():
    return io.read(_TESTDATA_DIR / "ethanol.magres")


def test_default_order_is_haeberlen():
    tensors = MSTensor.get(_ethanol())
    assert all(t.order == MagneticShielding.ORDER_HAEBERLEN for t in tensors)


def test_default_matches_explicit_haeberlen():
    atoms = _ethanol()
    default = MSTensor.get(atoms)
    explicit = MSTensor.get(atoms, order="h")
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a.eigenvalues, b.eigenvalues)
        np.testing.assert_array_equal(a.eigenvectors, b.eigenvectors)


def test_increasing_order_still_available():
    tensors = MSTensor.get(_ethanol(), order="i")
    assert all(t.order == MagneticShielding.ORDER_INCREASING for t in tensors)
    for t in tensors:
        assert np.all(np.diff(t.eigenvalues) >= 0)


def test_msEuler_default_uses_haeberlen_tensors():
    atoms = _ethanol()
    eulers = MSEuler.get(atoms)
    expected = np.array([t.euler_angles("zyz", passive=False) for t in MSTensor.get(atoms, order="h")])
    np.testing.assert_allclose(eulers, expected)
