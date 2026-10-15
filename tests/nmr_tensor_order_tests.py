"""
Warnings for relative Euler angles between tensors with different eigenvalue orders.

Two tensors of the same type with different orders give relative angles that depend on the
order, so ``euler_to`` (and ``equivalent_euler_to``, ``rotation_to``) warn. Mixing types with
different orders is deliberate (e.g. MS in Haeberlen and EFG in NQR order) and stays silent.
"""

import warnings

import numpy as np
import pytest

from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    return Q


def _triaxial(evals, seed, order, cls=NMRTensor, **kwargs):
    E = _random_rotation(seed)
    data = E @ np.diag(evals) @ E.T
    if cls is NMRTensor:
        return NMRTensor(data, order=order)
    return cls(data, order=order, **kwargs)


def _order_warnings(record):
    return [w for w in record if issubclass(w.category, UserWarning) and "ordering conventions" in str(w.message)]


def _rotation_from_zyz(a, b, g):
    def Rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def Ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    return Rz(a) @ Ry(b) @ Rz(g)


EVALS = [1.0, 2.0, 5.0]
EFG_EVALS = [-2.0, -1.0, 3.0]  # traceless, as for an EFG tensor


class TestSameTypeDifferentOrderWarns:

    def test_euler_to_warns_for_plain_tensors(self):
        A = _triaxial(EVALS, 1, "i")
        B = _triaxial(EVALS, 2, "h")
        with pytest.warns(UserWarning, match="different ordering conventions: i and h"):
            A.euler_to(B)

    def test_euler_to_warns_for_ms_tensors(self):
        A = _triaxial(EVALS, 3, "h", cls=MagneticShielding, species="1H")
        B = _triaxial(EVALS, 4, "i", cls=MagneticShielding, species="1H")
        with pytest.warns(UserWarning, match="different ordering conventions"):
            A.euler_to(B)

    def test_euler_to_warns_for_efg_tensors(self):
        A = _triaxial(EFG_EVALS, 5, "n", cls=ElectricFieldGradient, species="2H")
        B = _triaxial(EFG_EVALS, 6, "h", cls=ElectricFieldGradient, species="2H")
        with pytest.warns(UserWarning, match="different ordering conventions"):
            A.euler_to(B)

    def test_equivalent_euler_to_warns_once(self):
        A = _triaxial(EVALS, 7, "i")
        B = _triaxial(EVALS, 8, "h")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            A.equivalent_euler_to(B)
        assert len(_order_warnings(record)) == 1

    def test_euler_to_warns_once(self):
        A = _triaxial(EVALS, 9, "i")
        B = _triaxial(EVALS, 10, "d")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            A.euler_to(B)
        assert len(_order_warnings(record)) == 1

    def test_rotation_to_warns(self):
        A = _triaxial(EVALS, 11, "i")
        B = _triaxial(EVALS, 12, "h")
        with pytest.warns(UserWarning, match="different ordering conventions"):
            A.rotation_to(B)


class TestSameOrderSilent:

    def test_same_type_same_order_is_silent(self):
        A = _triaxial(EVALS, 1, "h")
        B = _triaxial(EVALS, 2, "h")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            A.euler_to(B)
            A.equivalent_euler_to(B)
            A.rotation_to(B)
        assert _order_warnings(record) == []


class TestDifferentTypeSilent:

    def test_ms_and_efg_with_different_orders_is_silent(self):
        ms = _triaxial(EVALS, 1, "h", cls=MagneticShielding, species="1H")
        efg = _triaxial(EFG_EVALS, 2, "n", cls=ElectricFieldGradient, species="2H")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ms.euler_to(efg)
            efg.euler_to(ms)
            ms.equivalent_euler_to(efg)
        assert _order_warnings(record) == []

    def test_plain_and_ms_with_different_orders_is_silent(self):
        plain = _triaxial(EVALS, 3, "i")
        ms = _triaxial(EVALS, 4, "h", cls=MagneticShielding, species="1H")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            plain.euler_to(ms)
        assert _order_warnings(record) == []


class TestValuesUnchanged:

    @pytest.mark.parametrize("order_a, order_b", [("i", "h"), ("h", "d"), ("i", "n")])
    def test_euler_to_reproduces_relative_orientation(self, order_a, order_b):
        # The active angles satisfy E_B = E_A R(alpha, beta, gamma) (see euler_to), so
        # rebuilding B from them must reproduce B's own data. This holds whatever the warning.
        A = _triaxial(EVALS, 21, order_a)
        B = _triaxial(EVALS, 22, order_b)
        with pytest.warns(UserWarning):
            a, b, g = A.euler_to(B, convention="zyz", passive=False)
        E_A = A.eigenvectors
        R = _rotation_from_zyz(a, b, g)
        rebuilt = E_A @ R @ np.diag(B.eigenvalues) @ R.T @ E_A.T
        np.testing.assert_allclose(rebuilt, B.data, atol=1e-10)
