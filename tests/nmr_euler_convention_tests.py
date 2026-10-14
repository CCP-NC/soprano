#!/usr/bin/env python
"""
Euler-angle convention tests for ``NMRTensor``.

The conventions are pinned to an *external* definition, the TensorView for MATLAB paper
(Svenningsson & Mueller, Solid State Nucl. Magn. Reson. 123 (2023) 101849), and not to the
code under test. Oracle policy:

* Every oracle rotation is built in the test code (here and in ``tests/nmr_test_helpers.py``),
  element by element, from the paper's explicit matrices
  (Eq. 65 active ZYZ, Eq. 75 active ZXZ, Eq. 9 passive = active with ``(-g, -b, -a)``). No
  Soprano helper and no ``scipy.spatial.transform.Rotation`` is used to build an expected value.
* ``TestOracle*`` validates the oracle itself (orthogonality, Eqs. 8-11, the half-turn
  equivalent sets, the paper's worked examples) so that a wrong oracle cannot silently make
  wrong code pass.
* Expected values are, where possible, literal numbers (hand-derived frames, the paper's Fig. 3
  and Fig. 5 numbers, Table 1) or invariants: the tensor is rebuilt from the returned angles
  with the oracle (Eq. 3, ``A = R diag(PAS) R^T``) and compared with the *input data*.
* The relative orientation is checked against ``R_rel = E_A^T E_B`` (Eq. 31) built from the
  tensors' eigenvectors, up to the 16 symmetries ``P R_rel Q`` (half turns about the PAS axes
  of A on the left and of B on the right) and, for axial tensors, up to the free rotation about
  the unique axis. Axial tensors are checked through the double coset invariant
  ``|R[axis_a, axis_b]| = cos(theta)`` where the unique axes are found from the tensor data
  with ``numpy.linalg.eigh``, independently of Soprano.
* The few cross-checks between two independent Soprano code paths say so in their docstring.
* ``TestAgainstTensorViewSoftware`` compares with stored output of TensorView's own MATLAB code
  (run in Octave, see ``tests/reference_generators/tensorview.py``). Soprano reports the same member
  of the equivalent sets as that code (``in_canonical_range``), so single-tensor angles agree exactly.

All random tests are seeded.
"""

import json
import unittest
import warnings
from pathlib import Path

import numpy as np

from soprano.data.nmr import nmr_gamma
from soprano.nmr.coupling import DipolarCoupling
from soprano.nmr.site import Site
from soprano.nmr.tensor import MagneticShielding, NMRTensor, TensorConvention
from tests.nmr_test_helpers import Rx, Ry, Rz, mrs_ct, random_rotation

PI = np.pi
TWO_PI = 2 * np.pi
HALF_PI = np.pi / 2
CONVENTIONS = ("zyz", "zxz")
COMBOS = [(c, p) for c in CONVENTIONS for p in (False, True)]


# --------------------------------------------------------------------------------------
# The oracle: explicit matrices from the paper (Rz, Ry, Rx are in tests/nmr_test_helpers.py)
# --------------------------------------------------------------------------------------

def R_active_zyz(a, b, g):
    """TensorView Eq. 65, written out element by element."""
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    return np.array([
        [ca * cb * cg - sa * sg, -sa * cg - ca * cb * sg, ca * sb],
        [sa * cb * cg + ca * sg, ca * cg - sa * cb * sg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ])


def R_active_zxz(a, b, g):
    """TensorView Eq. 75, written out element by element."""
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    return np.array([
        [ca * cg - sa * cb * sg, -sa * cb * cg - ca * sg, sa * sb],
        [ca * cb * sg + sa * cg, ca * cb * cg - sa * sg, -ca * sb],
        [sb * sg, sb * cg, cb],
    ])


def R_active(angles, convention):
    a, b, g = angles
    return (R_active_zyz if convention == "zyz" else R_active_zxz)(a, b, g)


def R_passive(angles, convention):
    """Eq. 9: ``R_passive(a, b, g) = R_active(-g, -b, -a)``."""
    a, b, g = angles
    return R_active((-g, -b, -a), convention)


def R_from(angles, convention, passive):
    return (R_passive if passive else R_active)(angles, convention)


# half turns about x, y, z: Rx(pi), Ry(pi), Rz(pi) written exactly
HALF_TURNS = (np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]))


def paper_single_sets(angles, passive):
    """The four orientation-equivalent sets of one tensor (TensorView, half turns about the PAS)."""
    a, b, g = angles
    if passive:
        rows = [[a, b, g], [TWO_PI - a, PI - b, g + PI], [PI - a, PI - b, g + PI], [a + PI, b, g]]
    else:
        rows = [[a, b, g], [a + PI, PI - b, TWO_PI - g], [a + PI, PI - b, PI - g], [a, b, g + PI]]
    return np.array(rows) % TWO_PI


def paper_active_zyz_16(angles):
    """The paper's explicit list O1..O16 of equivalent relative orientations (active ZYZ)."""
    a, b, g = angles
    rows = [
        [a, b, g], [a + PI, PI - b, TWO_PI - g], [a + PI, PI - b, PI - g], [a, b, g + PI],
        [TWO_PI - a, PI - b, g + PI], [PI - a, b, PI - g], [PI - a, b, TWO_PI - g],
        [TWO_PI - a, PI - b, g], [PI - a, PI - b, g + PI], [TWO_PI - a, b, PI - g],
        [TWO_PI - a, b, TWO_PI - g], [PI - a, PI - b, g], [a + PI, b, g],
        [a, PI - b, TWO_PI - g], [a, PI - b, PI - g], [a + PI, b, g + PI],
    ]
    return np.array(rows) % TWO_PI


# --------------------------------------------------------------------------------------
# Comparators
# --------------------------------------------------------------------------------------

def ang_diff(x, y):
    return (np.asarray(x, float) - np.asarray(y, float) + PI) % TWO_PI - PI


def same_angles(x, y, tol=1e-7):
    return bool(np.all(np.abs(ang_diff(x, y)) < tol))


def contains_row(rows, row, tol=1e-7):
    return any(same_angles(r, row, tol) for r in rows)


def row_sets_equal(rows1, rows2, tol=1e-7):
    return all(contains_row(rows2, r, tol) for r in rows1) and all(contains_row(rows1, r, tol) for r in rows2)


def n_distinct_rows(rows, tol=1e-6):
    kept = []
    for r in rows:
        if not contains_row(kept, r, tol):
            kept.append(r)
    return len(kept)


def close(A, B, tol=1e-9):
    return float(np.max(np.abs(np.asarray(A) - np.asarray(B)))) < tol


def matrix_sets_equal(Ms1, Ms2, tol=1e-9):
    return (all(any(close(a, b, tol) for b in Ms2) for a in Ms1)
            and all(any(close(a, b, tol) for b in Ms1) for a in Ms2))


def proper(E):
    E = np.array(E, dtype=float)
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1
    return E


def orbit16(R):
    """``{P R Q}``: half turns of the first tensor on the left, of the second on the right."""
    return [P @ R @ Q for P in HALF_TURNS for Q in HALF_TURNS]


def orbit4(R):
    """``{R Q}``: the PAS of a single tensor is unchanged by a half turn about its axes."""
    return [R @ Q for Q in HALF_TURNS]


def unique_index(evals, tol=1e-6):
    """Index (0, 1, 2 = x, y, z) of the eigenvalue that differs from the other two, else None."""
    e = np.asarray(evals, float)
    c01, c12, c02 = abs(e[0] - e[1]) < tol, abs(e[1] - e[2]) < tol, abs(e[0] - e[2]) < tol
    if c01 and not c12:
        return 2
    if c12 and not c01:
        return 0
    if c02 and not c01:
        return 1
    return None


def unique_vec(data):
    """Unit vector of the unique axis of an axial tensor, found independently with eigh."""
    w, v = np.linalg.eigh(np.asarray(data, float))
    return v[:, 2] if (w[1] - w[0]) < (w[2] - w[1]) else v[:, 0]


def theta_between(tA, tB):
    nA, nB = unique_vec(tA.data), unique_vec(tB.data)
    return float(np.arccos(min(1.0, abs(nA @ nB))))


def rel_orientation_matches(Rret, Rrel, ua=None, ub=None, tol=1e-8):
    """Does ``Rret`` describe the relative orientation ``Rrel = E_A^T E_B``?

    Up to the 16 half-turn symmetries and, for an axial tensor (unique axis ``ua`` of A / ``ub`` of
    B, ``None`` if triaxial), up to the free rotation about that axis: ``Rret = Ru X`` with ``Ru``
    a rotation about ``e_ua`` (A axial), ``Rret = X Rv`` (B axial) or both.
    """
    for P in HALF_TURNS:
        for Q in HALF_TURNS:
            X = P @ Rrel @ Q
            if ua is None and ub is None:
                ok = close(Rret, X, tol)
            elif ub is None:
                ok = close((Rret @ X.T)[:, ua], np.eye(3)[ua], tol)
            elif ua is None:
                ok = close((X.T @ Rret)[:, ub], np.eye(3)[ub], tol)
            else:
                ok = abs(abs(Rret[ua, ub]) - abs(X[ua, ub])) < tol
            if ok:
                return True
    return False


def b_in_a_invariants(M, u):
    """Quantities of the 3x3 ``M`` unchanged by a rotation about axis ``u``."""
    others = [i for i in range(3) if i != u]
    block = M[np.ix_(others, others)]
    return np.array([M[u, u], np.hypot(M[u, others[0]], M[u, others[1]]),
                     *np.sort(np.linalg.eigvalsh(block))])


# --------------------------------------------------------------------------------------
# Tensor builders (random_rotation is in tests/nmr_test_helpers.py)
# --------------------------------------------------------------------------------------

def spaced_evals(rng, gap=0.5):
    while True:
        e = rng.uniform(-6, 6, size=3)
        s = np.sort(e)
        if np.min(np.diff(s)) > gap:
            return e


def random_triaxial(rng, order="i"):
    E = random_rotation(rng)
    ev = spaced_evals(rng)
    return NMRTensor(E @ np.diag(ev) @ E.T, order=order)


def tensor_from_frame(evals, E, order="i"):
    """Tensor with exactly the PAS axes ``E`` (columns): ``E diag(evals) E^T``."""
    return NMRTensor((np.asarray(evals, float), np.array(E, float)), order=order)


def axial_tensor(n, perp, uniq, order):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return NMRTensor(perp * np.eye(3) + (uniq - perp) * np.outer(n, n), order=order)


def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


# (order, perpendicular eigenvalue, unique eigenvalue); which axis is unique in the PAS follows
# from the order and is asserted where the tensors are built.
AXIAL_SPECS = {
    "z": [("h", -1.0, 2.0), ("n", -1.0, 2.0), ("n", 1.0, -2.0), ("i", -1.0, 2.0),
          ("d", 2.0, -1.0), ("h", 1.5, -0.5), ("i", 0.3, 2.9)],
    "x": [("i", 2.0, -1.0), ("d", -1.0, 2.0), ("i", 1.0, -2.0), ("d", -0.5, 3.0)],
}


def make_tensor(kind, rng):
    """kind: 'tri', 'axz' (unique PAS z) or 'axx' (unique PAS x)."""
    if kind == "tri":
        return random_triaxial(rng, order=("i", "d", "h")[int(rng.integers(3))])
    specs = AXIAL_SPECS["z" if kind == "axz" else "x"]
    order, perp, uniq = specs[int(rng.integers(len(specs)))]
    scale = rng.uniform(0.5, 3.0)
    t = axial_tensor(random_unit(rng), scale * perp, scale * uniq, order)
    assert unique_index(t.eigenvalues) == (2 if kind == "axz" else 0), (kind, order, t.eigenvalues)
    return t


def frame_of(t):
    """The PAS axes of a tensor from its public ``eigenvectors``, checked against the tensor data."""
    E = t.eigenvectors
    assert close(E.T @ E, np.eye(3), 1e-9)
    assert close(E @ np.diag(t.eigenvalues) @ E.T, t.data, 1e-9), "eigenvectors do not diagonalise the data"
    return proper(E)


def in_canonical_range(angles, passive, tol=1e-9):
    """The member of the equivalent sets that TensorView's code reports (MFtoEuler.m, RelativeAngles.m).

    The paper's ranges are 0 <= a, g <= 2pi and 0 <= b <= pi (Eqs. 5-7). TensorView then takes
    b < pi/2 (b >= pi/2 -> Eq. 18 / 22) and g < pi for active angles (Eq. 19), a < pi for passive
    ones (Eq. 23). Soprano flips within 1e-6 of the boundaries (TensorView exactly at them), so beta
    can exceed pi/2 by up to 1e-6 and the free angle can be -1e-6.
    """
    a, b, g = angles
    free = a if passive else g
    return bool(np.all(np.isfinite(angles)) and -tol <= a < TWO_PI + tol and -tol <= g < TWO_PI + tol
                and -tol <= b <= HALF_PI + 1e-6 and -1e-6 <= free < PI + tol)


def zero_or_pi(angle, tol=1e-9):
    """The unique-x gauge (active alpha, passive gamma) after TensorView's range normalisation."""
    return min(abs(float(ang_diff(angle, 0.0))), abs(float(ang_diff(angle, PI)))) < tol


class QuietTestCase(unittest.TestCase):
    """Silence warnings (scipy gimbal lock, NQR order, 'perfectly aligned') unless asserted."""

    def setUp(self):
        manager = warnings.catch_warnings()
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        warnings.simplefilter("ignore")

    def assertRebuilds(self, t, angles, convention, passive, msg="", tol=1e-9):
        R = R_from(angles, convention, passive)
        rebuilt = R @ np.diag(t.eigenvalues) @ R.T
        err = float(np.max(np.abs(rebuilt - t.data)))
        self.assertLess(err, tol, f"{msg} {convention} passive={passive} angles={angles}: rebuild error {err}")


# --------------------------------------------------------------------------------------
# A. The oracle validates itself
# --------------------------------------------------------------------------------------

class TestOracleDefinitions(QuietTestCase):

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(1)
        self.angle_sets = rng.uniform(-TWO_PI, TWO_PI, size=(40, 3))

    def test_matrices_are_proper_rotations(self):
        for angles in self.angle_sets:
            for conv in CONVENTIONS:
                for passive in (False, True):
                    R = R_from(angles, conv, passive)
                    with self.subTest(angles=angles, conv=conv, passive=passive):
                        self.assertTrue(close(R @ R.T, np.eye(3), 1e-12))
                        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_explicit_matrices_equal_products_of_elementary_rotations(self):
        # Eq. 65 = Rz(a) Ry(b) Rz(g); Eq. 75 = Rz(a) Rx(b) Rz(g)
        for a, b, g in self.angle_sets:
            with self.subTest(angles=(a, b, g)):
                self.assertTrue(close(R_active_zyz(a, b, g), Rz(a) @ Ry(b) @ Rz(g), 1e-12))
                self.assertTrue(close(R_active_zxz(a, b, g), Rz(a) @ Rx(b) @ Rz(g), 1e-12))

    def test_elementary_rotations_are_right_handed_active(self):
        # Rz(90) takes x to y, Rx(90) takes y to z, Ry(90) takes z to x
        self.assertTrue(close(Rz(HALF_PI) @ [1, 0, 0], [0, 1, 0], 1e-12))
        self.assertTrue(close(Rx(HALF_PI) @ [0, 1, 0], [0, 0, 1], 1e-12))
        self.assertTrue(close(Ry(HALF_PI) @ [0, 0, 1], [1, 0, 0], 1e-12))

    def test_half_turn_matrices_are_rx_ry_rz_of_pi(self):
        for P, R in zip(HALF_TURNS[1:], (Rx(PI), Ry(PI), Rz(PI))):
            self.assertTrue(close(P, R, 1e-12))

    def test_eq9_passive_active_relation_both_forms(self):
        for a, b, g in self.angle_sets:
            for conv in CONVENTIONS:
                with self.subTest(angles=(a, b, g), conv=conv):
                    Ra = R_active((a, b, g), conv)
                    # R_active(a,b,g) = R_passive(-g,-b,-a)
                    self.assertTrue(close(Ra, R_passive((-g, -b, -a), conv), 1e-12))
                    # R_active(a,b,g) = R_passive(pi-g, b, pi-a)
                    self.assertTrue(close(Ra, R_passive((PI - g, b, PI - a), conv), 1e-12))
                    # and, equivalently, R_passive(a,b,g) = R_active(-g,-b,-a), product form
                    prod = Rz(-g) @ (Ry(-b) if conv == "zyz" else Rx(-b)) @ Rz(-a)
                    self.assertTrue(close(R_passive((a, b, g), conv), prod, 1e-12))

    def test_eq8_double_cover(self):
        # R(a,b,g) = R(a+pi, 2pi-b, g+pi), all four of {active, passive} x {ZYZ, ZXZ}
        for a, b, g in self.angle_sets:
            for conv in CONVENTIONS:
                for passive in (False, True):
                    with self.subTest(angles=(a, b, g), conv=conv, passive=passive):
                        self.assertTrue(close(R_from((a, b, g), conv, passive),
                                              R_from((a + PI, TWO_PI - b, g + PI), conv, passive), 1e-12))

    def test_eq10_eq11_zyz_zxz_conversion(self):
        for a, b, g in self.angle_sets:
            for passive in (False, True):
                with self.subTest(angles=(a, b, g), passive=passive):
                    # Eq. 10: R_ZYZ(a,b,g) = R_ZXZ(a+pi/2, b, g-pi/2)
                    self.assertTrue(close(R_from((a, b, g), "zyz", passive),
                                          R_from((a + HALF_PI, b, g - HALF_PI), "zxz", passive), 1e-12))
                    # Eq. 11: R_ZXZ(a,b,g) = R_ZYZ(a-pi/2, b, g+pi/2)
                    self.assertTrue(close(R_from((a, b, g), "zxz", passive),
                                          R_from((a - HALF_PI, b, g + HALF_PI), "zyz", passive), 1e-12))

    def test_paper_single_sets_are_half_turns_of_the_pas(self):
        for a, b, g in self.angle_sets:
            for conv in CONVENTIONS:
                for passive in (False, True):
                    with self.subTest(angles=(a, b, g), conv=conv, passive=passive):
                        rows = paper_single_sets((a, b, g), passive)
                        mats = [R_from(r, conv, passive) for r in rows]
                        self.assertTrue(matrix_sets_equal(mats, orbit4(mats[0]), 1e-9))
                        self.assertEqual(len({tuple(np.round(m, 6).ravel()) for m in mats}), 4)

    def test_active_set_assignment_matches_the_paper(self):
        # R(O2) = R(O1) Rx(pi), R(O3) = R(O1) Ry(pi), R(O4) = R(O1) Rz(pi) for ZYZ; O2 and O3 swap for ZXZ
        for a, b, g in self.angle_sets:
            rows = paper_single_sets((a, b, g), False)
            with self.subTest(angles=(a, b, g)):
                m = [R_active_zyz(*r) for r in rows]
                self.assertTrue(close(m[1], m[0] @ Rx(PI), 1e-12))
                self.assertTrue(close(m[2], m[0] @ Ry(PI), 1e-12))
                self.assertTrue(close(m[3], m[0] @ Rz(PI), 1e-12))
                m = [R_active_zxz(*r) for r in rows]
                self.assertTrue(close(m[1], m[0] @ Ry(PI), 1e-12))
                self.assertTrue(close(m[2], m[0] @ Rx(PI), 1e-12))
                self.assertTrue(close(m[3], m[0] @ Rz(PI), 1e-12))

    def test_active_zyz_16_list_is_the_p_r_q_orbit(self):
        for a, b, g in self.angle_sets:
            rows = paper_active_zyz_16((a, b, g))
            mats = [R_active_zyz(*r) for r in rows]
            with self.subTest(angles=(a, b, g)):
                self.assertTrue(matrix_sets_equal(mats, orbit16(mats[0]), 1e-9))
                self.assertEqual(n_distinct_rows(rows), 16)

    def test_hand_derived_frames(self):
        # Rx(90) = Rz(-90) Ry(90) Rz(90): third column (0,-1,0) = (ca sb, sa sb, cb) gives b=90, a=-90;
        # third row (0,1,0) = (-sb cg, sb sg, cb) gives g=90.
        Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
        self.assertTrue(close(R_active_zyz(np.radians(270), HALF_PI, HALF_PI), Rx90, 1e-12))
        # Ry(90) = ZYZ(0, 90, 0) = ZXZ(90, 90, -90) (Eq. 10)
        Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
        self.assertTrue(close(R_active_zyz(0, HALF_PI, 0), Ry90, 1e-12))
        self.assertTrue(close(R_active_zxz(HALF_PI, HALF_PI, -HALF_PI), Ry90, 1e-12))
        # Rx(90) = ZXZ(0, 90, 0)
        self.assertTrue(close(R_active_zxz(0, HALF_PI, 0), Rx90, 1e-12))
        # passive: R_passive(a,b,g) = R_active(-g,-b,-a); passive ZYZ [90, 90, 270] equals Rx(90)
        self.assertTrue(close(R_passive(np.radians([90, 90, 270]), "zyz"), Rx90, 1e-12))
        # passive ZYZ [180, 90, 180] equals Ry(90)
        self.assertTrue(close(R_passive(np.radians([180, 90, 180]), "zyz"), Ry90, 1e-12))
        # passive ZXZ [270, 90, 90] equals Ry(90)
        self.assertTrue(close(R_passive(np.radians([270, 90, 90]), "zxz"), Ry90, 1e-12))

    def test_fig5_both_reported_sets_are_one_orientation(self):
        # Fig. 5: [4.1, 89.7, 154.7] and [175.9, 89.7, 205.1] deg (active ZYZ). The second is O7 =
        # [pi-a, b, 2pi-g] = [175.9, 89.7, 205.3] of the first. 205.1 is not a rounding of 205.3 (154.7 is
        # 154.65-154.75, so 2pi - g is 205.25-205.35): the paper's two numbers disagree by 0.2 deg, presumably
        # because the programs were given slightly different tensors. Hence the 0.25 deg tolerance below.
        first = np.radians([4.1, 89.7, 154.7])
        second = np.radians([175.9, 89.7, 205.1])
        o7 = paper_active_zyz_16(first)[6]
        self.assertTrue(same_angles(o7, np.radians([175.9, 89.7, 205.3]), 1e-9))
        self.assertTrue(same_angles(o7, second, np.radians(0.25)))
        orbit = orbit16(R_active_zyz(*first))
        self.assertTrue(any(close(R_active_zyz(*o7), M, 1e-12) for M in orbit))
        self.assertTrue(any(close(R_active_zyz(*second), M, 6e-3) for M in orbit))


# --------------------------------------------------------------------------------------
# B. Single tensor: the definition
# --------------------------------------------------------------------------------------

class TestSingleTensorDefinition(QuietTestCase):

    def test_random_triaxial_tensors_are_rebuilt_by_eq3(self):
        rng = np.random.default_rng(2024)
        for i in range(300):
            order = ("i", "d", "h", "n")[i % 4]
            t = random_triaxial(rng, order=order)
            for conv, passive in COMBOS:
                angles = t.euler_angles(conv, passive)
                with self.subTest(case=i, order=order, conv=conv, passive=passive):
                    self.assertTrue(in_canonical_range(angles, passive), f"case {i}: {angles}")
                    self.assertRebuilds(t, angles, conv, passive, msg=f"case {i} order {order}")

    def test_degrees_flag_is_radians_times_180_over_pi(self):
        rng = np.random.default_rng(5)
        for i in range(30):
            t = random_triaxial(rng)
            for conv, passive in COMBOS:
                rad = t.euler_angles(conv, passive)
                deg = t.euler_angles(conv, passive, degrees=True)
                np.testing.assert_allclose(deg, rad * 180.0 / PI, rtol=1e-12, atol=1e-12)

    def test_calls_are_repeatable_and_do_not_mutate_the_tensor(self):
        rng = np.random.default_rng(6)
        for i in range(60):
            t = random_triaxial(rng, order=("i", "h")[i % 2])
            evecs, evals, data = t.eigenvectors, t.eigenvalues, t.data
            for conv, passive in COMBOS:
                first = t.euler_angles(conv, passive)
                second = t.euler_angles(conv, passive)
                with self.subTest(case=i, conv=conv, passive=passive):
                    np.testing.assert_array_equal(first, second)
                    np.testing.assert_array_equal(t.eigenvectors, evecs)
                    np.testing.assert_array_equal(t.eigenvalues, evals)
                    np.testing.assert_array_equal(t.data, data)

    def test_order_is_accepted_as_enum_or_string_with_identical_angles(self):
        rng = np.random.default_rng(7)
        E, ev = random_rotation(rng), np.array([-3.0, 0.5, 4.0])
        data = E @ np.diag(ev) @ E.T
        for letter, enum in (("i", TensorConvention.Increasing), ("d", TensorConvention.Decreasing),
                             ("h", TensorConvention.Haeberlen), ("n", TensorConvention.NQR)):
            for conv, passive in COMBOS:
                np.testing.assert_allclose(NMRTensor(data, order=letter).euler_angles(conv, passive),
                                           NMRTensor(data, order=enum).euler_angles(conv, passive), atol=1e-12)

    def test_improper_and_permuted_eigenvector_input(self):
        # The same tensor given with improper frames and with permuted / negated columns
        rng = np.random.default_rng(8)
        for i in range(60):
            E = random_rotation(rng)
            ev = spaced_evals(rng)
            data = E @ np.diag(ev) @ E.T
            perm = rng.permutation(3)
            signs = rng.choice([-1.0, 1.0], size=3)
            E2 = E[:, perm] * signs
            variants = {"improper": (ev, E * [1, 1, -1]), "permuted/negated": (ev[perm], E2)}
            for name, (evals, frame) in variants.items():
                if name == "permuted/negated":
                    self.assertTrue(close(frame @ np.diag(evals) @ frame.T, data, 1e-9))
                for order in ("i", "h"):
                    t = NMRTensor((evals, frame), order=order)
                    self.assertTrue(close(t.data, data, 1e-9), name)
                    for conv, passive in COMBOS:
                        angles = t.euler_angles(conv, passive)
                        with self.subTest(case=i, kind=name, order=order, conv=conv, passive=passive):
                            self.assertTrue(in_canonical_range(angles, passive))
                            self.assertRebuilds(t, angles, conv, passive, msg=f"case {i} {name}")

    def test_identity_frame_gives_zero_angles(self):
        # diag(1,2,3) in increasing order has E = identity: every angle is zero
        t = NMRTensor(np.diag([1.0, 2.0, 3.0]), order="i")
        for conv, passive in COMBOS:
            np.testing.assert_allclose(ang_diff(t.euler_angles(conv, passive), 0.0), 0.0, atol=1e-12)

    def test_hand_derived_frames_with_explicit_angles(self):
        # Each frame is given as literal active/passive angles built with the oracle; Soprano's angles
        # must lie among the four equivalent sets of the literal ones (beta < pi/2 keeps two of them).
        cases = [(np.radians([30.0, 60.0, 45.0]),),
                 (np.radians([200.0, 130.0, 300.0]),),
                 (np.radians([10.0, 20.0, 330.0]),),
                 (np.radians([359.0, 89.0, 181.0]),)]
        for (literal,) in cases:
            sets = {False: paper_single_sets(literal, False), True: paper_single_sets(literal, True)}
            for conv, passive in COMBOS:
                E = R_from(literal, conv, passive)
                t = tensor_from_frame([1.0, 2.0, 3.0], E, order="i")
                angles = t.euler_angles(conv, passive)
                with self.subTest(literal=np.degrees(literal), conv=conv, passive=passive):
                    self.assertTrue(contains_row(sets[passive], angles),
                                    f"{np.degrees(angles)} not in {np.degrees(sets[passive])}")
                    self.assertLess(angles[1], HALF_PI + 1e-6)
                    if literal[1] < HALF_PI:
                        # an orientation that already has beta < pi/2 comes back as itself or
                        # as the gamma/alpha + pi partner
                        partner = sets[passive][3]
                        self.assertTrue(same_angles(angles, literal) or same_angles(angles, partner))

    def test_90_degree_frames_have_literal_angles(self):
        # Derived by hand (see TestOracleDefinitions.test_hand_derived_frames): all angles are in {0,90,180,270}
        Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
        Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
        deg = np.radians
        # (frame, convention, passive) -> every equivalent set of the literal angle triple
        literal = {
            ("Ry90", "zyz", False): [[0, 90, 0], [180, 90, 0], [180, 90, 180], [0, 90, 180]],
            ("Ry90", "zyz", True): [[180, 90, 180], [180, 90, 0], [0, 90, 0], [0, 90, 180]],
            ("Rx90", "zyz", False): [[270, 90, 90], [90, 90, 270], [90, 90, 90], [270, 90, 270]],
            ("Rx90", "zyz", True): [[90, 90, 270], [270, 90, 90], [90, 90, 90], [270, 90, 270]],
            ("Ry90", "zxz", False): [[90, 90, 270], [270, 90, 90], [270, 90, 270], [90, 90, 90]],
            ("Ry90", "zxz", True): [[270, 90, 90], [90, 90, 270], [270, 90, 270], [90, 90, 90]],
            ("Rx90", "zxz", False): [[0, 90, 0], [180, 90, 0], [180, 90, 180], [0, 90, 180]],
            ("Rx90", "zxz", True): [[180, 90, 180], [180, 90, 0], [0, 90, 0], [0, 90, 180]],
        }
        frames = {"Rx90": Rx90, "Ry90": Ry90}
        for (name, conv, passive), rows in literal.items():
            t = tensor_from_frame([1.0, 2.0, 3.0], frames[name])
            angles = t.euler_angles(conv, passive)
            # the literal O1 reproduces the frame (oracle check of the hand derivation)
            self.assertTrue(close(R_from(deg(rows[0]), conv, passive), frames[name], 1e-12), (name, conv, passive))
            with self.subTest(frame=name, conv=conv, passive=passive):
                self.assertTrue(contains_row(deg(rows), angles), f"{np.degrees(angles)} not in {rows}")
                self.assertRebuilds(t, angles, conv, passive)

    def test_paper_fig3_csa_tensor(self):
        # Fig. 3: delta = (215, 151, 42) ppm, PAS diag(215, 151, 42) in Haeberlen order, oriented in the
        # molecular frame by passive ZYZ angles (38, 84, 180) degrees.
        literal = np.radians([38.0, 84.0, 180.0])
        E = R_passive(literal, "zyz")
        A_mf = E @ np.diag([215.0, 151.0, 42.0]) @ E.T
        t = NMRTensor(A_mf, order="h")
        np.testing.assert_allclose(t.eigenvalues, [215.0, 151.0, 42.0], atol=1e-9)
        angles = t.euler_angles("zyz", passive=True)
        self.assertTrue(contains_row(paper_single_sets(literal, True), angles),
                        f"{np.degrees(angles)} is not an equivalent set of [38, 84, 180]")
        # beta < pi/2 and alpha < pi single out the paper's own numbers
        np.testing.assert_allclose(np.degrees(angles), [38.0, 84.0, 180.0], atol=1e-6)
        self.assertRebuilds(t, angles, "zyz", True)
        # the active angles of the same tensor describe the same A_mf
        active = t.euler_angles("zyz", passive=False)
        self.assertRebuilds(t, active, "zyz", False)
        R = R_active_zyz(*active)
        self.assertTrue(close(R @ np.diag([215.0, 151.0, 42.0]) @ R.T, A_mf, 1e-9))
        # and so do the ZXZ ones
        for passive in (False, True):
            self.assertRebuilds(t, t.euler_angles("zxz", passive), "zxz", passive)


# --------------------------------------------------------------------------------------
# C. Single tensor: equivalent sets
# --------------------------------------------------------------------------------------

class TestSingleTensorEquivalentSets(QuietTestCase):

    def test_four_distinct_sets_each_reproduce_the_tensor(self):
        rng = np.random.default_rng(31)
        for i in range(60):
            t = random_triaxial(rng, order=("i", "d", "h")[i % 3])
            for conv, passive in COMBOS:
                rows = t.equivalent_euler_angles(conv, passive)
                with self.subTest(case=i, conv=conv, passive=passive):
                    self.assertEqual(rows.shape, (4, 3))
                    self.assertEqual(n_distinct_rows(rows), 4)
                    for row in rows:
                        self.assertRebuilds(t, row, conv, passive, msg=f"case {i}")
                    self.assertTrue(contains_row(rows, t.euler_angles(conv, passive)))

    def test_sets_are_the_papers_explicit_formulas_and_the_matrix_orbit(self):
        rng = np.random.default_rng(32)
        for i in range(60):
            t = random_triaxial(rng, order=("i", "h")[i % 2])
            for conv, passive in COMBOS:
                rows = t.equivalent_euler_angles(conv, passive)
                o1 = t.euler_angles(conv, passive)
                with self.subTest(case=i, conv=conv, passive=passive):
                    # paper's formulas; for ZXZ the same four solutions up to the O2/O3 swap
                    self.assertTrue(row_sets_equal(rows, paper_single_sets(o1, passive)),
                                    f"{np.degrees(rows)} vs {np.degrees(paper_single_sets(o1, passive))}")
                    # matrix generated: R(O1) {1, Rx(pi), Ry(pi), Rz(pi)}
                    mats = [R_from(r, conv, passive) for r in rows]
                    self.assertTrue(matrix_sets_equal(mats, orbit4(R_from(o1, conv, passive))))

    def test_degenerate_tensors_still_return_four_equivalent_sets(self):
        rng = np.random.default_rng(33)
        for kind in ("axz", "axx"):
            for i in range(15):
                t = make_tensor(kind, rng)
                for conv, passive in COMBOS:
                    rows = t.equivalent_euler_angles(conv, passive)
                    with self.subTest(kind=kind, case=i, conv=conv, passive=passive):
                        self.assertEqual(rows.shape, (4, 3))
                        for row in rows:
                            self.assertRebuilds(t, row, conv, passive, msg=f"{kind} {i}")


# --------------------------------------------------------------------------------------
# D. ZYZ <-> ZXZ (Eqs. 10, 11)
# --------------------------------------------------------------------------------------

class TestZyzZxzMapping(QuietTestCase):

    def test_zxz_and_zyz_rebuild_the_same_tensor_through_eq10(self):
        rng = np.random.default_rng(41)
        for i in range(120):
            t = random_triaxial(rng, order=("i", "d", "h")[i % 3])
            for passive in (False, True):
                zyz = t.euler_angles("zyz", passive)
                zxz = t.euler_angles("zxz", passive)
                with self.subTest(case=i, passive=passive):
                    # Eq. 10 applied to Soprano's ZYZ angles describes the tensor in ZXZ ...
                    mapped = zyz + np.array([HALF_PI, 0.0, -HALF_PI])
                    self.assertRebuilds(t, mapped, "zxz", passive, msg=f"case {i}")
                    # ... and Eq. 11 applied to the ZXZ angles describes it in ZYZ
                    back = zxz + np.array([-HALF_PI, 0.0, HALF_PI])
                    self.assertRebuilds(t, back, "zyz", passive, msg=f"case {i}")

    def test_canonical_zxz_angles_are_the_eq10_shift_of_the_zyz_ones(self):
        # beta is untouched by Eq. 10; the other two shift by +-pi/2 up to the half turn about the third axis
        # (gamma + pi active, alpha + pi passive) that the canonical range choice may add.
        rng = np.random.default_rng(42)
        for i in range(120):
            t = random_triaxial(rng, order=("i", "h")[i % 2])
            for passive in (False, True):
                zyz = t.euler_angles("zyz", passive)
                zxz = t.euler_angles("zxz", passive)
                with self.subTest(case=i, passive=passive):
                    self.assertAlmostEqual(zyz[1], zxz[1], places=9)
                    mapped = zyz + np.array([HALF_PI, 0.0, -HALF_PI])
                    self.assertTrue(same_angles(zxz, mapped, 1e-8)
                                    or same_angles(zxz, mapped + [PI if passive else 0.0, 0.0, 0.0 if passive else PI], 1e-8),
                                    f"case {i}: zyz {np.degrees(zyz)} zxz {np.degrees(zxz)}")

    def test_axial_tensors_map_too(self):
        rng = np.random.default_rng(43)
        for kind in ("axz", "axx"):
            for i in range(20):
                t = make_tensor(kind, rng)
                for passive in (False, True):
                    zyz = t.euler_angles("zyz", passive)
                    with self.subTest(kind=kind, case=i, passive=passive):
                        self.assertRebuilds(t, zyz + [HALF_PI, 0.0, -HALF_PI], "zxz", passive, tol=1e-9)
                        zxz = t.euler_angles("zxz", passive)
                        self.assertRebuilds(t, zxz + [-HALF_PI, 0.0, HALF_PI], "zyz", passive, tol=1e-9)


# --------------------------------------------------------------------------------------
# E. Gimbal lock and special orientations
# --------------------------------------------------------------------------------------

def special_frames():
    """(name, frame) pairs: proper rotations at the orientations where angle extraction is delicate."""
    frames = []
    d = np.radians
    for a in (0.0, 90.0, 180.0, 270.0, 33.0):
        for g in (0.0, 90.0, 180.0, 77.0):
            for b in (0.0, 90.0, 180.0, 45.0, 135.0, 1e-9, 180.0 - 1e-9, 90.0 - 1e-7, 90.0 + 1e-7):
                frames.append((f"zyz({a},{b},{g})", R_active_zyz(d(a), d(b), d(g))))
    frames += [(f"zxz({a},{b},{g})", R_active_zxz(d(a), d(b), d(g)))
               for a in (0.0, 90.0, 180.0) for b in (0.0, 90.0, 180.0) for g in (0.0, 90.0, 180.0)]
    frames += [("identity", np.eye(3)),
               ("cyclic x->y->z->x", np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])),
               ("cyclic z->y->x", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])),
               ("x,z,-y", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])),
               ("-x,-y,z", np.diag([-1.0, -1.0, 1.0])), ("x,-y,-z", np.diag([1.0, -1.0, -1.0])),
               ("-x,y,-z", np.diag([-1.0, 1.0, -1.0]))]
    for name, rot in (("x", Rx), ("y", Ry), ("z", Rz)):
        for ang in (45.0, 90.0, 180.0, 270.0):
            frames.append((f"R{name}({ang})", rot(d(ang))))
    # a 45 degree tilt about an in-plane diagonal axis
    n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    frames.append(("45 deg about (1,1,0)", np.eye(3) + np.sin(PI / 4) * K + (1 - np.cos(PI / 4)) * K @ K))
    return frames


class TestSpecialOrientations(QuietTestCase):

    def test_triaxial_special_frames_all_conventions(self):
        for name, E in special_frames():
            t = tensor_from_frame([1.0, 2.0, 3.0], E, order="i")
            self.assertTrue(close(t.data, E @ np.diag([1.0, 2.0, 3.0]) @ E.T, 1e-12))
            for conv, passive in COMBOS:
                angles = t.euler_angles(conv, passive)
                with self.subTest(frame=name, conv=conv, passive=passive):
                    self.assertTrue(in_canonical_range(angles, passive), f"{name}: {np.degrees(angles)}")
                    self.assertRebuilds(t, angles, conv, passive, msg=name, tol=1e-8)

    def test_special_frames_equivalent_sets_reproduce_the_tensor(self):
        for name, E in special_frames()[::7]:
            t = tensor_from_frame([1.0, 2.0, 3.0], E, order="i")
            for conv, passive in COMBOS:
                for row in t.equivalent_euler_angles(conv, passive):
                    with self.subTest(frame=name, conv=conv, passive=passive):
                        self.assertRebuilds(t, row, conv, passive, msg=name, tol=1e-8)

    def test_beta_pi_frame_is_reported_with_the_equivalent_small_beta(self):
        # Ra(a, pi, g) is a frame with beta = pi; the tensor is the same as for an equivalent beta < pi/2 set
        # (here beta -> 0) so the reported beta must be < pi/2.
        for a in (0.0, 0.7, 2.0):
            for g in (0.0, 1.1, 3.0):
                t = tensor_from_frame([1.0, 2.0, 3.0], R_active_zyz(a, PI, g))
                for conv, passive in COMBOS:
                    with self.subTest(a=a, g=g, conv=conv, passive=passive):
                        angles = t.euler_angles(conv, passive)
                        self.assertLess(angles[1], HALF_PI + 1e-6)
                        self.assertRebuilds(t, angles, conv, passive, tol=1e-8)

    def test_special_frames_with_axial_tensors(self):
        for name, E in special_frames()[::3]:
            for evals, order, expected_unique in (([-1.0, -1.0, 2.0], "h", 2), ([-1.0, 2.0, 2.0], "i", 0)):
                t = tensor_from_frame(evals, E, order=order)
                for conv, passive in COMBOS:
                    angles = t.euler_angles(conv, passive)
                    with self.subTest(frame=name, order=order, conv=conv, passive=passive):
                        self.assertTrue(in_canonical_range(angles, passive), f"{name}: {np.degrees(angles)}")
                        self.assertRebuilds(t, angles, conv, passive, msg=name, tol=1e-8)


# --------------------------------------------------------------------------------------
# F. Axial and degenerate single tensors
# --------------------------------------------------------------------------------------

def unique_x_frames():
    d = np.radians
    angle_sets = [[0, 0, 0], [20, 0, 0], [0, 90, 0], [0, 90, 90], [0, 30, 0], [0, 30, 90],
                  [45, 60, 90], [45, 60, 0], [0, 30, 0], [45, 60, 45], [120, 100, 200], [300, 170, 10],
                  [10, 90, 180], [0, 180, 0], [90, 90, 0], [0, 45, 270]]
    frames = [(f"zyz{a}", R_active_zyz(*d(a))) for a in angle_sets]
    for name, rot in (("x", Rx), ("y", Ry), ("z", Rz)):
        for ang in (30.0, 90.0, 180.0):
            frames.append((f"R{name}({ang})", rot(d(ang))))
    rng = np.random.default_rng(66)
    frames += [(f"random{i}", random_rotation(rng)) for i in range(40)]
    return frames


class TestSingleTensorAxialAndDegenerate(QuietTestCase):

    def test_unique_z_tensors_have_the_free_angle_zero(self):
        # active: gamma = 0; passive: alpha = 0 (both conventions)
        orders = [("h", [-1.0, -1.0, 2.0]), ("n", [-1.0, -1.0, 2.0]), ("n", [1.0, 1.0, -2.0]),
                  ("i", [-1.0, -1.0, 2.0]), ("d", [2.0, 2.0, -1.0]), ("h", [1.5, 1.5, -0.5]),
                  ("i", [-3.0, -3.0, 1.0]), ("h", [2.0, 2.0, 5.0])]
        frames = unique_x_frames()
        for order, evals in orders:
            for j in range(len(frames)):
                if j % 3 and j > 16:  # keep all the special frames, thin the random ones
                    continue
                name, E = frames[j]
                t = tensor_from_frame(evals, E, order=order)
                self.assertEqual(unique_index(t.eigenvalues), 2)
                for conv, passive in COMBOS:
                    angles = t.euler_angles(conv, passive)
                    free = 0 if passive else 2
                    with self.subTest(order=order, evals=evals, frame=name, conv=conv, passive=passive):
                        self.assertAlmostEqual(angles[free], 0.0, places=12, msg=f"{np.degrees(angles)}")
                        self.assertTrue(in_canonical_range(angles, passive))
                        self.assertRebuilds(t, angles, conv, passive, msg=name, tol=1e-8)
                        # the unique axis (PAS z) is written back as +-n
                        axis = R_from(angles, conv, passive)[:, 2]
                        self.assertTrue(close(axis, E[:, 2], 1e-8) or close(axis, -E[:, 2], 1e-8))

    def test_unique_z_data_matrix_input_with_random_axes(self):
        rng = np.random.default_rng(52)
        for i in range(120):
            kind_order = [("h", -1.0, 2.0), ("n", 1.0, -2.0), ("i", -1.0, 2.0), ("d", 2.0, -1.0)][i % 4]
            order, perp, uniq = kind_order
            n = random_unit(rng)
            t = axial_tensor(n, perp, uniq, order)
            for conv, passive in COMBOS:
                angles = t.euler_angles(conv, passive)
                with self.subTest(case=i, order=order, conv=conv, passive=passive):
                    self.assertAlmostEqual(angles[0 if passive else 2], 0.0, places=12)
                    axis = R_from(angles, conv, passive)[:, 2]
                    self.assertTrue(close(axis, n, 1e-8) or close(axis, -n, 1e-8),
                                    f"case {i}: axis {axis} vs {n}")
                    self.assertRebuilds(t, angles, conv, passive, msg=f"case {i}")

    def test_unique_x_tensors_have_active_alpha_zero_or_pi(self):
        # TensorView (MFtoEuler.m): unique along x -> choose the frame with active alpha = 0, then
        # normalise the range as usual, which gives alpha = pi when that frame has beta >= pi/2.
        # Includes the orientations where the paper's closed form (division by sin(g) cos(g)) breaks.
        for order, evals in (("i", [-1.0, 2.0, 2.0]), ("d", [2.0, -1.0, -1.0]), ("i", [0.5, 3.0, 3.0]),
                             ("d", [4.0, 1.0, 1.0])):
            for name, E in unique_x_frames():
                t = tensor_from_frame(evals, E, order=order)
                self.assertEqual(unique_index(t.eigenvalues), 0)
                for conv in CONVENTIONS:
                    angles = t.euler_angles(conv, False)
                    with self.subTest(order=order, frame=name, conv=conv):
                        self.assertTrue(zero_or_pi(angles[0]), f"{name}: {np.degrees(angles)}")
                        self.assertTrue(in_canonical_range(angles, False), f"{name}: {np.degrees(angles)}")
                        # one of the four equivalent sets is TensorView's alpha = 0 frame (Eqs. 16-19)
                        self.assertTrue(any(abs(float(ang_diff(r[0], 0.0))) < 1e-9
                                            for r in paper_single_sets(angles, False)))
                        self.assertRebuilds(t, angles, conv, False, msg=name, tol=1e-8)
                        # unique axis (PAS x) is written back as +-E[:, 0]
                        axis = R_active(angles, conv)[:, 0]
                        self.assertTrue(close(axis, E[:, 0], 1e-8) or close(axis, -E[:, 0], 1e-8),
                                        f"{name}: axis {axis} vs {E[:, 0]}")

    def test_unique_x_passive_angles_describe_the_tensor_and_axis(self):
        for order, evals in (("i", [-1.0, 2.0, 2.0]), ("d", [2.0, -1.0, -1.0])):
            for name, E in unique_x_frames():
                t = tensor_from_frame(evals, E, order=order)
                for conv in CONVENTIONS:
                    angles = t.euler_angles(conv, True)
                    with self.subTest(order=order, frame=name, conv=conv):
                        # passive angles of the active alpha = 0 frame (Eq. 9: gamma_p = -alpha = 0),
                        # normalised, so gamma is 0 or pi
                        self.assertTrue(zero_or_pi(angles[2]), f"{name}: {np.degrees(angles)}")
                        self.assertTrue(in_canonical_range(angles, True), f"{name}: {np.degrees(angles)}")
                        self.assertRebuilds(t, angles, conv, True, msg=name, tol=1e-8)
                        axis = R_passive(angles, conv)[:, 0]
                        self.assertTrue(close(axis, E[:, 0], 1e-8) or close(axis, -E[:, 0], 1e-8))

    def test_unique_x_angles_do_not_depend_on_the_arbitrary_frame_about_x(self):
        # E and E Rx(phi) are the same tensor, so the gauge and the range normalisation must give the
        # same angles (gamma + pi, O4, is allowed for frames on the gamma = pi boundary).
        rng = np.random.default_rng(53)
        frames = unique_x_frames()
        for name, E in frames[::2]:
            t0 = tensor_from_frame([-1.0, 2.0, 2.0], E)
            for phi in rng.uniform(0, TWO_PI, size=3):
                t1 = tensor_from_frame([-1.0, 2.0, 2.0], E @ Rx(phi))
                self.assertTrue(close(t0.data, t1.data, 1e-12))
                for conv in CONVENTIONS:
                    a0, a1 = t0.euler_angles(conv, False), t1.euler_angles(conv, False)
                    with self.subTest(frame=name, phi=phi, conv=conv):
                        self.assertTrue(same_angles(a0, a1, 1e-7) or same_angles(a0, a1 + [0.0, 0.0, PI], 1e-7),
                                        f"{np.degrees(a0)} vs {np.degrees(a1)}")

    def test_unique_x_axis_with_random_axes_from_data_matrix(self):
        rng = np.random.default_rng(54)
        for i in range(120):
            order, perp, uniq = [("i", 2.0, -1.0), ("d", -1.0, 2.0), ("i", 1.0, -2.0), ("d", -0.5, 3.0)][i % 4]
            n = random_unit(rng)
            t = axial_tensor(n, perp, uniq, order)
            self.assertEqual(unique_index(t.eigenvalues), 0)
            for conv in CONVENTIONS:
                for passive in (False, True):
                    angles = t.euler_angles(conv, passive)
                    with self.subTest(case=i, order=order, conv=conv, passive=passive):
                        axis = R_from(angles, conv, passive)[:, 0]
                        self.assertTrue(close(axis, n, 1e-8) or close(axis, -n, 1e-8))
                        self.assertRebuilds(t, angles, conv, passive, msg=f"case {i}")
                        self.assertTrue(zero_or_pi(angles[2 if passive else 0]), np.degrees(angles))
                        self.assertTrue(in_canonical_range(angles, passive))

    def test_axis_on_coordinate_axes_and_planes(self):
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 0], [0, 1, 1],
                [1, 0, 1], [1, 1, 1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [1, 1, -0.5], [1, -1, -1]]
        specs = [("h", -1.0, 2.0, 2), ("n", 1.0, -2.0, 2), ("i", 2.0, -1.0, 0), ("d", -1.0, 2.0, 0)]
        for n in axes:
            n = np.array(n, float) / np.linalg.norm(n)
            for order, perp, uniq, ua in specs:
                t = axial_tensor(n, perp, uniq, order)
                self.assertEqual(unique_index(t.eigenvalues), ua)
                for conv, passive in COMBOS:
                    angles = t.euler_angles(conv, passive)
                    with self.subTest(n=n, order=order, conv=conv, passive=passive):
                        self.assertRebuilds(t, angles, conv, passive, msg=f"n={n}")
                        axis = R_from(angles, conv, passive)[:, ua]
                        self.assertTrue(close(axis, n, 1e-8) or close(axis, -n, 1e-8), f"{axis} vs {n}")

    def test_dipolar_like_tensors_point_along_the_internuclear_vector(self):
        rng = np.random.default_rng(55)
        for sign in (1.0, -1.0):
            for order in ("n", "i"):
                for i in range(20):
                    r = random_unit(rng)
                    D = sign * 3.0 * (3 * np.outer(r, r) - np.eye(3))
                    t = NMRTensor(D, order=order)
                    # the unique axis is z in NQR order, and in increasing order only for the positive sign
                    ua = 2 if (order == "n" or sign > 0) else 0
                    self.assertEqual(unique_index(t.eigenvalues), ua)
                    for conv, passive in COMBOS:
                        angles = t.euler_angles(conv, passive)
                        with self.subTest(sign=sign, order=order, case=i, conv=conv, passive=passive):
                            axis = R_from(angles, conv, passive)[:, ua]
                            self.assertTrue(close(axis, r, 1e-8) or close(axis, -r, 1e-8))
                            self.assertRebuilds(t, angles, conv, passive)
                            if ua == 2:
                                self.assertAlmostEqual(angles[0 if passive else 2], 0.0, places=12)

    def test_dipolar_coupling_object_has_paper_components_and_axis(self):
        # (iii) the dipolar tensor has PAS components (-1, -1, 2) times the coupling
        rng = np.random.default_rng(56)
        g_c, g_h = nmr_gamma("C", iso=13), nmr_gamma("H", iso=1)
        for gammas in ((g_c, g_h), (-g_c, g_h)):  # same and opposite sign gyromagnetic ratios
            for i in range(10):
                r = random_unit(rng) * 1.09
                c = DipolarCoupling.from_distance_vector(r, "13C", "1H", gamma_1=gammas[0], gamma_2=gammas[1])
                ev = c.tensor.eigenvalues
                self.assertAlmostEqual(ev[2], -2 * ev[0], delta=1e-6 * abs(ev[2]))
                self.assertAlmostEqual(ev[0], ev[1], delta=1e-6 * abs(ev[2]))
                for conv, passive in COMBOS:
                    angles = c.tensor.euler_angles(conv, passive)
                    with self.subTest(gammas=gammas, case=i, conv=conv, passive=passive):
                        axis = R_from(angles, conv, passive)[:, 2]
                        rhat = r / np.linalg.norm(r)
                        self.assertTrue(close(axis, rhat, 1e-8) or close(axis, -rhat, 1e-8))
                        self.assertAlmostEqual(angles[0 if passive else 2], 0.0, places=12)

    def test_spherical_tensors_have_zero_angles(self):
        rng = np.random.default_rng(57)
        for iso in (0.0, 1.0, -7.3):
            for order in ("i", "d", "h", "n"):
                for t in (NMRTensor(iso * np.eye(3), order=order),
                          tensor_from_frame([iso] * 3, random_rotation(rng), order=order)):
                    self.assertEqual(t.degeneracy, 3)
                    for conv, passive in COMBOS:
                        np.testing.assert_array_equal(t.euler_angles(conv, passive), np.zeros(3))
                        np.testing.assert_array_equal(t.euler_angles(conv, passive, degrees=True), np.zeros(3))

    def test_nearly_degenerate_tensors_follow_the_degeneracy_tolerance(self):
        # tolerance is 1e-6: a 1e-7 splitting is axial (gauge applies), 1e-5 is triaxial (no gauge)
        rng = np.random.default_rng(58)
        for delta, axial in ((1e-7, True), (1e-5, False)):
            for i in range(25):
                E = random_rotation(rng)
                for evals, ua in (([1.0, 1.0 + delta, 3.0], 2), ([1.0, 3.0 - delta, 3.0], 0)):
                    t = tensor_from_frame(evals, E, order="i")
                    self.assertEqual(t.degeneracy, 2 if axial else 1)
                    for conv, passive in COMBOS:
                        angles = t.euler_angles(conv, passive)
                        with self.subTest(delta=delta, case=i, evals=evals, conv=conv, passive=passive):
                            self.assertTrue(np.all(np.isfinite(angles)))
                            self.assertRebuilds(t, angles, conv, passive, tol=1e-6, msg=f"delta {delta}")
                            if axial and ua == 2:
                                self.assertAlmostEqual(angles[0 if passive else 2], 0.0, places=12)
                            if axial and ua == 0:
                                self.assertTrue(zero_or_pi(angles[2 if passive else 0]), np.degrees(angles))
                            self.assertTrue(in_canonical_range(angles, passive))

    def test_near_spherical_tensors(self):
        for delta, degeneracy in ((1e-8, 3), (1e-5, 1)):
            t = NMRTensor(np.diag([1.0, 1.0 + delta, 1.0 + 2 * delta]), order="i")
            self.assertEqual(t.degeneracy, degeneracy)
            for conv, passive in COMBOS:
                angles = t.euler_angles(conv, passive)
                self.assertTrue(np.all(np.isfinite(angles)))
                if degeneracy == 3:
                    np.testing.assert_array_equal(angles, np.zeros(3))


# --------------------------------------------------------------------------------------
# G. Relative orientation, triaxial tensors
# --------------------------------------------------------------------------------------

class TestRelativeOrientationTriaxial(QuietTestCase):

    def _pairs(self, seed, n):
        rng = np.random.default_rng(seed)
        for i in range(n):
            yield i, random_triaxial(rng, order=("i", "d", "h")[i % 3]), random_triaxial(rng, order=("i", "h", "n")[i % 3])

    def test_euler_to_realises_the_relative_rotation_up_to_symmetries(self):
        for i, A, B in self._pairs(71, 250):
            Rrel = frame_of(A).T @ frame_of(B)
            for conv, passive in COMBOS:
                angles = A.euler_to(B, conv, passive)
                with self.subTest(case=i, conv=conv, passive=passive):
                    self.assertTrue(np.all(np.isfinite(angles)))
                    self.assertTrue(rel_orientation_matches(R_from(angles, conv, passive), Rrel),
                                    f"case {i}: {np.degrees(angles)}")

    def test_rotation_to_is_the_relative_rotation(self):
        for i, A, B in self._pairs(72, 60):
            Rrel = frame_of(A).T @ frame_of(B)
            R = A.rotation_to(B)
            with self.subTest(case=i):
                self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)
                self.assertTrue(close(R, Rrel, 1e-9))

    def test_active_relative_angles_rebuild_tensor_b_in_the_pas_of_a(self):
        # Eq. 31 / 3: the tensor B written in A's PAS is R diag(B) R^T for the active angles (physical check)
        for i, A, B in self._pairs(73, 100):
            B_in_A = frame_of(A).T @ B.data @ frame_of(A)
            for conv in CONVENTIONS:
                R = R_active(A.euler_to(B, conv, False), conv)
                rebuilt = R @ np.diag(B.eigenvalues) @ R.T
                # equal up to a half turn of A's PAS (sign flips of the off-diagonal elements)
                self.assertTrue(any(close(P @ B_in_A @ P.T, rebuilt, 1e-8) for P in HALF_TURNS),
                                f"case {i} {conv}")

    def test_sixteen_equivalent_relative_orientations(self):
        for i, A, B in self._pairs(74, 60):
            Rrel = frame_of(A).T @ frame_of(B)
            for conv, passive in COMBOS:
                rows = A.equivalent_euler_to(B, conv, passive)
                with self.subTest(case=i, conv=conv, passive=passive):
                    self.assertEqual(rows.shape, (16, 3))
                    self.assertEqual(n_distinct_rows(rows), 16)
                    mats = [R_from(r, conv, passive) for r in rows]
                    self.assertEqual(len({tuple(np.round(m, 6).ravel()) for m in mats}), 16)
                    first = R_from(A.euler_to(B, conv, passive), conv, passive)
                    self.assertTrue(matrix_sets_equal(mats, orbit16(first)))
                    for m in mats:
                        self.assertTrue(rel_orientation_matches(m, Rrel))

    def test_active_zyz_sixteen_are_the_papers_o1_to_o16(self):
        for i, A, B in self._pairs(75, 60):
            o1 = A.euler_to(B, "zyz", False)
            rows = A.equivalent_euler_to(B, "zyz", False)
            with self.subTest(case=i):
                self.assertTrue(row_sets_equal(rows, paper_active_zyz_16(o1)),
                                f"case {i}: {np.degrees(o1)}")

    def test_antisymmetry_a_to_b_is_the_inverse_of_b_to_a(self):
        for i, A, B in self._pairs(76, 100):
            for conv, passive in COMBOS:
                ab = R_from(A.euler_to(B, conv, passive), conv, passive)
                ba = R_from(B.euler_to(A, conv, passive), conv, passive)
                with self.subTest(case=i, conv=conv, passive=passive):
                    self.assertTrue(rel_orientation_matches(ab, ba.T), f"case {i}")

    def test_composition_a_to_c_is_a_to_b_then_b_to_c(self):
        rng = np.random.default_rng(77)
        for i in range(80):
            A, B, C = (random_triaxial(rng, order=("i", "d", "h")[int(rng.integers(3))]) for _ in range(3))
            for conv, passive in COMBOS:
                ab = R_from(A.euler_to(B, conv, passive), conv, passive)
                bc = R_from(B.euler_to(C, conv, passive), conv, passive)
                ac = R_from(A.euler_to(C, conv, passive), conv, passive)
                # each is only fixed up to half turns of the shared frame B: allow one in the middle
                with self.subTest(case=i, conv=conv, passive=passive):
                    self.assertTrue(any(rel_orientation_matches(ac, ab @ S @ bc) for S in HALF_TURNS),
                                    f"case {i}")

    def test_passive_angles_are_the_inverse_parametrisation_of_the_active_rotation(self):
        # Cross-check of two Soprano code paths (passive vs active euler_to); the oracle matrices are independent.
        # Both decompose the same R_rel and only the range normalisation differs, which acts on R by a half
        # turn of B's PAS on the right (Eqs. 16-23). So the two matrices agree up to one of those four, not
        # merely up to the 16 symmetries.
        for i, A, B in self._pairs(78, 80):
            Rrel = frame_of(A).T @ frame_of(B)
            for conv in CONVENTIONS:
                active = R_active(A.euler_to(B, conv, False), conv)
                passive = R_passive(A.euler_to(B, conv, True), conv)
                with self.subTest(case=i, conv=conv):
                    self.assertTrue(rel_orientation_matches(active, Rrel))
                    self.assertTrue(any(close(passive, active @ Q, 1e-9) for Q in HALF_TURNS))

    def test_identical_tensors_give_zero_angles_with_a_warning(self):
        rng = np.random.default_rng(79)
        for kind in ("tri", "axz", "axx"):
            t = make_tensor(kind, rng)
            for conv, passive in COMBOS:
                with self.subTest(kind=kind, conv=conv, passive=passive):
                    with self.assertWarns(UserWarning):
                        angles = t.euler_to(t.copy(), conv, passive)
                    np.testing.assert_array_equal(angles, np.zeros(3))

    def test_fig5_relative_orientation_from_the_paper(self):
        # B oriented in the PAS of A by the paper's first set; A has the identity frame.
        first = np.radians([4.1, 89.7, 154.7])
        second = np.radians([175.9, 89.7, 205.1])
        A = tensor_from_frame([1.0, 2.0, 3.0], np.eye(3))
        for source, other in ((first, second), (second, first)):
            B = tensor_from_frame([-4.0, 0.5, 7.0], R_active_zyz(*source))
            rows = A.equivalent_euler_to(B, "zyz", False)
            returned = A.euler_to(B, "zyz", False)
            with self.subTest(source=np.degrees(source)):
                self.assertTrue(any(close(R_active_zyz(*returned), M, 6e-3) for M in orbit16(R_active_zyz(*first))))
                # the 16-set contains both literal sets of Fig. 5 (the second within the paper's 0.2 deg discrepancy)
                self.assertTrue(contains_row(rows, first, np.radians(0.25)))
                self.assertTrue(contains_row(rows, second, np.radians(0.25)))
                self.assertTrue(contains_row(rows, other, np.radians(0.25)))
        # exactly: the first set is returned as itself (beta < pi/2, gamma < pi: no renormalisation)
        B = tensor_from_frame([-4.0, 0.5, 7.0], R_active_zyz(*first))
        np.testing.assert_allclose(np.degrees(A.euler_to(B, "zyz", False)), [4.1, 89.7, 154.7], atol=1e-8)


# --------------------------------------------------------------------------------------
# H. Relative orientation with axial tensors
# --------------------------------------------------------------------------------------

KINDS = ("tri", "axz", "axx")


def BOTH_AXIAL_ZYZ(beta, passive):
    """TensorView's ZYZ angles for two axial tensors (beta < pi/2): active (0, beta, 0) (Section 6).

    The passive angles of that rotation are (-0, -beta, -0) (Eq. 9) = (pi, beta, pi) (Eq. 8), which
    the alpha < pi rule turns into (0, beta, pi); TensorView's RelativeAngles.m displays the same.
    """
    return [0.0, beta, PI] if passive else [0.0, beta, 0.0]


class TestRelativeOrientationAxial(QuietTestCase):

    def _check_pair(self, A, B, conv, passive, angles, msg):
        ua, ub = unique_index(A.eigenvalues), unique_index(B.eigenvalues)
        Rrel = frame_of(A).T @ frame_of(B)
        R = R_from(angles, conv, passive)
        self.assertTrue(np.all(np.isfinite(angles)), msg)
        self.assertTrue(rel_orientation_matches(R, Rrel, ua, ub), f"{msg}: angles {np.degrees(angles)}")
        # physical statement: B written in A's PAS, up to what the axial tensors are blind to
        B_in_A = frame_of(A).T @ B.data @ frame_of(A)
        rebuilt = R @ np.diag(B.eigenvalues) @ R.T
        if ua is None:
            self.assertTrue(any(close(P @ B_in_A @ P.T, rebuilt, 1e-8) for P in HALF_TURNS), msg)
        else:
            self.assertTrue(close(b_in_a_invariants(B_in_A, ua), b_in_a_invariants(rebuilt, ua), 1e-8), msg)

    def test_every_pair_type_realises_the_relative_orientation(self):
        for ka in KINDS:
            for kb in KINDS:
                rng = np.random.default_rng(90 + 3 * KINDS.index(ka) + KINDS.index(kb))
                for i in range(25):
                    A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                    for conv, passive in COMBOS:
                        angles = A.euler_to(B, conv, passive)
                        with self.subTest(pair=(ka, kb), case=i, conv=conv, passive=passive):
                            self._check_pair(A, B, conv, passive, angles, f"{ka}->{kb} case {i} {conv} passive={passive}")

    def test_free_angle_gauge_for_one_axial_tensor(self):
        # A axial (unique z): first active angle 0 / last passive; B axial (unique z): last active / first passive
        for ka, kb in (("axz", "tri"), ("tri", "axz")):
            rng = np.random.default_rng(100 + KINDS.index(ka))
            for i in range(40):
                A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                for conv, passive in COMBOS:
                    angles = A.euler_to(B, conv, passive)
                    if ka == "axz":
                        idx = 2 if passive else 0
                    else:
                        idx = 0 if passive else 2
                    with self.subTest(pair=(ka, kb), case=i, conv=conv, passive=passive):
                        self.assertAlmostEqual(angles[idx], 0.0, places=12, msg=f"{np.degrees(angles)}")

    def test_unique_x_axial_with_triaxial_is_exact(self):
        # Without a gauge choice the plain decomposition of the relative rotation is returned
        for ka, kb in (("axx", "tri"), ("tri", "axx"), ("axx", "axx")):
            rng = np.random.default_rng(110)
            for i in range(25):
                A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                for conv, passive in COMBOS:
                    angles = A.euler_to(B, conv, passive)
                    with self.subTest(pair=(ka, kb), case=i, conv=conv, passive=passive):
                        self._check_pair(A, B, conv, passive, angles, f"{ka}->{kb} case {i}")

    def test_both_axial_zyz_is_zero_beta_zero_with_papers_asin_formula(self):
        rng = np.random.default_rng(120)
        for ka, kb in (("axz", "axz"), ("axz", "axx")):
            for i in range(60):
                A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                EA = frame_of(A)
                B_in_A = EA.T @ B.data @ EA
                evB = B.eigenvalues
                ratio = (B_in_A[2, 2] - evB[2]) / (evB[0] - evB[2])
                beta = np.arcsin(np.sqrt(np.clip(ratio, 0.0, 1.0)))  # the paper's formula, positive root
                for passive in (False, True):
                    angles = A.euler_to(B, "zyz", passive)
                    with self.subTest(pair=(ka, kb), case=i, passive=passive):
                        np.testing.assert_allclose(angles, BOTH_AXIAL_ZYZ(beta, passive), atol=1e-6)
                        self._check_pair(A, B, "zyz", passive, angles, f"{ka}->{kb} case {i}")

    def test_both_axial_zyz_beta_is_theta_or_its_complement(self):
        # beta = theta between the unique axes if both are along the same PAS axis, else pi/2 - theta
        rng = np.random.default_rng(121)
        for ka in ("axz", "axx"):
            for kb in ("axz", "axx"):
                for i in range(40):
                    A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                    theta = theta_between(A, B)
                    expected = theta if ka == kb else HALF_PI - theta
                    for passive in (False, True):
                        angles = A.euler_to(B, "zyz", passive)
                        with self.subTest(pair=(ka, kb), case=i, passive=passive):
                            np.testing.assert_allclose(angles, BOTH_AXIAL_ZYZ(expected, passive), atol=1e-6)

    def test_both_axial_zxz_table1(self):
        # Table 1. Its rows are keyed on the unique axis of B, the tensor being oriented (TensorView's
        # RelativeAngles.m tests PASv1, the eigenvalues of Tensor 1 = B). With A unique along z:
        # B unique along z -> (90 deg, theta, 0); along x -> (0, 90 deg, pi/2 - theta). TensorView then
        # normalises the range (beta >= 90 deg -> Eq. 18, gamma >= 180 deg -> Eq. 19), so the second row
        # is displayed, and returned, as (180 deg, 90 deg, 180 deg - (pi/2 - theta)).
        rng = np.random.default_rng(122)
        for kb in ("axz", "axx"):
            for i in range(60):
                A, B = make_tensor("axz", rng), make_tensor(kb, rng)
                theta = theta_between(A, B)
                table = [HALF_PI, theta, 0.0] if kb == "axz" else [0.0, HALF_PI, HALF_PI - theta]
                displayed = table if kb == "axz" else [PI, HALF_PI, PI - (HALF_PI - theta)]
                angles = A.euler_to(B, "zxz", False)
                with self.subTest(kb=kb, case=i):
                    np.testing.assert_allclose(angles, displayed, atol=1e-6)
                    self.assertTrue(contains_row(paper_single_sets(table, False), displayed, 1e-9))
                    # the passive angles describe the same rotation (Eq. 9) up to a half turn of B's PAS
                    passive = A.euler_to(B, "zxz", True)
                    self.assertTrue(in_canonical_range(passive, True))
                    self.assertTrue(any(close(R_passive(passive, "zxz"), R_active_zxz(*table) @ Q, 1e-6)
                                        for Q in HALF_TURNS))

    def test_both_axial_zxz_with_unique_x_first_tensor(self):
        # Not covered by the paper: its asin formula uses B written along A's PAS z, which for A unique along
        # x is one of A's arbitrary perpendicular axes (TensorView's output then does not reproduce the tensors, see
        # TestAgainstTensorViewSoftware). Physical check (|R[axis_a, axis_b]| = cos theta) instead.
        rng = np.random.default_rng(123)
        for kb in ("axz", "axx"):
            for i in range(40):
                A, B = make_tensor("axx", rng), make_tensor(kb, rng)
                theta = theta_between(A, B)
                for passive in (False, True):
                    angles = A.euler_to(B, "zxz", passive)
                    R = R_from(angles, "zxz", passive)
                    with self.subTest(kb=kb, case=i, passive=passive):
                        self.assertAlmostEqual(abs(R[0, 0 if kb == "axx" else 2]), np.cos(theta), places=8)
                        self._check_pair(A, B, "zxz", passive, angles, f"axx->{kb} case {i}")

    def test_both_axial_pairs_realise_the_invariant_for_all_orders_and_directions(self):
        rng = np.random.default_rng(124)
        for ka in ("axz", "axx"):
            for kb in ("axz", "axx"):
                for i in range(30):
                    A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                    theta = theta_between(A, B)
                    ua, ub = unique_index(A.eigenvalues), unique_index(B.eigenvalues)
                    for conv, passive in COMBOS:
                        R = R_from(A.euler_to(B, conv, passive), conv, passive)
                        with self.subTest(pair=(ka, kb), case=i, conv=conv, passive=passive):
                            self.assertAlmostEqual(abs(R[ua, ub]), np.cos(theta), places=8)

    def test_regression_tilted_axial_a_against_axial_b_along_z_is_not_aligned(self):
        # unique axis n = (1, 1, -0.5)/1.5 makes 70.53 deg (cos = 1/3) with z: neither zeros nor 'aligned'
        theta = np.arccos(1.0 / 3.0)
        n = np.array([1.0, 1.0, -0.5]) / 1.5
        for order_a, perp_a, uniq_a in AXIAL_SPECS["z"]:
            for order_b, perp_b, uniq_b in AXIAL_SPECS["z"][:3]:
                A = axial_tensor(n, perp_a, uniq_a, order_a)
                B = axial_tensor([0, 0, 1], perp_b, uniq_b, order_b)
                for conv, passive in COMBOS:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        angles = A.euler_to(B, conv, passive)
                    with self.subTest(order_a=order_a, order_b=order_b, conv=conv, passive=passive):
                        self.assertFalse(any("aligned" in str(w.message) for w in caught), [str(w.message) for w in caught])
                        self.assertGreater(np.max(np.abs(angles)), 0.5)
                        # both unique along z: ZYZ (0, theta, 0); ZXZ (90 deg, theta, 0) in the active sense
                        if not passive:
                            expected = [0, theta, 0] if conv == "zyz" else [HALF_PI, theta, 0]
                            np.testing.assert_allclose(angles, expected, atol=1e-6)
                        R = R_from(angles, conv, passive)
                        self.assertAlmostEqual(abs(R[2, 2]), 1.0 / 3.0, places=8)

    def test_regression_tilted_unique_x_axial_against_axial_along_z(self):
        theta = np.arccos(1.0 / 3.0)
        n = np.array([1.0, 1.0, -0.5]) / 1.5
        A = axial_tensor(n, 2.0, -1.0, "i")  # unique axis along PAS x
        B = axial_tensor([0, 0, 1], -1.0, 2.0, "h")  # unique axis along PAS z
        self.assertEqual((unique_index(A.eigenvalues), unique_index(B.eigenvalues)), (0, 2))
        for conv, passive in COMBOS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                angles = A.euler_to(B, conv, passive)
                back = B.euler_to(A, conv, passive)
            with self.subTest(conv=conv, passive=passive):
                self.assertFalse(any("aligned" in str(w.message) for w in caught))
                R, Rb = R_from(angles, conv, passive), R_from(back, conv, passive)
                self.assertAlmostEqual(abs(R[0, 2]), 1.0 / 3.0, places=8)
                self.assertAlmostEqual(abs(Rb[2, 0]), 1.0 / 3.0, places=8)
                if conv == "zyz":
                    # different labels: beta = pi/2 - theta
                    np.testing.assert_allclose(angles, BOTH_AXIAL_ZYZ(HALF_PI - theta, passive), atol=1e-6)
                    np.testing.assert_allclose(back, BOTH_AXIAL_ZYZ(HALF_PI - theta, passive), atol=1e-6)

    def test_parallel_and_antiparallel_unique_axes_give_zeros_with_a_warning(self):
        rng = np.random.default_rng(130)
        for i in range(10):
            E = random_rotation(rng)
            A = tensor_from_frame([-1.0, -1.0, 2.0], E, order="h")
            variants = {
                "parallel, other frame about z": E @ Rz(rng.uniform(0.2, 6.0)),
                "antiparallel": E @ Rx(PI) @ Rz(rng.uniform(0.2, 6.0)),
            }
            for name, EB in variants.items():
                B = tensor_from_frame([-0.5, -0.5, 1.0], EB, order="h")
                for conv, passive in COMBOS:
                    with self.subTest(case=i, kind=name, conv=conv, passive=passive):
                        with self.assertWarnsRegex(UserWarning, "aligned"):
                            angles = A.euler_to(B, conv, passive)
                        np.testing.assert_allclose(angles, np.zeros(3), atol=1e-9)

    def test_parallel_unique_x_axes_give_zeros_with_a_warning(self):
        rng = np.random.default_rng(131)
        for i in range(10):
            E = random_rotation(rng)
            A = tensor_from_frame([-1.0, 2.0, 2.0], E, order="i")
            B = tensor_from_frame([-3.0, 1.0, 1.0], E @ Rx(rng.uniform(0.2, 6.0)), order="i")
            for conv, passive in COMBOS:
                with self.subTest(case=i, conv=conv, passive=passive):
                    with self.assertWarnsRegex(UserWarning, "aligned"):
                        angles = A.euler_to(B, conv, passive)
                    np.testing.assert_allclose(angles, np.zeros(3), atol=1e-9)

    def test_parallel_unique_axes_with_different_labels(self):
        # A unique along PAS z, B unique along PAS x, both along the same direction in the lab: B's x
        # has to be carried onto A's z, so |R[2, 0]| = 1 (|R[0, 2]| = 1 the other way round) and the
        # relative rotation is NOT the identity, whatever the free angles are.
        rng = np.random.default_rng(132)
        for i in range(10):
            n = random_unit(rng)
            Z = axial_tensor(n, -1.0, 2.0, "h")
            X = axial_tensor(n, 2.0, -1.0, "i")
            self.assertEqual((unique_index(Z.eigenvalues), unique_index(X.eigenvalues)), (2, 0))
            for (A, B, ua, ub) in ((Z, X, 2, 0), (X, Z, 0, 2)):
                for conv, passive in COMBOS:
                    angles = A.euler_to(B, conv, passive)
                    with self.subTest(case=i, first="Z" if A is Z else "X", conv=conv, passive=passive):
                        R = R_from(angles, conv, passive)
                        self.assertAlmostEqual(abs(R[ua, ub]), 1.0, places=8, msg=f"{np.degrees(angles)}")

    def test_perpendicular_unique_axes_follow_tensorviews_normalisation(self):
        # beta = pi/2 exactly. The gauge gives ZYZ (0, 90, 0) and ZXZ (90, 90, 0) (active); TensorView then
        # applies its beta >= 90 deg rule (Eq. 18 active, Eq. 22 passive) and the gamma / alpha < 180 deg rule,
        # so it displays the literal values below. Each is an equivalent set of the gauge-fixed angles.
        expected = {("zyz", False): [180, 90, 0], ("zxz", False): [270, 90, 0],
                    ("zyz", True): [0, 90, 0], ("zxz", True): [0, 90, 270]}
        gauge_active = {"zyz": [0, 90, 0], "zxz": [90, 90, 0]}
        A = axial_tensor([0, 0, 1], -1.0, 2.0, "h")
        B = axial_tensor([1, 0, 0], -1.0, 2.0, "h")  # both unique along PAS z, perpendicular in the lab
        for conv, passive in COMBOS:
            angles = A.euler_to(B, conv, passive)
            with self.subTest(conv=conv, passive=passive):
                np.testing.assert_allclose(np.degrees(angles), expected[(conv, passive)], atol=1e-7)
                R = R_from(angles, conv, passive)
                self.assertAlmostEqual(abs(R[2, 2]), 0.0, places=10)
                self.assertTrue(any(close(R, R_active(np.radians(gauge_active[conv]), conv) @ Q, 1e-9)
                                    for Q in HALF_TURNS))

    def test_perpendicular_unique_axes_of_different_labels(self):
        # A unique z, B unique x, unique axes perpendicular: theta = pi/2, ZYZ beta = pi/2 - theta = 0.
        # ZXZ Table 1 row 2 is (0, 90, 0), displayed by TensorView as (180, 90, 0) (Eqs. 18, 19).
        A = axial_tensor([0, 0, 1], -1.0, 2.0, "h")
        B = axial_tensor([0, 1, 0], 2.0, -1.0, "i")
        self.assertEqual((unique_index(A.eigenvalues), unique_index(B.eigenvalues)), (2, 0))
        for passive in (False, True):
            np.testing.assert_allclose(A.euler_to(B, "zyz", passive), [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(A.euler_to(B, "zxz", False), [PI, HALF_PI, 0], atol=1e-9)

    def test_equivalent_rows_reproduce_the_same_invariants_for_axial_pairs(self):
        for ka in KINDS:
            for kb in KINDS:
                if ka == kb == "tri":
                    continue
                rng = np.random.default_rng(140 + 3 * KINDS.index(ka) + KINDS.index(kb))
                for i in range(8):
                    A, B = make_tensor(ka, rng), make_tensor(kb, rng)
                    for conv, passive in COMBOS:
                        rows = A.equivalent_euler_to(B, conv, passive)
                        with self.subTest(pair=(ka, kb), case=i, conv=conv, passive=passive):
                            self.assertEqual(rows.shape, (16, 3))
                            for j, row in enumerate(rows):
                                self._check_pair(A, B, conv, passive, row, f"{ka}->{kb} case {i} row {j}")

    def test_axial_dipolar_pair_matrix(self):
        # The typical CSA/EFG-in-a-dipolar-frame use: dipolar (unique z in NQR order) and axial CSA
        rng = np.random.default_rng(150)
        for i in range(20):
            r = random_unit(rng)
            D = NMRTensor(2.0 * (3 * np.outer(r, r) - np.eye(3)), order="n")
            n = random_unit(rng)
            for order, perp, uniq in (("h", 1.5, -0.5), ("i", 2.0, -1.0)):
                C = axial_tensor(n, perp, uniq, order)
                theta = float(np.arccos(min(1.0, abs(r @ n))))
                ub = unique_index(C.eigenvalues)
                for conv in CONVENTIONS:
                    angles = D.euler_to(C, conv, False)
                    with self.subTest(case=i, order=order, conv=conv):
                        R = R_active(angles, conv)
                        self.assertAlmostEqual(abs(R[2, ub]), np.cos(theta), places=8)


# --------------------------------------------------------------------------------------
# H2. TensorView's own software (stored output of TensorView for MATLAB run in Octave)
# --------------------------------------------------------------------------------------

TENSORVIEW_REFERENCE = Path(__file__).parent / "test_data" / "tensorview_reference.json"
TENSORVIEW_MODES = {"AZYZ": ("zyz", False), "PZYZ": ("zyz", True), "AZXZ": ("zxz", False), "PZXZ": ("zxz", True)}


class TestAgainstTensorViewSoftware(QuietTestCase):
    """Compare with numbers produced by TensorView's ``MFtoEuler.m`` and ``RelativeAngles.m``.

    ``tests/reference_generators/tensorview.py`` runs TensorView (pinned commit) in Octave and
    stores its output, so this needs neither MATLAB nor Octave. ``RelativeAngles(Tensor1, Tensor2)``
    decomposes ``U2^-1 U1``, the orientation of Tensor 1 in the PAS of Tensor 2, so each relative case
    was run with Tensor 1 = B and Tensor 2 = A and is compared with ``A.euler_to(B)``.
    """

    @classmethod
    def setUpClass(cls):
        data = json.loads(TENSORVIEW_REFERENCE.read_text())
        cls.data = data
        # TensorView uses the symmetric part (the ALA shielding tensor is not symmetric)
        cls.tensors = {k: (np.array(v) + np.array(v).T) / 2 for k, v in data["tensors"].items()}

    def _tensor(self, name, order):
        return NMRTensor(self.tensors[name], order=order)

    def test_reference_angles_rebuild_their_tensors_with_the_oracle(self):
        # validates the stored data and that TensorView's matrices are the oracle's (Eqs. 3, 9, 65, 75)
        for case in self.data["single"]:
            t = self._tensor(case["tensor"], case["order"])
            for mode, angles in case["angles"].items():
                conv, passive = TENSORVIEW_MODES[mode]
                with self.subTest(tensor=case["tensor"], order=case["order"], mode=mode):
                    self.assertRebuilds(t, angles, conv, passive, tol=1e-8)

    def test_single_tensor_angles_are_tensorviews_literal_numbers(self):
        # the canonical member of the four equivalent sets is unique, so the numbers must agree exactly;
        # covers triaxial tensors, unique axis along z and along x, in all three orders and four modes
        for case in self.data["single"]:
            t = self._tensor(case["tensor"], case["order"])
            for mode, expected in case["angles"].items():
                conv, passive = TENSORVIEW_MODES[mode]
                with self.subTest(tensor=case["tensor"], order=case["order"], mode=mode):
                    self.assertTrue(same_angles(t.euler_angles(conv, passive), expected, 1e-7),
                                    f"{np.degrees(t.euler_angles(conv, passive))} vs {np.degrees(expected)}")

    def _relative_cases(self):
        for case in self.data["relative"]:
            A, B = self._tensor(case["A"], case["order"]), self._tensor(case["B"], case["order"])
            yield case, A, B, unique_index(A.eigenvalues), unique_index(B.eigenvalues)

    def test_euler_to_is_tensorview_with_tensor1_as_other(self):
        # Which of the 16 equivalent sets each program reports depends on the eigenvector signs of its
        # eigensolver, so the comparison is up to those 16 (the same formulas for every mode, Eqs. 48-63).
        # Excluded: two axial tensors with A's unique axis along x, where TensorView does not reproduce the tensors (next test).
        for case, A, B, ua, ub in self._relative_cases():
            if ua == 0 and ub is not None:
                continue
            for mode, tv in case["angles"].items():
                conv, passive = TENSORVIEW_MODES[mode]
                angles = A.euler_to(B, conv, passive)
                with self.subTest(A=case["A"], B=case["B"], order=case["order"], mode=mode):
                    self.assertTrue(contains_row(paper_active_zyz_16(tv), angles, 1e-6),
                                    f"{np.degrees(angles)} not equivalent to TensorView {np.degrees(tv)}")
                    self.assertTrue(in_canonical_range(angles, passive))

    def test_the_reverse_call_is_not_tensorview(self):
        # The reverse call, B.euler_to(A), is the orientation of A in the PAS of B; for triaxial pairs it is
        # not among TensorView's 16 sets, so the comparison above pins the direction.
        n = 0
        for case, A, B, ua, ub in self._relative_cases():
            if ua is not None or ub is not None:
                continue
            for mode, tv in case["angles"].items():
                conv, passive = TENSORVIEW_MODES[mode]
                with self.subTest(A=case["A"], B=case["B"], order=case["order"], mode=mode):
                    self.assertFalse(contains_row(paper_active_zyz_16(tv), B.euler_to(A, conv, passive), 1e-6))
                    n += 1
        self.assertGreater(n, 20)

    def test_both_axial_with_a_unique_along_x_tensorview_does_not_reproduce_the_tensors_but_soprano_does(self):
        # TensorView's asin formula (Eq. 79, Table 1) reads B along A's PAS z. With A's unique axis along x,
        # that is an arbitrary perpendicular axis of A, so TensorView's angles fail the physical invariant
        # |R[axis_a, axis_b]| = cos(theta) (theta from the tensor data with eigh), whereas Soprano's satisfy it.
        n = 0
        for case, A, B, ua, ub in self._relative_cases():
            if not (ua == 0 and ub is not None):
                continue
            theta = theta_between(A, B)
            for mode, tv in case["angles"].items():
                conv, passive = TENSORVIEW_MODES[mode]
                ours = R_from(A.euler_to(B, conv, passive), conv, passive)
                theirs = R_from(tv, conv, passive)
                with self.subTest(A=case["A"], B=case["B"], order=case["order"], mode=mode):
                    self.assertAlmostEqual(abs(ours[ua, ub]), np.cos(theta), places=8)
                    self.assertGreater(abs(abs(theirs[ua, ub]) - np.cos(theta)), 1e-3)
                    n += 1
        self.assertGreater(n, 0)

    def test_ala_example_relative_angles(self):
        # The ALA tensors of nmr_tests.test_relative_euler_angles, in TensorView's own increasing ('Ascending')
        # order: the numbers TensorView displays for Tensor 1 = EFG, Tensor 2 = MS (active ZYZ) and the
        # other way round. Soprano's ms.euler_to(efg) is equivalent to the first and not to the second.
        case = next(c for c in self.data["relative"] if (c["A"], c["B"], c["order"]) == ("ala_ms", "ala_efg", "i"))
        np.testing.assert_allclose(np.degrees(case["angles"]["AZYZ"]), [24.80660839, 89.95022697, 155.10491563],
                                   atol=1e-6)
        ms, efg = self._tensor("ala_ms", "i"), self._tensor("ala_efg", "i")
        ours = ms.euler_to(efg, "zyz", False)
        self.assertTrue(contains_row(paper_active_zyz_16(case["angles"]["AZYZ"]), ours, 1e-6))
        ms_in_efg_pas = np.radians([204.89508437, 89.95022697, 155.19339161])  # TensorView, Tensor 1 = MS
        self.assertFalse(contains_row(paper_active_zyz_16(ms_in_efg_pas), ours, 1e-6))


# --------------------------------------------------------------------------------------
# I. mrsimulator mapping
# --------------------------------------------------------------------------------------

def mrs_frame(angles):
    """mrsimulator's code: E = Rz(gamma) Ry(beta) Rz(alpha) (explicit matrices)."""
    alpha, beta, gamma = angles
    return Rz(gamma) @ Ry(beta) @ Rz(alpha)


@unittest.skipUnless(hasattr(NMRTensor, "mrsimulator_euler_angles"), "mrsimulator_euler_angles not available")
class TestMrsimulatorMapping(QuietTestCase):

    def _rebuilds(self, t, msg):
        angles = t.mrsimulator_euler_angles()
        E = mrs_frame(angles)
        err = float(np.max(np.abs(E @ np.diag(t.eigenvalues) @ E.T - t.data)))
        self.assertLess(err, 1e-9, f"{msg}: {np.degrees(angles)} error {err}")
        self.assertTrue(np.all(angles >= 0) and np.all(angles < TWO_PI), msg)

    def test_random_triaxial_tensors_are_rebuilt(self):
        rng = np.random.default_rng(160)
        for i in range(150):
            self._rebuilds(random_triaxial(rng, order=("i", "d", "h", "n")[i % 4]), f"case {i}")

    def test_axial_tensors_are_rebuilt(self):
        rng = np.random.default_rng(161)
        for kind in ("axz", "axx"):
            for i in range(60):
                self._rebuilds(make_tensor(kind, rng), f"{kind} case {i}")

    def test_unique_z_free_angle_is_zero(self):
        rng = np.random.default_rng(162)
        for i in range(30):
            self.assertEqual(make_tensor("axz", rng).mrsimulator_euler_angles()[0], 0.0)

    def test_spherical_tensor_gives_zero_angles(self):
        for iso in (0.0, 2.5):
            t = NMRTensor(iso * np.eye(3))
            np.testing.assert_array_equal(t.mrsimulator_euler_angles(), np.zeros(3))
            np.testing.assert_array_equal(t.mrsimulator_euler_angles(degrees=True), np.zeros(3))

    def test_special_orientations_are_rebuilt(self):
        for name, E in special_frames()[::2]:
            self._rebuilds(tensor_from_frame([1.0, 2.0, 3.0], E), name)

    def test_gamma_beta_alpha_is_an_active_zyz_equivalent_set(self):
        # Cross-check of two Soprano code paths: mrsimulator's Rz(g) Ry(b) Rz(a) is the active ZYZ
        # rotation with alpha and gamma interchanged, so (gamma, beta, alpha) is one of the four
        # active-ZYZ equivalent sets from equivalent_euler_angles.
        rng = np.random.default_rng(163)
        for i in range(100):
            t = random_triaxial(rng, order=("i", "d", "h")[i % 3])
            a, b, g = t.mrsimulator_euler_angles()
            rows = t.equivalent_euler_angles("zyz", passive=False)
            with self.subTest(case=i):
                self.assertTrue(contains_row(rows, [g, b, a], 1e-8), f"{np.degrees([g, b, a])} vs {np.degrees(rows)}")
        for i in range(40):
            t = make_tensor("axz", rng)
            a, b, g = t.mrsimulator_euler_angles()
            with self.subTest(kind="axz", case=i):
                self.assertTrue(contains_row(t.equivalent_euler_angles("zyz", False), [g, b, a], 1e-8))

    def test_degrees_flag(self):
        rng = np.random.default_rng(164)
        t = random_triaxial(rng)
        np.testing.assert_allclose(t.mrsimulator_euler_angles(degrees=True),
                                   t.mrsimulator_euler_angles() * 180 / PI, rtol=1e-12)

    def test_simpson_numbers_read_the_mrsimulator_way_give_the_x_flipped_tensor(self):
        # Rz(g) Ry(b) Rz(a) = M [Rz(-g) Ry(-b) Rz(-a)] M with M = diag(-1, 1, 1) (a mirror conjugation flips the sense
        # of every elementary rotation), so SIMPSON's passive numbers read as mrsimulator angles give M T M,
        # the tensor turned by pi about x.
        M = np.diag([-1.0, 1.0, 1.0])
        rng = np.random.default_rng(165)
        cases = [random_triaxial(rng) for _ in range(40)] + [make_tensor("axz", rng) for _ in range(10)] \
            + [make_tensor("axx", rng) for _ in range(10)]
        for i, t in enumerate(cases):
            passive = t.euler_angles("zyz", passive=True)
            E = mrs_frame(passive)
            read_as_mrs = E @ np.diag(t.eigenvalues) @ E.T
            with self.subTest(case=i):
                self.assertTrue(close(read_as_mrs, M @ t.data @ M, 1e-9))
                self.assertTrue(close(read_as_mrs, Rx(PI) @ t.data @ Rx(PI).T, 1e-9))

    def test_mrsimulator_and_simpson_angles_are_related_by_the_literal_formula(self):
        # (alpha, beta, gamma)_mrs = (-alpha_p - pi, beta_p, pi - gamma_p) up to 2pi for the same frame (Eq. 8 + 9)
        rng = np.random.default_rng(166)
        for i in range(60):
            t = random_triaxial(rng)
            ap, bp, gp = t.euler_angles("zyz", passive=True)
            am, bm, gm = t.mrsimulator_euler_angles()
            with self.subTest(case=i):
                self.assertTrue(same_angles([am, bm, gm], [-ap - PI, bp, PI - gp], 1e-9))
                # as matrices, mrsimulator's frame equals the passive frame exactly
                self.assertTrue(close(mrs_frame([am, bm, gm]), R_passive([ap, bp, gp], "zyz"), 1e-9))

    @unittest.skipIf(mrs_ct is None, "mrsimulator is not available")
    def test_against_mrsimulators_own_forward_function(self):
        rng = np.random.default_rng(167)
        for i in range(20):
            t = random_triaxial(rng, order="h")
            zeta, eta = t.reduced_anisotropy, t.asymmetry
            rebuilt = mrs_ct.from_haeberlen_params(
                euler_angles=list(t.mrsimulator_euler_angles()), zeta_sigma=zeta, eta_sigma=eta,
                isotropic_component=np.trace(t.data) / 3)
            np.testing.assert_allclose(rebuilt, t.data, atol=1e-9)

    def test_site_writers_use_the_documented_conventions(self):
        # convention mapping only; the round trips are in nmr_export_orientation_tests.py
        rng = np.random.default_rng(168)
        E, ev = random_rotation(rng), np.array([30.0, 90.0, 190.0])
        sigma = E @ np.diag(ev) @ E.T
        site = Site(isotope="13C", label="C1", index=0,
                    ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))
        written = np.radians([float(x) for x in site.to_simpson()[0].split()[5:8]])
        # SIMPSON: passive ZYZ of the (negated) shift tensor, Haeberlen order
        shift = NMRTensor(-sigma, order="h")
        self.assertTrue(same_angles(written, shift.euler_angles("zyz", passive=True), 1e-6))
        R = R_passive(written, "zyz")
        self.assertTrue(close(R @ np.diag(shift.eigenvalues) @ R.T, -sigma, 1e-6))
        if hasattr(Site, "to_mrsimulator"):
            mrs = site.to_mrsimulator()["shielding_symmetric"]
            ms = NMRTensor(sigma, order="h")
            self.assertTrue(same_angles([mrs["alpha"], mrs["beta"], mrs["gamma"]], ms.mrsimulator_euler_angles(), 1e-9))


# --------------------------------------------------------------------------------------
# J. Error handling
# --------------------------------------------------------------------------------------

class TestEulerErrorHandling(unittest.TestCase):

    def test_unsupported_convention_warns_and_does_not_crash_for_euler_angles(self):
        rng = np.random.default_rng(170)
        t = random_triaxial(rng)
        with self.assertWarnsRegex(UserWarning, "not handled"):
            angles = t.euler_angles("xyz")
        self.assertEqual(np.shape(angles), (3,))
        self.assertTrue(np.all(np.isfinite(angles)))

    def test_unsupported_convention_warns_for_euler_to_of_axial_tensors(self):
        rng = np.random.default_rng(171)
        A, B = make_tensor("axz", rng), make_tensor("tri", rng)
        with self.assertWarnsRegex(UserWarning, "zyz and zxz"):
            angles = A.euler_to(B, "xyz")
        self.assertEqual(np.shape(angles), (3,))

    def test_zyz_convention_is_case_insensitive(self):
        rng = np.random.default_rng(172)
        t = random_triaxial(rng)
        for conv in CONVENTIONS:
            for passive in (False, True):
                np.testing.assert_allclose(t.euler_angles(conv.upper(), passive), t.euler_angles(conv, passive),
                                           atol=1e-12)

    def test_unique_y_eigenvalue_pattern_is_rejected(self):
        # A [a, b, a] pattern can only be built from an unsorted (evals, evecs) pair in NQR order, where
        # equal absolute values keep their given order. It has no TensorView convention and raises.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = NMRTensor((np.array([1.0, -1.0, 1.0]), np.eye(3)), order="n")
            self.assertEqual(unique_index(t.eigenvalues), 1)
            for conv, passive in COMBOS:
                with self.assertRaises(ValueError):
                    t.euler_angles(conv, passive)

    def test_both_axial_euler_to_with_unique_y_is_rejected(self):
        # Same pattern as above ([a, b, a], NQR order). euler_to must raise like euler_angles
        # rather than fall through to the generic decomposition.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_y = NMRTensor((np.array([1.0, -1.0, 1.0]), np.eye(3)), order="n")
            t_z = NMRTensor(np.diag([-1.0, -1.0, 2.0]), order="i")
            for conv, passive in COMBOS:
                for a, b in ((t_y, t_z), (t_z, t_y)):
                    with self.assertRaises(ValueError):
                        a.euler_to(b, conv, passive)

    def test_invalid_orders_and_shapes_raise(self):
        with self.assertRaises(ValueError):
            NMRTensor(np.diag([1.0, 2.0, 3.0]), order="x")
        with self.assertRaises(ValueError):
            NMRTensor(np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
