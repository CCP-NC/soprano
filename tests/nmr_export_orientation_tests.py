#!/usr/bin/env python
"""
Orientation tests for the SIMPSON and mrsimulator writers.

The two programs take Euler angles in different (but equivalent for powder spectra)
conventions, so each writer is tested by rebuilding the lab-frame tensor from the written
angles with that program's own formula:

* SIMPSON (degrees): ``T = R diag(PAS) R^T`` with ``R = Rz(-gamma) Ry(-beta) Rz(-alpha)``
  (passive; ``lab_simpson`` in ``tests/nmr_test_helpers.py``).
* mrsimulator (radians): ``R = Rz(gamma) Ry(beta) Rz(alpha)`` as in
  ``mrsimulator.utils.cartesian_tensor`` (``_lab_mrsimulator``). In TensorView's terms this is the
  *active* ZYZ rotation with alpha and gamma interchanged; ``Rz``/``Ry`` are ordinary rotation
  matrices throughout.

Using lab-frame tensors with a non-zero relative azimuth and eta != 0 is what makes
these tests sensitive: tensors in the xz-plane, or axial ones with the unique axis along
x, cannot tell the senses apart.

Those formulas are what the writers are checked against, so they are themselves checked
against the programs, with stored output so that neither program is needed to run the tests:

* ``test_data/mrsimulator_reference.json``: tensors built by mrsimulator's own
  ``from_haeberlen_params`` for arbitrary angles (pins ``_lab_mrsimulator``), and a static
  13C CSA + 13C-1H dipolar powder spectrum simulated by mrsimulator's engine from the spin
  system Soprano writes;
* ``test_data/simpson_reference.json``: the same powder spectrum simulated by SIMPSON from the
  spin system Soprano writes.

Each stored spectrum is compared with an analytic powder spectrum computed directly from the
lab-frame tensors, which involves no Euler angles. Regenerate the files with
``tests/reference_generators/simulators.py``.

Powder spectra cannot tell SIMPSON's frame from its mirror image (a pi rotation about x), so
``TestSimpsonSingleCrystal`` also checks the absolute frame with single-crystal simulations:
the field direction in the written frame for a crystallite ``(alpha, beta)`` is
``Rz(alpha) Ry(beta) z``, and the two lines of a CSA + dipolar pair are ``larmor * n.delta.n
+- n.D.n / 2`` for the lab-frame tensors Soprano started from. The same is done with an anisotropic
J coupling in place of the dipole (``j_coupling_case``). The stored lines are in
``simpson_reference.json``; the live tests run SIMPSON itself if it is on the PATH (set
``SOPRANO_REQUIRE_SIMPSON`` to fail instead of skip; the CI job puts a ``simpson`` wrapper around
the ``jkshenton/simpson`` container image on the PATH).
"""

import json
import os
import shutil
import unittest
import warnings

import numpy as np

from soprano.data.nmr import EFG_TO_CHI, nmr_gamma, nmr_quadrupole
from soprano.nmr.coupling import DipolarCoupling, ISCoupling
from soprano.nmr.site import Site
from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor
from tests.nmr_test_helpers import (
    OFF_PLANE_AXIS,
    SPECTRUM_LB_HZ,
    SPECTRUM_PROTON_HZ,
    TEST_DATA,
    Ry,
    Rz,
    haeberlen_evals,
    j_coupling_case,
    lab_simpson,
    mrs_ct,
    simpson_single_crystal_input,
    simpson_single_crystal_lines,
    simpson_spectrum,
    spectrum_case,
)


def _lab_mrsimulator(evals, alpha, beta, gamma):
    """T_lab = E diag(evals) E^T, E = Rz(gamma) Ry(beta) Rz(alpha) (mrsimulator's code)."""
    E = Rz(gamma) @ Ry(beta) @ Rz(alpha)
    return E @ np.diag(evals) @ E.T


def _simpson_line(text, keyword):
    text = text if isinstance(text, str) else "\n".join(map(str, text))
    line = next(l for l in text.split("\n") if l.startswith(keyword))
    return line.split()


def _angles(d):
    return d["alpha"], d["beta"], d["gamma"]


def _axial(n, iso, zeta):
    """Axial tensor (eta = 0) with unique axis n and Haeberlen anisotropy zeta."""
    return iso * np.eye(3) + 1.5 * zeta * (np.outer(n, n) - np.eye(3) / 3)


# ---- the CSA + dipolar powder used for the stored simulator spectra (spectrum_case) ----

def carbon_larmor_hz():
    return SPECTRUM_PROTON_HZ * nmr_gamma("C", iso=13) / nmr_gamma("H", iso=1)


def analytic_powder(shift_ppm, D, freqs, larmor_hz, lb_hz=SPECTRUM_LB_HZ, n_dirs=400_000):
    """Static powder spectrum of a CSA weakly coupled to a heteronucleus, from lab-frame tensors.

    For a field direction n in the molecular frame the observed frequencies are
    ``larmor * 1e-6 * n.delta.n +- n.D.n / 2`` (D = d (3 r r^T - 1)), each with half the intensity.
    Directions are a Fibonacci grid on the sphere; the histogram on ``freqs`` (uniform) is convolved
    with a Lorentzian of FWHM ``lb_hz`` and normalised to unit area.
    """
    k = np.arange(n_dirs) + 0.5
    z = 1 - 2 * k / n_dirs
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z ** 2)
    n = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    shift = larmor_hz * 1e-6 * np.einsum("ni,ij,nj->n", n, shift_ppm, n)
    dip = 0.5 * np.einsum("ni,ij,nj->n", n, D, n)
    step = freqs[1] - freqs[0]
    edges = np.append(freqs - step / 2, freqs[-1] + step / 2)
    hist = np.histogram(shift + dip, edges)[0] + np.histogram(shift - dip, edges)[0]
    offsets = (np.arange(len(freqs)) - len(freqs) // 2) * step
    kernel = (lb_hz / 2) / np.pi / (offsets ** 2 + (lb_hz / 2) ** 2)
    spec = np.convolve(hist, kernel, mode="same")
    return spec / (spec.sum() * step)


def normalised(freqs, spec):
    """Baseline (median of the outer 5% on each side) removed, then scaled to unit area.

    The baseline absorbs the constant offset of a FFT whose first FID point is not halved
    (SIMPSON) and the Lorentzian tails folded back by a FFT-based apodisation (mrsimulator).
    """
    edge = max(1, len(spec) // 20)
    spec = np.asarray(spec, float) - np.median(np.concatenate([spec[:edge], spec[-edge:]]))
    return spec / (np.sum(spec) * (freqs[1] - freqs[0]))


def spectrum_mismatch(freqs, simulated, reference):
    """Largest deviation between two spectra after ``normalised``, relative to the reference maximum."""
    simulated, reference = normalised(freqs, simulated), normalised(freqs, reference)
    return float(np.max(np.abs(simulated - reference)) / np.max(reference))


class TestWriterOrientation(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_dipolar_angles_rebuild_the_tensor(self):
        d = -23328.153171
        D = d * (3 * np.outer(OFF_PLANE_AXIS, OFF_PLANE_AXIS) - np.eye(3))
        coupling = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                                   tensor=NMRTensor(D, order="n"))

        a, b, g = np.radians([float(x) for x in coupling.to_simpson().split()[4:7]])
        np.testing.assert_allclose(lab_simpson([-d, -d, 2 * d], a, b, g), D, atol=1e-6 * abs(d))

        mrs = coupling.to_mrsimulator()["dipolar"]
        np.testing.assert_allclose(_lab_mrsimulator([-d, -d, 2 * d], *_angles(mrs)), D, atol=1e-6 * abs(d))

    def test_shielding_angles_rebuild_the_tensor(self):
        sigma_evals = haeberlen_evals(50.0, -80.0, 0.6)
        sigma = lab_simpson(sigma_evals, *np.radians([30.0, 50.0, 70.0]))
        site = Site(isotope="13C", label="C1", index=0,
                    ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))

        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson()[0], "shift")[5:8]])
        # SIMPSON gets the shift tensor, whose eigenvalues are -sigma's
        np.testing.assert_allclose(lab_simpson(-sigma_evals, a, b, g), -sigma, atol=1e-9)

        mrs = site.to_mrsimulator()["shielding_symmetric"]
        np.testing.assert_allclose(
            _lab_mrsimulator(haeberlen_evals(50.0, mrs["zeta"], mrs["eta"]), *_angles(mrs)), sigma, atol=1e-9)

    def test_efg_angles_rebuild_the_tensor(self):
        vzz = 20000.0 / (EFG_TO_CHI * nmr_quadrupole("H", iso=2))
        evals = haeberlen_evals(0.0, vzz, 0.3)
        efg = lab_simpson(evals, *np.radians([70.0, 40.0, 15.0]))
        site = Site(isotope="2H", label="H1", index=0, efg=ElectricFieldGradient(efg, species="2H"))

        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson(q_order=1)[1], "quadrupole")[5:8]])
        np.testing.assert_allclose(lab_simpson(evals, a, b, g), efg, atol=1e-9 * vzz)

        q = site.to_mrsimulator()["quadrupolar"]
        np.testing.assert_allclose(_lab_mrsimulator(evals, *_angles(q)), efg, atol=1e-9 * vzz)

    def test_writers_do_not_change_the_site_tensors(self):
        vzz = 20000.0 / (EFG_TO_CHI * nmr_quadrupole("H", iso=2))
        efg = lab_simpson(haeberlen_evals(0.0, vzz, 0.3), *np.radians([70.0, 40.0, 15.0]))
        ms = lab_simpson([10.0, 20.0, 50.0], *np.radians([30.0, 50.0, 20.0]))
        site = Site(isotope="2H", label="H1", index=0,
                    ms=MagneticShielding(ms, species="2H", reference=0.0, gradient=-1.0),
                    efg=ElectricFieldGradient(efg, species="2H"))
        before = (site.ms.order, site.efg.order, site.ms.data, site.efg.data)
        site.to_mrsimulator()
        site.to_simpson(q_order=1)
        self.assertEqual((site.ms.order, site.efg.order), before[:2])
        np.testing.assert_array_equal(site.ms.data, before[2])
        np.testing.assert_array_equal(site.efg.data, before[3])

    def test_axial_tensors_keep_their_azimuth(self):
        # Axial tensors have a free angle; it must be the one that does not move the
        # unique axis. Off-plane axes (azimuth != 0) expose the wrong choice.
        sigma = _axial(OFF_PLANE_AXIS, 50.0, -80.0)
        site = Site(isotope="13C", label="C1", index=0,
                    ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))
        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson()[0], "shift")[5:8]])
        np.testing.assert_allclose(lab_simpson(-haeberlen_evals(50.0, -80.0, 0.0), a, b, g), -sigma, atol=1e-9)
        mrs = site.to_mrsimulator()["shielding_symmetric"]
        np.testing.assert_allclose(
            _lab_mrsimulator(haeberlen_evals(50.0, mrs["zeta"], 0.0), *_angles(mrs)), sigma, atol=1e-9)

        vzz = 20000.0 / (EFG_TO_CHI * nmr_quadrupole("H", iso=2))
        efg = _axial(OFF_PLANE_AXIS, 0.0, vzz)
        site = Site(isotope="2H", label="H1", index=0, efg=ElectricFieldGradient(efg, species="2H"))
        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson(q_order=1)[1], "quadrupole")[5:8]])
        np.testing.assert_allclose(lab_simpson(haeberlen_evals(0.0, vzz, 0.0), a, b, g), efg, atol=1e-9 * vzz)
        q = site.to_mrsimulator()["quadrupolar"]
        np.testing.assert_allclose(_lab_mrsimulator(haeberlen_evals(0.0, vzz, 0.0), *_angles(q)), efg, atol=1e-9 * vzz)

    def test_j_coupling_angles_rebuild_the_tensor(self):
        from soprano.data.nmr import nmr_gamma
        from soprano.nmr.utils import _J_constant
        scale = _J_constant(1.0, nmr_gamma("C", iso=13), nmr_gamma("H", iso=1))

        non_axial = lab_simpson(haeberlen_evals(150.0, 400.0, 0.5), *np.radians([20.0, 60.0, 100.0]))
        axial = _axial(OFF_PLANE_AXIS, 150.0, 400.0)

        for J_lab in (non_axial, axial):
            coupling = ISCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                                  tensor=NMRTensor(J_lab / scale, order="h"))
            evals = haeberlen_evals(coupling.Jisotropy, coupling.J_reduced_anisotropy, coupling.J_asymmetry)

            a, b, g = np.radians([float(x) for x in coupling.to_simpson().split()[6:9]])
            np.testing.assert_allclose(lab_simpson(evals, a, b, g), J_lab, atol=1e-9 * 400.0)

            mrs = coupling.to_mrsimulator()["j_symmetric"]
            np.testing.assert_allclose(_lab_mrsimulator(evals, *_angles(mrs)), J_lab, atol=1e-9 * 400.0)

    def test_j_coupling_angles_do_not_depend_on_tensor_order(self):
        # zeta and eta are always Haeberlen, so the angles must be too, whatever order the tensor
        # was built with; a negative gamma product (15N) flips K's eigenvalue order relative to J's
        from soprano.nmr.utils import _J_constant

        non_axial = lab_simpson(haeberlen_evals(150.0, 400.0, 0.5), *np.radians([20.0, 60.0, 100.0]))
        axial = _axial(OFF_PLANE_AXIS, 150.0, 400.0)
        for species2, element, iso in (("13C", "C", 13), ("15N", "N", 15)):
            scale = _J_constant(1.0, nmr_gamma("H", iso=1), nmr_gamma(element, iso=iso))
            for name, J_lab in (("non-axial", non_axial), ("axial", axial)):
                for order in ("h", "i", "d", "n"):
                    with self.subTest(species2=species2, tensor=name, order=order):
                        coupling = ISCoupling(site_i=0, site_j=1, species1="1H", species2=species2,
                                              tensor=NMRTensor(J_lab / scale, order=order))
                        evals = haeberlen_evals(coupling.Jisotropy, coupling.J_reduced_anisotropy,
                                                coupling.J_asymmetry)

                        a, b, g = np.radians([float(x) for x in coupling.to_simpson().split()[6:9]])
                        np.testing.assert_allclose(lab_simpson(evals, a, b, g), J_lab, atol=1e-9 * 400.0)

                        mrs = coupling.to_mrsimulator()["j_symmetric"]
                        np.testing.assert_allclose(_lab_mrsimulator(evals, *_angles(mrs)), J_lab,
                                                   atol=1e-9 * 400.0)
                        self.assertEqual(coupling.tensor.order, order)  # the coupling is not altered

    def test_dipolar_output_does_not_depend_on_tensor_order(self):
        # the coupling constant and angles must not depend on the order the tensor was built
        # with; d < 0 puts the unique eigenvalue first in increasing order
        for d in (-23328.153171, 4000.0):
            D = d * (3 * np.outer(OFF_PLANE_AXIS, OFF_PLANE_AXIS) - np.eye(3))
            for order in ("n", "i", "d", "h"):
                with self.subTest(d=d, order=order):
                    coupling = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                                               tensor=NMRTensor(D, order=order))
                    self.assertAlmostEqual(coupling.coupling_constant, d, delta=1e-9 * abs(d))

                    a, b, g = np.radians([float(x) for x in coupling.to_simpson().split()[4:7]])
                    np.testing.assert_allclose(lab_simpson([-d, -d, 2 * d], a, b, g), D, atol=1e-6 * abs(d))

                    mrs = coupling.to_mrsimulator()["dipolar"]
                    self.assertAlmostEqual(mrs["D"], d, delta=1e-9 * abs(d))
                    np.testing.assert_allclose(_lab_mrsimulator([-d, -d, 2 * d], *_angles(mrs)), D,
                                               atol=1e-6 * abs(d))
                    self.assertEqual(coupling.tensor.order, order)

    def test_reading_simpson_angles_the_mrsimulator_way_rotates_the_tensor_about_x(self):
        # The two formulas differ by a mirror conjugation, which for a symmetric tensor is a pi
        # rotation about x. Powder spectra cannot see it, but it is why the numbers differ and
        # why the SIMPSON angles must not be passed to mrsimulator.
        flip_x = np.diag([-1.0, 1.0, 1.0])
        for T in (lab_simpson(haeberlen_evals(0.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0])),
                  _axial(OFF_PLANE_AXIS, 0.0, 60.0)):
            tensor = NMRTensor(T, order="h")
            evals = tensor.eigenvalues
            simpson_angles = tensor.euler_angles(passive=True)
            mrs_angles = tensor.mrsimulator_euler_angles()
            np.testing.assert_allclose(lab_simpson(evals, *simpson_angles), T, atol=1e-9)
            np.testing.assert_allclose(_lab_mrsimulator(evals, *mrs_angles), T, atol=1e-9)
            np.testing.assert_allclose(_lab_mrsimulator(evals, *simpson_angles), flip_x @ T @ flip_x, atol=1e-9)

    @unittest.skipIf(mrs_ct is None, "mrsimulator is not available")
    def test_angles_agree_with_the_mrsimulator_helpers(self):
        # mrsimulator's own forward function is the reference for what its angles mean
        sigma = lab_simpson(haeberlen_evals(50.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0]))
        t = NMRTensor(sigma, order="h")
        zeta, eta = t.reduced_anisotropy, t.asymmetry
        rebuilt = mrs_ct.from_haeberlen_params(
            euler_angles=list(t.mrsimulator_euler_angles()), zeta_sigma=zeta, eta_sigma=eta,
            isotropic_component=np.trace(sigma) / 3)
        np.testing.assert_allclose(rebuilt, sigma, atol=1e-9)

        # the helper's own angles for the same tensor describe the same orientation
        helper = mrs_ct.to_symmetric_tensor(tensor=sigma, type="shielding")
        np.testing.assert_allclose(
            mrs_ct.from_haeberlen_params([helper.alpha, helper.beta, helper.gamma], helper.zeta, helper.eta,
                                         np.trace(sigma) / 3), sigma, atol=1e-9)

        d = -23328.153171
        D = d * (3 * np.outer(OFF_PLANE_AXIS, OFF_PLANE_AXIS) - np.eye(3))
        dip = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                              tensor=NMRTensor(D, order="n"))
        rebuilt = mrs_ct.from_haeberlen_params(
            euler_angles=[dip.to_mrsimulator()["dipolar"][k] for k in ("alpha", "beta", "gamma")],
            zeta_sigma=2 * d, eta_sigma=0.0, isotropic_component=0.0)
        np.testing.assert_allclose(rebuilt, D, atol=1e-6 * abs(d))

    def test_mrsimulator_shielding_angles_do_not_depend_on_tensor_order(self):
        sigma = lab_simpson(haeberlen_evals(50.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0]))
        written = {}
        for order in ("h", "i", "d"):
            ms = MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0, order=order)
            ss = Site(isotope="13C", label="C1", index=0, ms=ms).to_mrsimulator()["shielding_symmetric"]
            written[order] = [ss["zeta"], ss["eta"], ss["alpha"], ss["beta"], ss["gamma"]]
        np.testing.assert_allclose(written["i"], written["h"], atol=1e-9)
        np.testing.assert_allclose(written["d"], written["h"], atol=1e-9)

    def test_mrsimulator_anisotropy_follows_the_referencing_gradient(self):
        # sigma_iso = 80, reduced shielding anisotropy = +60
        for gradient, expected in [(-1.0, 60.0), (-0.9, 54.0)]:
            ms = MagneticShielding(np.diag([40.0, 60.0, 140.0]), species="13C",
                                   reference=170.0, gradient=gradient)
            zeta = Site(isotope="13C", label="C", index=0, ms=ms).to_mrsimulator()["shielding_symmetric"]["zeta"]
            self.assertAlmostEqual(zeta, expected, places=6)

    def test_documented_dipole_line(self):
        # docs/simpson-conventions.md: a bond at polar 70 deg, azimuth 40 deg is written `0 70 140` for SIMPSON
        # (azimuth 180 - gamma) and (0, 70 deg, 40 deg) in radians for mrsimulator
        coupling = spectrum_case()[4]
        self.assertEqual(coupling.to_simpson().split()[4:7], ["0.000000", "70.000000", "140.000000"])
        mrs = coupling.to_mrsimulator()["dipolar"]
        np.testing.assert_allclose(_angles(mrs), np.radians([0.0, 70.0, 40.0]), atol=1e-12)


def _simpson_lines(text):
    """SIMPSON shift and dipole lines -> (shift tensor in ppm, dipolar tensor in Hz), read the SIMPSON way."""
    shift, dipole = (next(line.split() for line in text.splitlines() if line.strip().startswith(k)) for k in ("shift", "dipole"))
    iso, aniso, eta = float(shift[2].rstrip("p")), float(shift[3].rstrip("p")), float(shift[4])
    d = float(dipole[3])
    return ((iso, aniso, eta, np.radians([float(x) for x in shift[5:8]])),
            (d, np.radians([float(x) for x in dipole[4:7]])))


def _tensors_from_simpson(text, frame):
    (iso, aniso, eta, ms_angles), (d, d_angles) = _simpson_lines(text)
    Es, Ed = frame(*ms_angles), frame(*d_angles)
    return Es @ np.diag(haeberlen_evals(iso, aniso, eta)) @ Es.T, Ed @ np.diag([-d, -d, 2 * d]) @ Ed.T


def _tensors_from_mrsimulator(spin_system, frame):
    """Shift tensor (ppm) and dipolar tensor (Hz); mrsimulator's zeta is the shielding anisotropy."""
    carbon, coupling = spin_system["sites"][0], spin_system["couplings"][0]["dipolar"]
    ms = carbon["shielding_symmetric"]
    Es, Ed = frame(*_angles(ms)), frame(*_angles(coupling))
    shielding = Es @ np.diag(haeberlen_evals(0.0, ms["zeta"], ms["eta"])) @ Es.T
    d = coupling["D"]
    return carbon["isotropic_chemical_shift"] * np.eye(3) - shielding, Ed @ np.diag([-d, -d, 2 * d]) @ Ed.T


def _simpson_frame(a, b, g):
    return Rz(-g) @ Ry(-b) @ Rz(-a)


def _mrsimulator_frame(a, b, g):
    return Rz(g) @ Ry(b) @ Rz(a)


def _active_frame(a, b, g):
    """TensorView's active ZYZ, i.e. Appendix B of the mrsimulator paper: the wrong reading for both."""
    return Rz(a) @ Ry(b) @ Rz(g)


class TestAgainstSimulators(unittest.TestCase):
    """The formulas above against stored output of mrsimulator and SIMPSON (see the module docstring)."""

    MATCH, MISMATCH = 0.06, 0.15  # spectrum_mismatch of the right and of a wrong reading (~0.02 and ~0.25)

    @classmethod
    def setUpClass(cls):
        cls.mrs = json.loads((TEST_DATA / "mrsimulator_reference.json").read_text())
        cls.simpson = json.loads((TEST_DATA / "simpson_reference.json").read_text())

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_lab_mrsimulator_is_mrsimulators_from_haeberlen_params(self):
        for case in self.mrs["forward"]:
            evals = haeberlen_evals(case["iso"], case["zeta"], case["eta"])
            expected = np.array(case["tensor"])
            np.testing.assert_allclose(_lab_mrsimulator(evals, *case["angles"]), expected, atol=1e-9)
            E = _active_frame(*case["angles"])
            self.assertGreater(np.max(np.abs(E @ np.diag(evals) @ E.T - expected)), 1e-3)

    def _check_spectrum(self, data, rebuild, right, wrong, larmor):
        freqs, stored = np.array(data["freqs_hz"]), np.array(data["amplitude"])
        sigma, D = spectrum_case()[:2]
        exact = analytic_powder(-sigma, D, freqs, larmor)
        self.assertLess(spectrum_mismatch(freqs, stored, exact), self.MATCH)
        # the numbers that were simulated, read with the program's formula, are the lab tensors ...
        shift, dip = rebuild(right)
        np.testing.assert_allclose(shift, -sigma, atol=1e-6)
        np.testing.assert_allclose(dip, D, atol=1e-6)
        # ... and read the active (TensorView / Appendix B) way they give a spectrum the program did not produce
        self.assertGreater(spectrum_mismatch(freqs, stored, analytic_powder(*rebuild(wrong), freqs, larmor)),
                           self.MISMATCH)

    def test_mrsimulator_spectrum_of_the_written_spin_system(self):
        data = self.mrs["spectrum"]
        self._check_spectrum(data, lambda frame: _tensors_from_mrsimulator(data["spin_system"], frame),
                             _mrsimulator_frame, _active_frame, data["larmor_hz"])

    def test_simpson_spectrum_of_the_written_spin_system(self):
        data = self.simpson["spectrum"]
        self._check_spectrum(data, lambda frame: _tensors_from_simpson(data["input"], frame),
                             _simpson_frame, _active_frame, carbon_larmor_hz())

    def test_powder_spectra_cannot_tell_simpson_from_mrsimulator(self):
        # reading SIMPSON's numbers with mrsimulator's formula mirrors every tensor (a pi rotation about x),
        # which leaves the powder spectrum unchanged, so these spectra pin passive vs active but not the mirror
        data = self.simpson["spectrum"]
        freqs, stored = np.array(data["freqs_hz"]), np.array(data["amplitude"])
        mirrored = _tensors_from_simpson(data["input"], _mrsimulator_frame)
        self.assertLess(spectrum_mismatch(freqs, stored, analytic_powder(*mirrored, freqs, carbon_larmor_hz())),
                        self.MATCH)

    def test_the_writers_still_write_the_simulated_orientations(self):
        # the stored spectra were simulated from the writers' output; today's output must describe the same tensors
        _, _, carbon, proton, coupling = spectrum_case()
        text = "\n".join([carbon.to_simpson()[0], coupling.to_simpson()])
        for now, then in zip(_tensors_from_simpson(text, _simpson_frame),
                             _tensors_from_simpson(self.simpson["spectrum"]["input"], _simpson_frame)):
            np.testing.assert_allclose(now, then, atol=1e-6)
        spin_system = {"sites": [carbon.to_mrsimulator(), proton.to_mrsimulator()],
                       "couplings": [{"site_index": [0, 1], **coupling.to_mrsimulator()}]}
        for now, then in zip(_tensors_from_mrsimulator(spin_system, _mrsimulator_frame),
                             _tensors_from_mrsimulator(self.mrs["spectrum"]["spin_system"], _mrsimulator_frame)):
            np.testing.assert_allclose(now, then, atol=1e-6)


# ---- single-crystal SIMPSON runs (the runner is in tests/nmr_test_helpers.py) ----

SINGLE_CRYSTAL_TOL_HZ = 2.0  # frequency grid of the simulation is 0.6 Hz


def simpson_available():
    return shutil.which("simpson") is not None


def field_direction(alpha, beta, mirror=False):
    """Field direction in the written frame for SIMPSON crystallite (alpha, beta) in degrees."""
    a, b = np.radians(alpha), np.radians(beta)
    return np.array([np.sin(b) * np.cos(a), (-1 if mirror else 1) * np.sin(b) * np.sin(a), np.cos(b)])


def single_crystal_lines(shift_ppm, D, larmor_hz, n):
    """The two lines (Hz, sorted) of a CSA coupled to a heteronucleus for field direction n."""
    centre = larmor_hz * 1e-6 * n @ shift_ppm @ n
    half_splitting = 0.5 * n @ D @ n
    return sorted([centre + half_splitting, centre - half_splitting])


class TestSimpsonSingleCrystal(unittest.TestCase):
    """Absolute frame of the SIMPSON writers: single crystallites, where the powder average cannot hide it."""

    @classmethod
    def setUpClass(cls):
        stored = json.loads((TEST_DATA / "simpson_reference.json").read_text())
        cls.stored, cls.stored_j = stored["single_crystal"], stored["single_crystal_j"]

    def setUp(self):
        warnings.simplefilter("ignore")
        self.sigma, self.D, self.carbon, _, self.coupling = spectrum_case()
        _, self.J, self.j_carbon, self.j_coupling = j_coupling_case()
        self.larmor = carbon_larmor_hz()

    def predicted(self, alpha, beta, mirror=False, coupling_hz=None):
        """Lines of the CSA with the dipole, or with the coupling tensor ``coupling_hz`` (e.g. J)."""
        coupling_hz = self.D if coupling_hz is None else coupling_hz
        return single_crystal_lines(-self.sigma, coupling_hz, self.larmor, field_direction(alpha, beta, mirror))

    def assert_same_input(self, stored_text, written_text):
        # The numbers on the interaction lines are compared numerically: the isotropic shift is zero
        # up to rounding noise that differs between BLAS/LAPACK builds (e.g. macOS and Linux).
        stored, written = stored_text.splitlines(), written_text.splitlines()
        self.assertEqual(len(stored), len(written))
        for stored_line, written_line in zip(stored, written):
            stored_tokens, written_tokens = stored_line.split(), written_line.split()
            if stored_tokens[:1] in (["shift"], ["dipole"], ["jcoupling"]):
                self.assertEqual(stored_tokens[:2], written_tokens[:2])
                np.testing.assert_allclose([float(t.rstrip("p")) for t in written_tokens[2:]],
                                           [float(t.rstrip("p")) for t in stored_tokens[2:]],
                                           rtol=1e-9, atol=1e-8)
            else:
                self.assertEqual(stored_line, written_line)

    def test_stored_lines_are_those_of_the_lab_frame_tensors(self):
        for (alpha, beta), lines in zip(self.stored["orientations_deg"], self.stored["lines_hz"]):
            np.testing.assert_allclose(self.predicted(alpha, beta), lines, atol=SINGLE_CRYSTAL_TOL_HZ)

    def test_stored_lines_exclude_the_mirrored_frame(self):
        # the mirror image (y -> -y) is what a powder spectrum cannot exclude
        errors = [np.max(np.abs(np.array(self.predicted(a, b, mirror=True)) - lines))
                  for (a, b), lines in zip(self.stored["orientations_deg"], self.stored["lines_hz"])]
        self.assertGreater(min(errors), 100 * SINGLE_CRYSTAL_TOL_HZ)

    def test_stored_input_is_what_the_writers_write_now(self):
        self.assert_same_input(self.stored["input"], simpson_single_crystal_input(self.carbon, self.coupling))

    # The same CSA with an anisotropic J coupling instead of the dipole. The splitting depends only
    # on J, so it fixes the frame of the jcoupling angles on its own.

    def test_stored_j_lines_are_those_of_the_lab_frame_tensors(self):
        for (alpha, beta), lines in zip(self.stored_j["orientations_deg"], self.stored_j["lines_hz"]):
            np.testing.assert_allclose(self.predicted(alpha, beta, coupling_hz=self.J), lines,
                                       atol=SINGLE_CRYSTAL_TOL_HZ)

    def test_stored_j_splittings_exclude_the_mirrored_frame(self):
        errors = []
        for (alpha, beta), lines in zip(self.stored_j["orientations_deg"], self.stored_j["lines_hz"]):
            mirrored = self.predicted(alpha, beta, mirror=True, coupling_hz=self.J)
            errors.append(abs((mirrored[1] - mirrored[0]) - (lines[1] - lines[0])))
        self.assertGreater(min(errors), 20 * SINGLE_CRYSTAL_TOL_HZ)

    def test_stored_j_input_is_what_the_writers_write_now(self):
        self.assert_same_input(self.stored_j["input"], simpson_single_crystal_input(self.j_carbon, self.j_coupling))

    @unittest.skipUnless(simpson_available() or os.environ.get("SOPRANO_REQUIRE_SIMPSON"), "simpson is not available")
    def test_simpson_single_crystal_live(self):
        text = simpson_single_crystal_input(self.carbon, self.coupling)
        for alpha, beta in [(70, 55), (300, 110)]:
            np.testing.assert_allclose(simpson_single_crystal_lines(text, alpha, beta),
                                       self.predicted(alpha, beta), atol=SINGLE_CRYSTAL_TOL_HZ)

    @unittest.skipUnless(simpson_available() or os.environ.get("SOPRANO_REQUIRE_SIMPSON"), "simpson is not available")
    def test_simpson_j_single_crystal_live(self):
        text = simpson_single_crystal_input(self.j_carbon, self.j_coupling)
        for alpha, beta in [(130, 80), (200, 40)]:
            np.testing.assert_allclose(simpson_single_crystal_lines(text, alpha, beta),
                                       self.predicted(alpha, beta, coupling_hz=self.J), atol=SINGLE_CRYSTAL_TOL_HZ)

    @unittest.skipUnless(simpson_available() or os.environ.get("SOPRANO_REQUIRE_SIMPSON"), "simpson is not available")
    def test_simpson_powder_spectrum_live(self):
        freqs, spec = simpson_spectrum()[:2]
        exact = analytic_powder(-self.sigma, self.D, freqs, self.larmor)
        self.assertLess(spectrum_mismatch(freqs, spec, exact), TestAgainstSimulators.MATCH)


if __name__ == "__main__":
    unittest.main()
