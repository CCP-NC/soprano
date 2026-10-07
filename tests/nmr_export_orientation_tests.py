#!/usr/bin/env python
"""
Orientation tests for the SIMPSON and mrsimulator writers.

The two programs take Euler angles in different (but equivalent for powder spectra)
conventions, so each writer is tested by rebuilding the lab-frame tensor from the written
angles with that program's own formula:

* SIMPSON (degrees): ``T = R diag(PAS) R^T`` with ``R = Rz(-gamma) Ry(-beta) Rz(-alpha)``
  (passive; the ``_lab_simpson`` helper below).
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
"""

import json
import os
import unittest
import warnings
from pathlib import Path

import numpy as np

from soprano.data.nmr import EFG_TO_CHI, nmr_gamma, nmr_quadrupole
from soprano.nmr.coupling import DipolarCoupling, ISCoupling
from soprano.nmr.site import Site
from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor

try:
    from mrsimulator.utils import cartesian_tensor as mrs_ct
except Exception:  # not installed, or not importable on this Python
    if os.environ.get("SOPRANO_REQUIRE_MRSIMULATOR"):  # set by the mrsimulator CI job
        raise
    mrs_ct = None


def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _lab_simpson(evals, alpha, beta, gamma):
    """T_lab = E diag(evals) E^T, E = Rz(-gamma) Ry(-beta) Rz(-alpha) (SIMPSON, passive)."""
    E = _Rz(-gamma) @ _Ry(-beta) @ _Rz(-alpha)
    return E @ np.diag(evals) @ E.T


def _lab_mrsimulator(evals, alpha, beta, gamma):
    """T_lab = E diag(evals) E^T, E = Rz(gamma) Ry(beta) Rz(alpha) (mrsimulator's code)."""
    E = _Rz(gamma) @ _Ry(beta) @ _Rz(alpha)
    return E @ np.diag(evals) @ E.T


def _haeberlen(iso, zeta, eta):
    return np.array([iso - 0.5 * zeta * (1 + eta), iso - 0.5 * zeta * (1 - eta), iso + zeta])


def _simpson_line(text, keyword):
    text = text if isinstance(text, str) else "\n".join(map(str, text))
    line = next(l for l in text.split("\n") if l.startswith(keyword))
    return line.split()


def _angles(d):
    return d["alpha"], d["beta"], d["gamma"]


def _axial(n, iso, zeta):
    """Axial tensor (eta = 0) with unique axis n and Haeberlen anisotropy zeta."""
    return iso * np.eye(3) + 1.5 * zeta * (np.outer(n, n) - np.eye(3) / 3)


# a direction off the xz-plane: polar 70 deg, azimuth 40 deg
_TH, _PH = np.radians(70), np.radians(40)
_N = np.array([np.sin(_TH) * np.cos(_PH), np.sin(_TH) * np.sin(_PH), np.cos(_TH)])


# ---- the CSA + dipolar powder used for the stored simulator spectra ----

TEST_DATA = Path(__file__).parent / "test_data"
SPECTRUM_PROTON_HZ = 400e6
SPECTRUM_LB_HZ = 200.0  # Lorentzian FWHM applied by the simulator and to the analytic spectrum


def spectrum_case():
    """13C CSA (zeta_sigma = -80 ppm, eta = 0.6, tilted) coupled to 1H (d = -4 kHz, bond off the xz-plane).

    The CSA and dipolar widths are comparable (8 and 12 kHz at 400 MHz), so the spectrum depends
    strongly on the relative orientation of the two tensors.
    """
    sigma = _lab_simpson(_haeberlen(0.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0]))
    D = -4000.0 * (3 * np.outer(_N, _N) - np.eye(3))
    carbon = Site(isotope="13C", label="C1", index=0,
                  ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))
    proton = Site(isotope="1H", label="H1", index=1)
    coupling = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H", tensor=NMRTensor(D, order="n"))
    return sigma, D, carbon, proton, coupling


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
        D = d * (3 * np.outer(_N, _N) - np.eye(3))
        coupling = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                                   tensor=NMRTensor(D, order="n"))

        a, b, g = np.radians([float(x) for x in coupling.to_simpson().split()[4:7]])
        np.testing.assert_allclose(_lab_simpson([-d, -d, 2 * d], a, b, g), D, atol=1e-6 * abs(d))

        mrs = coupling.to_mrsimulator()["dipolar"]
        np.testing.assert_allclose(_lab_mrsimulator([-d, -d, 2 * d], *_angles(mrs)), D, atol=1e-6 * abs(d))

    def test_shielding_angles_rebuild_the_tensor(self):
        sigma_evals = _haeberlen(50.0, -80.0, 0.6)
        sigma = _lab_simpson(sigma_evals, *np.radians([30.0, 50.0, 70.0]))
        site = Site(isotope="13C", label="C1", index=0,
                    ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))

        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson()[0], "shift")[5:8]])
        # SIMPSON gets the shift tensor, whose eigenvalues are -sigma's
        np.testing.assert_allclose(_lab_simpson(-sigma_evals, a, b, g), -sigma, atol=1e-9)

        mrs = site.to_mrsimulator()["shielding_symmetric"]
        np.testing.assert_allclose(
            _lab_mrsimulator(_haeberlen(50.0, mrs["zeta"], mrs["eta"]), *_angles(mrs)), sigma, atol=1e-9)

    def test_efg_angles_rebuild_the_tensor(self):
        vzz = 20000.0 / (EFG_TO_CHI * nmr_quadrupole("H", iso=2))
        evals = _haeberlen(0.0, vzz, 0.3)
        efg = _lab_simpson(evals, *np.radians([70.0, 40.0, 15.0]))
        site = Site(isotope="2H", label="H1", index=0, efg=ElectricFieldGradient(efg, species="2H"))

        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson(q_order=1)[1], "quadrupole")[5:8]])
        np.testing.assert_allclose(_lab_simpson(evals, a, b, g), efg, atol=1e-9 * vzz)

        q = site.to_mrsimulator()["quadrupolar"]
        np.testing.assert_allclose(_lab_mrsimulator(evals, *_angles(q)), efg, atol=1e-9 * vzz)

    def test_axial_tensors_keep_their_azimuth(self):
        # Axial tensors have a free angle; it must be the one that does not move the
        # unique axis. Off-plane axes (azimuth != 0) expose the wrong choice.
        sigma = _axial(_N, 50.0, -80.0)
        site = Site(isotope="13C", label="C1", index=0,
                    ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))
        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson()[0], "shift")[5:8]])
        np.testing.assert_allclose(_lab_simpson(-_haeberlen(50.0, -80.0, 0.0), a, b, g), -sigma, atol=1e-9)
        mrs = site.to_mrsimulator()["shielding_symmetric"]
        np.testing.assert_allclose(
            _lab_mrsimulator(_haeberlen(50.0, mrs["zeta"], 0.0), *_angles(mrs)), sigma, atol=1e-9)

        vzz = 20000.0 / (EFG_TO_CHI * nmr_quadrupole("H", iso=2))
        efg = _axial(_N, 0.0, vzz)
        site = Site(isotope="2H", label="H1", index=0, efg=ElectricFieldGradient(efg, species="2H"))
        a, b, g = np.radians([float(x) for x in _simpson_line(site.to_simpson(q_order=1)[1], "quadrupole")[5:8]])
        np.testing.assert_allclose(_lab_simpson(_haeberlen(0.0, vzz, 0.0), a, b, g), efg, atol=1e-9 * vzz)
        q = site.to_mrsimulator()["quadrupolar"]
        np.testing.assert_allclose(_lab_mrsimulator(_haeberlen(0.0, vzz, 0.0), *_angles(q)), efg, atol=1e-9 * vzz)

    def test_j_coupling_angles_rebuild_the_tensor(self):
        from soprano.data.nmr import nmr_gamma
        from soprano.nmr.utils import _J_constant
        scale = _J_constant(1.0, nmr_gamma("C", iso=13), nmr_gamma("H", iso=1))

        non_axial = _lab_simpson(_haeberlen(150.0, 400.0, 0.5), *np.radians([20.0, 60.0, 100.0]))
        axial = _axial(_N, 150.0, 400.0)

        for J_lab in (non_axial, axial):
            coupling = ISCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                                  tensor=NMRTensor(J_lab / scale, order="h"))
            evals = _haeberlen(coupling.Jisotropy, coupling.J_reduced_anisotropy, coupling.J_asymmetry)

            a, b, g = np.radians([float(x) for x in coupling.to_simpson().split()[6:9]])
            np.testing.assert_allclose(_lab_simpson(evals, a, b, g), J_lab, atol=1e-9 * 400.0)

            mrs = coupling.to_mrsimulator()["j_symmetric"]
            np.testing.assert_allclose(_lab_mrsimulator(evals, *_angles(mrs)), J_lab, atol=1e-9 * 400.0)

    def test_reading_simpson_angles_the_mrsimulator_way_rotates_the_tensor_about_x(self):
        # The two formulas differ by a mirror conjugation, which for a symmetric tensor is a pi
        # rotation about x. Powder spectra cannot see it, but it is why the numbers differ and
        # why the SIMPSON angles must not be passed to mrsimulator.
        flip_x = np.diag([-1.0, 1.0, 1.0])
        for T in (_lab_simpson(_haeberlen(0.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0])),
                  _axial(_N, 0.0, 60.0)):
            tensor = NMRTensor(T, order="h")
            evals = tensor.eigenvalues
            simpson_angles = tensor.euler_angles(passive=True)
            mrs_angles = tensor.mrsimulator_euler_angles()
            np.testing.assert_allclose(_lab_simpson(evals, *simpson_angles), T, atol=1e-9)
            np.testing.assert_allclose(_lab_mrsimulator(evals, *mrs_angles), T, atol=1e-9)
            np.testing.assert_allclose(_lab_mrsimulator(evals, *simpson_angles), flip_x @ T @ flip_x, atol=1e-9)

    @unittest.skipIf(mrs_ct is None, "mrsimulator is not available")
    def test_angles_agree_with_the_mrsimulator_helpers(self):
        # mrsimulator's own forward function is the reference for what its angles mean
        sigma = _lab_simpson(_haeberlen(50.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0]))
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
        D = d * (3 * np.outer(_N, _N) - np.eye(3))
        dip = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                              tensor=NMRTensor(D, order="n"))
        rebuilt = mrs_ct.from_haeberlen_params(
            euler_angles=[dip.to_mrsimulator()["dipolar"][k] for k in ("alpha", "beta", "gamma")],
            zeta_sigma=2 * d, eta_sigma=0.0, isotropic_component=0.0)
        np.testing.assert_allclose(rebuilt, D, atol=1e-6 * abs(d))

    def test_mrsimulator_shielding_angles_do_not_depend_on_tensor_order(self):
        sigma = _lab_simpson(_haeberlen(50.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0]))
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
                                   reference=0.0, gradient=gradient)
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
    return Es @ np.diag(_haeberlen(iso, aniso, eta)) @ Es.T, Ed @ np.diag([-d, -d, 2 * d]) @ Ed.T


def _tensors_from_mrsimulator(spin_system, frame):
    """Shift tensor (ppm) and dipolar tensor (Hz); mrsimulator's zeta is the shielding anisotropy."""
    carbon, coupling = spin_system["sites"][0], spin_system["couplings"][0]["dipolar"]
    ms = carbon["shielding_symmetric"]
    Es, Ed = frame(*_angles(ms)), frame(*_angles(coupling))
    shielding = Es @ np.diag(_haeberlen(0.0, ms["zeta"], ms["eta"])) @ Es.T
    d = coupling["D"]
    return carbon["isotropic_chemical_shift"] * np.eye(3) - shielding, Ed @ np.diag([-d, -d, 2 * d]) @ Ed.T


def _simpson_frame(a, b, g):
    return _Rz(-g) @ _Ry(-b) @ _Rz(-a)


def _mrsimulator_frame(a, b, g):
    return _Rz(g) @ _Ry(b) @ _Rz(a)


def _active_frame(a, b, g):
    """TensorView's active ZYZ, i.e. Appendix B of the mrsimulator paper: the wrong reading for both."""
    return _Rz(a) @ _Ry(b) @ _Rz(g)


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
            evals = _haeberlen(case["iso"], case["zeta"], case["eta"])
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


if __name__ == "__main__":
    unittest.main()
