"""
Helpers shared by the NMR orientation tests and the reference generators in
``tests/reference_generators``.

Nothing here uses Soprano's rotation code: the rotation matrices are written out explicitly.
The simulation case and the SIMPSON runners live here (not in a test module) so that the
generators and the tests both import them from one place.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from soprano.data.nmr import nmr_gamma
from soprano.nmr.coupling import DipolarCoupling, ISCoupling
from soprano.nmr.site import Site
from soprano.nmr.tensor import MagneticShielding, NMRTensor
from soprano.nmr.utils import _J_constant

try:  # mrsimulator currently installs only on Python <= 3.12
    from mrsimulator.utils import cartesian_tensor as mrs_ct
except Exception:  # not installed, or not importable on this Python
    if os.environ.get("SOPRANO_REQUIRE_MRSIMULATOR"):  # set by the mrsimulator CI job
        raise
    mrs_ct = None


# ---- rotations ----

def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 2] *= -1
    return q


# ---- tensors written the SIMPSON way ----

def lab_simpson(evals, alpha, beta, gamma):
    """T_lab = E diag(evals) E^T, E = Rz(-gamma) Ry(-beta) Rz(-alpha) (SIMPSON, passive)."""
    E = Rz(-gamma) @ Ry(-beta) @ Rz(-alpha)
    return E @ np.diag(evals) @ E.T


def haeberlen_evals(iso, zeta, eta):
    return np.array([iso - 0.5 * zeta * (1 + eta), iso - 0.5 * zeta * (1 - eta), iso + zeta])


# a direction off the xz-plane: polar 70 deg, azimuth 40 deg
_TH, _PH = np.radians(70), np.radians(40)
OFF_PLANE_AXIS = np.array([np.sin(_TH) * np.cos(_PH), np.sin(_TH) * np.sin(_PH), np.cos(_TH)])


# ---- the CSA + dipolar case used for the stored simulator output ----

TEST_DATA = Path(__file__).parent / "test_data"
SPECTRUM_PROTON_HZ = 400e6
SPECTRUM_LB_HZ = 200.0  # Lorentzian FWHM applied by the simulator and to the analytic spectrum


def spectrum_case():
    """13C CSA (zeta_sigma = -80 ppm, eta = 0.6, tilted) coupled to 1H (d = -4 kHz, bond off the xz-plane).

    The CSA and dipolar widths are comparable (8 and 12 kHz at 400 MHz), so the spectrum depends
    strongly on the relative orientation of the two tensors.
    """
    sigma = lab_simpson(haeberlen_evals(0.0, -80.0, 0.6), *np.radians([30.0, 50.0, 70.0]))
    D = -4000.0 * (3 * np.outer(OFF_PLANE_AXIS, OFF_PLANE_AXIS) - np.eye(3))
    carbon = Site(isotope="13C", label="C1", index=0,
                  ms=MagneticShielding(sigma, species="13C", reference=0.0, gradient=-1.0))
    proton = Site(isotope="1H", label="H1", index=1)
    coupling = DipolarCoupling(site_i=0, site_j=1, species1="13C", species2="1H", tensor=NMRTensor(D, order="n"))
    return sigma, D, carbon, proton, coupling


def j_coupling_case():
    """The 13C CSA of ``spectrum_case()`` with an anisotropic 13C-1H J coupling instead of the dipole.

    J (Hz, lab frame) has J_iso = 150 Hz, zeta = 400 Hz and eta = 0.5, with its PAS off the xz-plane.
    Returns ``(sigma, J, carbon, coupling)``.
    """
    sigma, _, carbon, _, _ = spectrum_case()
    J = lab_simpson(haeberlen_evals(150.0, 400.0, 0.5), *np.radians([20.0, 60.0, 100.0]))
    scale = _J_constant(1.0, nmr_gamma("C", iso=13), nmr_gamma("H", iso=1))
    coupling = ISCoupling(site_i=0, site_j=1, species1="13C", species2="1H",
                          tensor=NMRTensor(J / scale, order="h"))
    return sigma, J, carbon, coupling


# ---- SIMPSON: powder spectrum ----

SW_HZ = 100_000.0
NP = 4096

SIMPSON_INPUT = """\
spinsys {{
    channels 13C 1H
    nuclei 13C 1H
    {shift}
    {dipole}
}}

par {{
    proton_frequency {proton_hz}
    spin_rate 0
    start_operator I1x
    detect_operator I1p
    np {np}
    sw {sw}
    crystal_file zcw28656
    gamma_angles 1
    verbose 0
}}

proc pulseq {{}} {{
    global par
    acq_block {{
        delay [expr 1e6/$par(sw)]
    }}
}}

proc main {{}} {{
    global par
    set f [fsimpson]
    faddlb $f {lb} 0
    fft $f
    fsave $f spectrum.spe -xreim
}}
"""


def simpson_spectrum():
    """Run SIMPSON on the spin system Soprano writes for ``spectrum_case()``.

    Returns ``(freqs, amplitude, input_text, shift_line, dipole_line, version_line)``.
    """
    sigma, D, carbon, proton, coupling = spectrum_case()
    shift = carbon.to_simpson()[0]
    dipole = coupling.to_simpson()
    text = SIMPSON_INPUT.format(shift=shift, dipole=dipole, proton_hz=SPECTRUM_PROTON_HZ, np=NP, sw=SW_HZ,
                                lb=SPECTRUM_LB_HZ)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "spectrum.in").write_text(text)
        subprocess.run(["simpson", "spectrum.in"], cwd=tmp, check=True, capture_output=True)
        spe = np.loadtxt(Path(tmp) / "spectrum.spe")
    banner = subprocess.run(["simpson"], capture_output=True, text=True)
    version = next(line for line in (banner.stdout + banner.stderr).splitlines() if "version" in line).strip()
    return spe[:, 0], spe[:, 1], text, shift, dipole, version


# ---- SIMPSON: single crystallites ----

SINGLE_CRYSTAL_ORIENTATIONS = [(20, 30), (70, 55), (130, 80), (200, 40), (300, 110), (45, 65)]  # (alpha, beta), deg

SINGLE_CRYSTAL_INPUT = """\
spinsys {{
    channels 13C 1H
    nuclei 13C 1H
    {shift}
    {coupling}
}}

par {{
    proton_frequency {proton_hz}
    spin_rate 0
    start_operator I1x
    detect_operator I1p
    np 8192
    sw 80000
    crystal_file xtal
    gamma_angles 1
    verbose 0
}}

proc pulseq {{}} {{
    global par
    acq_block {{
        delay [expr 1e6/$par(sw)]
    }}
}}

proc main {{}} {{
    global par
    set f [fsimpson]
    fsave $f out.fid -xreim
}}
"""


def simpson_single_crystal_input(carbon, coupling):
    """SIMPSON input for the 13C site and one coupling (dipolar or J) to 1H, as Soprano writes them."""
    return SINGLE_CRYSTAL_INPUT.format(shift=carbon.to_simpson()[0], coupling=coupling.to_simpson(),
                                       proton_hz=SPECTRUM_PROTON_HZ)


def simpson_single_crystal_lines(text, alpha, beta):
    """Run SIMPSON for the single crystallite (alpha, beta) and return the two strongest lines (Hz, sorted)."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "t.in").write_text(text)
        (Path(tmp) / "xtal").write_text(f"1\n{alpha} {beta} 1\n")
        subprocess.run(["simpson", "t.in"], cwd=tmp, check=True, capture_output=True)
        fid = np.loadtxt(Path(tmp) / "out.fid")
    sw, nfft = 80000.0, 2 ** 17
    signal = fid[:, 1] + 1j * fid[:, 2]
    signal[0] *= 0.5
    spec = np.fft.fftshift(np.fft.fft(signal, nfft)).real
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sw))
    peaks = [i for i in range(1, nfft - 1)
             if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > 0.2 * spec.max()]
    strongest = sorted(peaks, key=lambda i: -spec[i])[:2]
    return sorted(freqs[strongest])
