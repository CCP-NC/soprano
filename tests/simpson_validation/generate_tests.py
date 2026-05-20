#!/usr/bin/env python3
"""
Generate Simpson validation test cases for Soprano's SpinSystem.to_simpson().

Each test creates:
  - a .spinsys file (from Soprano)
  - a .in file (Simpson input script)

Tests are designed with analytically predictable single-crystal results
(crystal_file alpha0beta0) so peak positions can be verified unambiguously.
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from soprano.nmr.spin_system import SpinSystem
from soprano.nmr.site import Site
from soprano.nmr.tensor import MagneticShielding, ElectricFieldGradient
from soprano.nmr.coupling import DipolarCoupling, ISCoupling
from soprano.nmr.tensor import NMRTensor


OUTDIR = os.path.dirname(__file__)


def write_in_file(path, spinsys_file, par_extra, pulseq, main):
    """Write a Simpson .in file."""
    defaults = {
        "proton_frequency": "400e6",
        "start_operator": "I1x",
        "detect_operator": "I1p",
        "np": "8192",
        "sw": "500000",
        "num_cores": "1",
        "crystal_file": "alpha0beta0",
        "verbose": "0",
    }
    defaults.update(par_extra)

    lines = [f"source {spinsys_file}", ""]
    lines.append("par {")
    for k, v in defaults.items():
        lines.append(f"    {k} {v}")
    lines.append("}")
    lines.append("")
    lines.append("proc pulseq {} {")
    lines.append(pulseq)
    lines.append("}")
    lines.append("")
    lines.append("proc main {} {")
    lines.append(main)
    lines.append("}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_shielding_tensor(pas_evals, beta_deg=0, gamma_deg=0):
    """Create a shielding tensor in the lab frame from PAS eigenvalues + Euler angles.

    pas_evals: shielding eigenvalues [sigma_xx, sigma_yy, sigma_zz] in Haeberlen PAS
    beta_deg:  passive ZYZ beta angle (rotation about Y')
    gamma_deg: passive ZYZ gamma angle (rotation about Z'')
    """
    pas = np.diag(pas_evals)
    R = Rotation.from_euler("ZYZ", [0, beta_deg, gamma_deg], degrees=True).as_matrix()
    sigma_lab = R @ pas @ R.T
    return MagneticShielding(sigma_lab, species="13C", reference=0.0, gradient=-1.0)


def make_efg_tensor(pas_evals, beta_deg=0, gamma_deg=0):
    """Create an EFG tensor in the lab frame from PAS eigenvalues + Euler angles.

    pas_evals: EFG eigenvalues [V_xx, V_yy, V_zz] in Haeberlen PAS (traceless)
    """
    pas = np.diag(pas_evals)
    R = Rotation.from_euler("ZYZ", [0, beta_deg, gamma_deg], degrees=True).as_matrix()
    efg_lab = R @ pas @ R.T
    return ElectricFieldGradient(efg_lab, species="2H")


def make_j_tensor(J_iso, zeta, eta=0.0, beta_deg=0, gamma_deg=0):
    """Create a J-coupling tensor in the lab frame.

    ISCoupling expects the reduced K tensor (units of 1e19 T^2/J).
    We convert desired J values (Hz) to K values internally.
    """
    import scipy.constants as cnst
    from soprano.data.nmr import nmr_gamma

    gh = nmr_gamma("H", iso=1)
    gc = nmr_gamma("C", iso=13)
    # K -> J conversion factor from _J_constant
    scale = cnst.h * gh * gc / (4 * np.pi**2) * 1e19  # Hz per unit K

    K_iso = J_iso / scale
    zeta_K = zeta / scale

    evals = np.array([
        K_iso - 0.5 * zeta_K * (1 + eta),
        K_iso - 0.5 * zeta_K * (1 - eta),
        K_iso + zeta_K,
    ])
    pas = np.diag(evals)
    R = Rotation.from_euler("ZYZ", [0, beta_deg, gamma_deg], degrees=True).as_matrix()
    j_lab = R @ pas @ R.T
    return NMRTensor(j_lab, order="h")


# =============================================================================
# TEST 1: Isotropic Shift
# =============================================================================
def generate_test_01():
    """Single 13C with isotropic shift +100 ppm."""
    ms = make_shielding_tensor([-100, -100, -100], beta_deg=0)  # shielding = -shift
    site = Site(isotope="13C", label="C1", index=0, ms=ms)
    spin = SpinSystem(sites=[site])

    spinsys = spin.to_simpson(observed_nucleus="13C")
    with open(os.path.join(OUTDIR, "test_01.spinsys"), "w") as f:
        f.write(spinsys)

    write_in_file(
        os.path.join(OUTDIR, "test_01.in"),
        "test_01.spinsys",
        {"start_operator": "I1x", "detect_operator": "I1p"},
        pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
        main="""    global par
    set f [fsimpson]
    fsave $f test_01_fid.dat -xreim
    fft $f
    fsave $f test_01_spe.dat -xreim""",
    )


# =============================================================================
# TEST 2: CSA Anisotropy + Euler beta
# =============================================================================
def generate_test_02():
    """Single 13C with delta_iso=0, zeta=+200 ppm, eta=0 at three beta angles."""
    # Shielding PAS eigenvalues: sigma = -delta  =>  [100, 100, -200]
    pas = [100, 100, -200]
    for suffix, beta in [("a", 0), ("b", 90), ("c", 54.73561032)]:
        ms = make_shielding_tensor(pas, beta_deg=beta)
        site = Site(isotope="13C", label="C1", index=0, ms=ms)
        spin = SpinSystem(sites=[site])

        fname = f"test_02{suffix}"
        spinsys = spin.to_simpson(observed_nucleus="13C")
        with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
            f.write(spinsys)

        write_in_file(
            os.path.join(OUTDIR, f"{fname}.in"),
            f"{fname}.spinsys",
            {"start_operator": "I1x", "detect_operator": "I1p"},
            pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
            main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
        )


# =============================================================================
# TEST 3: CSA Asymmetry + Euler gamma
# =============================================================================
def generate_test_03():
    """Single 13C with delta_iso=0, zeta=+300 ppm, eta=0.5 at beta=90, two gammas."""
    # Shielding PAS: sigma_xx=225, sigma_yy=75, sigma_zz=-300
    pas = [225, 75, -300]
    for suffix, gamma in [("a", 0), ("b", 90)]:
        ms = make_shielding_tensor(pas, beta_deg=90, gamma_deg=gamma)
        site = Site(isotope="13C", label="C1", index=0, ms=ms)
        spin = SpinSystem(sites=[site])

        fname = f"test_03{suffix}"
        spinsys = spin.to_simpson(observed_nucleus="13C")
        with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
            f.write(spinsys)

        write_in_file(
            os.path.join(OUTDIR, f"{fname}.in"),
            f"{fname}.spinsys",
            {"start_operator": "I1x", "detect_operator": "I1p"},
            pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
            main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
        )


# =============================================================================
# TEST 4: Quadrupolar Coupling (first-order)
# =============================================================================
def generate_test_04():
    """Single 2H (I=1) with Cq=100 kHz, eta=0 at two beta angles."""
    # EFG PAS: traceless, V_zz chosen to give Cq = 100 kHz
    # Cq = EFG_TO_CHI * Q * V_zz  (Hz)
    # We need to compute V_zz for 2H to get Cq=100 kHz.
    from soprano.data.nmr import EFG_TO_CHI, nmr_quadrupole

    Q = nmr_quadrupole("H", iso=2)  # quadrupole moment in barns
    Vzz = 100e3 / (EFG_TO_CHI * Q)  # in atomic units

    pas = [-0.5 * Vzz, -0.5 * Vzz, Vzz]  # traceless, eta=0

    for suffix, beta in [("a", 0), ("b", 90)]:
        efg = make_efg_tensor(pas, beta_deg=beta)
        site = Site(isotope="2H", label="H1", index=0, efg=efg)
        spin = SpinSystem(sites=[site])

        fname = f"test_04{suffix}"
        spinsys = spin.to_simpson(observed_nucleus="2H", q_order=1)
        with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
            f.write(spinsys)

        write_in_file(
            os.path.join(OUTDIR, f"{fname}.in"),
            f"{fname}.spinsys",
            {
                "start_operator": "I1x",
                "detect_operator": "I1p",
                "sw": "200000",  # need large SW for 75 kHz peaks
            },
            pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
            main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
        )


# =============================================================================
# TEST 5: Dipolar Coupling
# =============================================================================
def generate_test_05():
    """1H-13C pair at 1.0 A, only dipolar coupling. Two orientations."""
    # Dipolar constant d for r=1.0 A
    from soprano.data.nmr import nmr_gamma
    from soprano.nmr.utils import _dip_constant

    gh = nmr_gamma("H", iso=1)
    gc = nmr_gamma("C", iso=13)
    d = _dip_constant(1.0e-10, gh, gc)  # Hz, negative

    site_h = Site(isotope="1H", label="H1", index=0)
    site_c = Site(isotope="13C", label="C1", index=1)

    for suffix, beta in [("a", 0), ("b", 90)]:
        # Build dipolar tensor directly in lab frame
        # beta=0  => unique axis along z
        # beta=90 => unique axis along x
        if beta == 0:
            D = d * np.diag([-1, -1, 2])
        else:
            D = d * np.diag([2, -1, -1])

        dip = DipolarCoupling(
            site_i=0, site_j=1,
            species1="1H", species2="13C",
            tensor=NMRTensor(D, order="n"),
        )

        spin = SpinSystem(sites=[site_h, site_c], couplings=[dip])

        fname = f"test_05{suffix}"
        spinsys = spin.to_simpson(observed_nucleus="13C")
        with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
            f.write(spinsys)

        write_in_file(
            os.path.join(OUTDIR, f"{fname}.in"),
            f"{fname}.spinsys",
            {
                "start_operator": "I2x",   # 13C is spin 2
                "detect_operator": "I2p",
                "sw": "200000",
            },
            pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
            main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
        )


# =============================================================================
# TEST 6: Isotropic J-Coupling
# =============================================================================
def generate_test_06():
    """1H-13C pair with J_iso = +100 Hz, no anisotropy."""
    site_h = Site(isotope="1H", label="H1", index=0)
    site_c = Site(isotope="13C", label="C1", index=1)

    jtensor = make_j_tensor(J_iso=100.0, zeta=0.0)
    jcoupling = ISCoupling(
        site_i=0, site_j=1,
        species1="1H", species2="13C",
        tensor=jtensor,
    )

    spin = SpinSystem(sites=[site_h, site_c], couplings=[jcoupling])

    fname = "test_06"
    spinsys = spin.to_simpson(observed_nucleus="13C")
    with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
        f.write(spinsys)

    write_in_file(
        os.path.join(OUTDIR, f"{fname}.in"),
        f"{fname}.spinsys",
        {
            "start_operator": "I2x",
            "detect_operator": "I2p",
            "sw": "2000",  # small SW for high resolution
        },
        pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
        main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
    )


# =============================================================================
# TEST 7: Anisotropic J-Coupling (reduced anisotropy test)
# =============================================================================
def generate_test_07():
    """1H-13C pair with J_iso=0, zeta=+200 Hz, eta=0. Two beta angles."""
    site_h = Site(isotope="1H", label="H1", index=0)
    site_c = Site(isotope="13C", label="C1", index=1)

    for suffix, beta in [("a", 0), ("b", 90)]:
        jtensor = make_j_tensor(J_iso=0.0, zeta=200.0, beta_deg=beta)
        jcoupling = ISCoupling(
            site_i=0, site_j=1,
            species1="1H", species2="13C",
            tensor=jtensor,
        )

        spin = SpinSystem(sites=[site_h, site_c], couplings=[jcoupling])

        fname = f"test_07{suffix}"
        spinsys = spin.to_simpson(observed_nucleus="13C")
        with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
            f.write(spinsys)

        write_in_file(
            os.path.join(OUTDIR, f"{fname}.in"),
            f"{fname}.spinsys",
            {
                "start_operator": "I2x",
                "detect_operator": "I2p",
                "sw": "2000",
            },
            pulseq="""    global par
    delay [expr 1e6/$par(sw)]
    store 1
    acq $par(np) 1""",
            main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
        )


# =============================================================================
# TEST 8: Quadrupole-Dipole Cross-Term (13C-14N under MAS)
# =============================================================================
def generate_test_08():
    """13C-14N pair with quadrupole on 14N and dipolar coupling.

    Under MAS the first-order dipolar and quadrupolar terms average to zero
    for 14N (I=1).  The only surviving effect on the 13C spectrum is the
    second-order quadrupole-dipole cross-term, which produces a broadened
    centreband.  This test verifies that to_simpson() emits
    ``quadrupole_x_dipole`` and that Simpson accepts it.
    """
    # EFG on 14N: Cq = 1.18 MHz, eta = 0.54, beta = 5°
    from soprano.data.nmr import EFG_TO_CHI, nmr_quadrupole
    Q = nmr_quadrupole("N", iso=14)
    Vzz = 1.18e6 / (EFG_TO_CHI * Q)
    Vxx = -0.5 * Vzz * (1 + 0.54)
    Vyy = -0.5 * Vzz * (1 - 0.54)
    efg_pas = np.diag([Vxx, Vyy, Vzz])
    R = Rotation.from_euler("ZYZ", [0, 5, 0], degrees=True).as_matrix()
    efg_lab = R @ efg_pas @ R.T
    efg = ElectricFieldGradient(efg_lab, species="14N")

    site_c = Site(isotope="13C", label="C1", index=0)
    site_n = Site(isotope="14N", label="N1", index=1, efg=efg)

    # Dipolar coupling d = -660.2 Hz (aligned with z-axis, beta=0)
    d = -660.2
    D = d * np.diag([-1, -1, 2])
    dip = DipolarCoupling(
        site_i=0, site_j=1,
        species1="13C", species2="14N",
        tensor=NMRTensor(D, order="n"),
    )

    spin = SpinSystem(sites=[site_c, site_n], couplings=[dip])

    fname = "test_08"
    spinsys = spin.to_simpson(observed_nucleus="13C")
    with open(os.path.join(OUTDIR, f"{fname}.spinsys"), "w") as f:
        f.write(spinsys)

    write_in_file(
        os.path.join(OUTDIR, f"{fname}.in"),
        f"{fname}.spinsys",
        {
            "start_operator": "I1x",
            "detect_operator": "I1p",
            "sw": "500",
            "np": "2048",
            "spin_rate": "12000",
            "variable tsw": "1e6/sw",
            "crystal_file": "rep30",
            "gamma_angles": "16",
        },
        pulseq="""    global par
    maxdt [expr 1.0e6/$par(spin_rate)/24.0]
    acq_block {
        delay $par(tsw)
    }""",
        main=f"""    global par
    set f [fsimpson]
    fsave $f {fname}_fid.dat -xreim
    fft $f
    fsave $f {fname}_spe.dat -xreim""",
    )


if __name__ == "__main__":
    generate_test_01()
    generate_test_02()
    generate_test_03()
    generate_test_04()
    generate_test_05()
    generate_test_06()
    generate_test_07()
    generate_test_08()
    print(f"Test files written to: {OUTDIR}")
