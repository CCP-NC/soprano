#!/usr/bin/env python3
"""
Verify Simpson simulation outputs against analytically expected peak positions.

Usage:
    python verify.py

Looks for `*_spe.dat` files in the same directory and compares the peak
positions to the expected values tabulated in EXPECTED below.
"""

import os
import glob
import numpy as np

# Tolerance for peak matching (Hz)
TOL_HZ = 500.0  # generous for broadening/FFT binning issues

# ---------------------------------------------------------------------------
# Expected peak frequencies in Hz for each test
# ---------------------------------------------------------------------------
# For 13C at 400 MHz proton frequency, the Larmor frequency is:
# nu_C = (gamma_C / gamma_H) * 400 MHz ≈ 100.60 MHz
# So 1 ppm for 13C ≈ 100.60 Hz
_NU_C_400MHZ = 100.601532  # MHz
_PPM_TO_HZ_13C = _NU_C_400MHZ  # Hz per ppm for 13C at 400 MHz proton freq

EXPECTED = {
    "test_08": {
        "description": "Quadrupole-dipole cross-term (13C-14N MAS) — spectrum must be broadened",
        "check": "width",
        "min_width_hz": 10.0,
        "min_sig_points": 5,
    },
    "test_01": {
        "description": "Isotropic shift +100 ppm @ 400 MHz (13C Larmor ~100.6 MHz)",
        "peaks_hz": [100 * _PPM_TO_HZ_13C],
        "tol_hz": 50.0,
    },
    "test_02a": {
        "description": "CSA β=0° → +200 ppm",
        "peaks_hz": [200 * _PPM_TO_HZ_13C],
        "tol_hz": 50.0,
    },
    "test_02b": {
        "description": "CSA β=90° → −100 ppm",
        "peaks_hz": [-100 * _PPM_TO_HZ_13C],
        "tol_hz": 50.0,
    },
    "test_02c": {
        "description": "CSA magic angle → 0 ppm",
        "peaks_hz": [0.0],
        "tol_hz": 50.0,
    },
    "test_03a": {
        "description": "CSA η=0.5, β=90°, γ=0° → −225 ppm",
        "peaks_hz": [-225 * _PPM_TO_HZ_13C],
        "tol_hz": 50.0,
    },
    "test_03b": {
        "description": "CSA η=0.5, β=90°, γ=90° → −75 ppm",
        "peaks_hz": [-75 * _PPM_TO_HZ_13C],
        "tol_hz": 50.0,
    },
    "test_04a": {
        "description": "Quad 2H Cq=100kHz, β=0° → ±75 kHz",
        "peaks_hz": [-75000.0, 75000.0],
        "tol_hz": 500.0,
    },
    "test_04b": {
        "description": "Quad 2H Cq=100kHz, β=90° → ±37.5 kHz",
        "peaks_hz": [-37500.0, 37500.0],
        "tol_hz": 500.0,
    },
    "test_05a": {
        "description": "Dipolar β=0° → splitting ≈ 60.5 kHz",
        # 13C doublet split by |2d| ≈ 60.5 kHz
        # Carrier is at 0 Hz for 13C (no shift), so peaks at ±30.25 kHz
        # UNLESS the 2π bug is present, in which case splitting ≈ 380 kHz
        "peaks_hz": [-30250.0, 30250.0],
        "tol_hz": 1000.0,
    },
    "test_05b": {
        "description": "Dipolar β=90° → splitting ≈ 30.25 kHz",
        "peaks_hz": [-15125.0, 15125.0],
        "tol_hz": 1000.0,
    },
    "test_06": {
        "description": "J-coupling isotropic 100 Hz",
        "peaks_hz": [-50.0, 50.0],
        "tol_hz": 5.0,
    },
    "test_07a": {
        "description": "J-aniso β=0° → splitting 200 Hz (if ζ used)",
        # Expected with correct ζ=200: peaks at ±100 Hz
        # If bug present (Δ=300): peaks at ±150 Hz
        "peaks_hz": [-100.0, 100.0],
        "tol_hz": 5.0,
    },
    "test_07b": {
        "description": "J-aniso β=90° → splitting 100 Hz (if ζ used)",
        "peaks_hz": [-50.0, 50.0],
        "tol_hz": 5.0,
    },
}


def load_spe(filename):
    """Load a Simpson .spe file (x re im format)."""
    data = np.loadtxt(filename)
    freq = data[:, 0]
    re = data[:, 1]
    im = data[:, 2]
    intensity = re + 1j * im
    return freq, np.abs(intensity)


def find_peaks(freq, intensity, n_peaks=1, threshold=0.1):
    """Find the n highest peaks in the spectrum."""
    # Simple peak finding: sort by intensity
    # For delta functions this is sufficient
    idx = np.argsort(intensity)[::-1]
    # Remove duplicates that are adjacent (same peak)
    unique_peaks = []
    for i in idx:
        if intensity[i] < threshold * intensity[idx[0]]:
            break
        if not unique_peaks or abs(freq[i] - unique_peaks[-1]) > 10.0:
            unique_peaks.append(freq[i])
        if len(unique_peaks) >= n_peaks:
            break
    return np.array(unique_peaks)


def check_test(name, expected_info):
    spe_file = os.path.join(os.path.dirname(__file__), f"{name}_spe.dat")
    if not os.path.exists(spe_file):
        print(f"  {name}: MISSING {spe_file}")
        return False

    freq, intensity = load_spe(spe_file)
    desc = expected_info["description"]

    # Width-based check (for cross-term broadening tests)
    if expected_info.get("check") == "width":
        threshold = 0.1 * intensity.max()
        sig = intensity > threshold
        n_sig = sig.sum()
        width = freq[sig].max() - freq[sig].min() if n_sig > 0 else 0.0

        min_width = expected_info["min_width_hz"]
        min_points = expected_info["min_sig_points"]

        if n_sig < min_points:
            print(f"  {name}: FAIL  ({desc})")
            print(f"         Expected ≥{min_points} significant points, found {n_sig}")
            return False
        if width < min_width:
            print(f"  {name}: FAIL  ({desc})")
            print(f"         Expected width ≥{min_width:.1f} Hz, found {width:.1f} Hz")
            return False

        print(f"  {name}: PASS  ({desc})")
        print(f"         Width: {width:.1f} Hz ({n_sig} points above threshold)")
        return True

    # Peak-position check (default for sharp-line tests)
    n_expected = len(expected_info["peaks_hz"])
    peaks = find_peaks(freq, intensity, n_peaks=n_expected)

    tol = expected_info["tol_hz"]
    expected = np.array(expected_info["peaks_hz"])

    if len(peaks) != len(expected):
        print(f"  {name}: FAIL  ({desc})")
        print(f"         Expected {len(expected)} peaks, found {len(peaks)}")
        print(f"         Found peaks at: {peaks}")
        return False

    # Match peaks to expected (allowing for sign ambiguity where physical)
    matched = []
    for p in peaks:
        dists = np.abs(expected - p)
        matched.append((dists.min(), dists.argmin()))

    max_err = max(m[0] for m in matched)
    if max_err > tol:
        print(f"  {name}: FAIL  ({desc})")
        print(f"         Expected: {expected}")
        print(f"         Found:    {peaks}")
        print(f"         Max error: {max_err:.1f} Hz (tol: {tol:.1f} Hz)")
        return False

    print(f"  {name}: PASS  ({desc})")
    print(f"         Peaks at: {peaks} Hz (expected: {expected} Hz)")
    return True


def main():
    print("=" * 70)
    print("Simpson Validation Results")
    print("=" * 70)

    all_pass = True
    for name in sorted(EXPECTED.keys()):
        if not check_test(name, EXPECTED[name]):
            all_pass = False
        print()

    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see details above")
        print()
        print("Common causes of failure:")
        print("  • Dipolar 2π bug: splitting ~6.3× too large (test_05)")
        print("  • J-aniso Δ-vs-ζ bug: splitting 1.5× too large (test_07)")
        print("  • Euler angle convention mismatch: peak at wrong frequency (test_02-04)")
    print("=" * 70)


if __name__ == "__main__":
    main()
