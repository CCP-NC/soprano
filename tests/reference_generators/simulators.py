#!/usr/bin/env python
"""
Regenerate the stored simulator output used by ``tests/nmr_export_orientation_tests.py``.

* ``tests/test_data/mrsimulator_reference.json`` (needs ``mrsimulator``, which currently installs on
  Python <= 3.12): tensors from mrsimulator's ``cartesian_tensor.from_haeberlen_params`` for a set of
  arbitrary Euler angles, and the static powder spectrum of ``spectrum_case()`` simulated by
  mrsimulator from the spin system Soprano writes.
* ``tests/test_data/simpson_reference.json`` (needs ``simpson`` on the PATH): the same spectrum
  simulated by SIMPSON from the spin system Soprano writes.

Each part is skipped if its program is not available. Run from the repository root:

    python tests/reference_generators/simulators.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.nmr_export_orientation_tests import (  # noqa: E402
    SPECTRUM_LB_HZ,
    SPECTRUM_PROTON_HZ,
    TEST_DATA,
    spectrum_case,
)

SW_HZ = 100_000.0
NP = 4096


def _compact(x, decimals):
    """Spectra are stored scaled to a maximum of 1 and rounded, to keep the files small."""
    return np.round(np.asarray(x, float), decimals).tolist()


def _plain(obj):
    """numpy scalars -> floats, recursively (for pydantic and json)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def mrsimulator_reference():
    try:
        import mrsimulator
        from mrsimulator import Simulator, SpinSystem
        from mrsimulator import signal_processor as sp
        from mrsimulator.method.lib import BlochDecaySpectrum
        from mrsimulator.utils import cartesian_tensor as ct
    except ImportError:
        print("mrsimulator not importable: skipping mrsimulator_reference.json")
        return

    rng = np.random.default_rng(1001)
    forward = []
    for _ in range(12):
        angles = [float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, np.pi)), float(rng.uniform(0, 2 * np.pi))]
        zeta, eta, iso = float(rng.uniform(-100, 100)), float(rng.uniform(0, 1)), float(rng.uniform(-50, 50))
        tensor = ct.from_haeberlen_params(euler_angles=angles, zeta_sigma=zeta, eta_sigma=eta,
                                          isotropic_component=iso)
        forward.append({"angles": angles, "zeta": zeta, "eta": eta, "iso": iso, "tensor": np.asarray(tensor).tolist()})

    sigma, D, carbon, proton, coupling = spectrum_case()
    spin_system = _plain({"sites": [carbon.to_mrsimulator(), proton.to_mrsimulator()],
                          "couplings": [{"site_index": [0, 1], **coupling.to_mrsimulator()}]})
    # B0 that gives a 400 MHz proton frequency with mrsimulator's own gyromagnetic ratio
    from mrsimulator.spin_system.isotope import Isotope
    b0 = SPECTRUM_PROTON_HZ / (Isotope(symbol="1H").gyromagnetic_ratio * 1e6)
    method = BlochDecaySpectrum(
        channels=["13C"], magnetic_flux_density=b0, rotor_frequency=0, rotor_angle=0,
        spectral_dimensions=[{"count": NP, "spectral_width": SW_HZ, "reference_offset": 0}])
    sim = Simulator(spin_systems=[SpinSystem(**spin_system)], methods=[method])
    sim.config.integration_density = 200
    # the default "octant" assumes every tensor shares the symmetry of the first; not so here
    sim.config.integration_volume = "hemisphere"
    sim.run()
    processor = sp.SignalProcessor(operations=[
        sp.IFFT(), sp.apodization.Exponential(FWHM=f"{SPECTRUM_LB_HZ} Hz"), sp.FFT()])
    spectrum = processor.apply_operations(dataset=sim.methods[0].simulation).real
    # the coordinates are in ppm of the 13C Larmor frequency (origin_offset); positive shift = positive ppm
    larmor_hz = float(spectrum.x[0].origin_offset.to("Hz").value)
    freqs = np.asarray(spectrum.x[0].coordinates.value) * 1e-6 * larmor_hz
    amplitude = np.asarray(spectrum.y[0].components[0])
    data = {
        "description": "Output of mrsimulator; see tests/reference_generators/simulators.py",
        "mrsimulator": mrsimulator.__version__,
        "forward": forward,
        "spectrum": {"spin_system": spin_system, "magnetic_flux_density": b0,
                     "larmor_hz": larmor_hz,
                     "lb_hz": SPECTRUM_LB_HZ, "freqs_hz": _compact(freqs, 3),
                     "amplitude": _compact(amplitude / np.max(amplitude), 7)},
    }
    out = TEST_DATA / "mrsimulator_reference.json"
    out.write_text(json.dumps(data) + "\n")
    print(f"wrote {out}")


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


def simpson_reference():
    if shutil.which("simpson") is None:
        print("simpson not on the PATH: skipping simpson_reference.json")
        return
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
    data = {
        "description": "Output of SIMPSON; see tests/reference_generators/simulators.py",
        "simpson": version,
        "spectrum": {"input": text, "shift": shift, "dipole": dipole, "lb_hz": SPECTRUM_LB_HZ,
                     "freqs_hz": _compact(spe[:, 0], 3), "amplitude": _compact(spe[:, 1] / np.max(spe[:, 1]), 7)},
    }
    out = TEST_DATA / "simpson_reference.json"
    out.write_text(json.dumps(data) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    simpson_reference()
    mrsimulator_reference()
