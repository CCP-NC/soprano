#!/usr/bin/env python
"""
Regenerate ``tests/test_data/tensorview_reference.json`` by running TensorView for MATLAB.

TensorView (Svenningsson & Mueller, Solid State Nucl. Magn. Reson. 123 (2023) 101849) is the
reference for Soprano's Euler-angle conventions. This script downloads its ``Source/functions``
at a pinned commit, runs ``MFtoEuler`` (one tensor) and ``RelativeAngles`` (two tensors) on a
fixed set of tensors with GNU Octave, and stores the inputs and TensorView's outputs. The
tests in ``nmr_euler_convention_tests.py`` then compare Soprano against the stored numbers, so
they do not need Octave.

TensorView targets MATLAB. Two mechanical changes make it run in Octave, neither of which
touches the numerics:

* ``x == "Name"`` string comparisons become ``strcmp(x, "Name")`` (in Octave a double-quoted
  string is a char array, so ``==`` compares element-wise and fails for unequal lengths);
* ``round(x, n)`` (n decimal digits, missing in Octave) is provided by a small shim.

Requirements: ``octave-cli`` on the PATH and network access to GitHub. Run from anywhere:

    python tests/reference_generators/tensorview.py
"""

import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

TENSORVIEW_COMMIT = "f2f92b5a81f6a6081bf9338c778ad13e0eb6728a"
TENSORVIEW_URL = ("https://raw.githubusercontent.com/LeoSvenningsson/TensorViewforMatlab/"
                  f"{TENSORVIEW_COMMIT}/Source/functions/{{}}.m")
FUNCTIONS = ["CreateRotationMatrix", "MFtoEuler", "RelativeAngles", "RotateTensor", "tryallanglestest"]
MODES = ["AZYZ", "PZYZ", "AZXZ", "PZXZ"]
# Soprano order -> TensorView "PAS tensor order"
ORDERS = {"i": "Ascending", "d": "Descending", "n": "Absascending"}
OUTPUT = Path(__file__).resolve().parents[1] / "test_data" / "tensorview_reference.json"

ROUND_SHIM = """function y = round(x, n)
% MATLAB's round(x, n), n decimal digits (not available in Octave)
if nargin < 2
  y = builtin("round", x);
else
  y = builtin("round", x .* 10.^n) ./ 10.^n;
end
end
"""


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 2] *= -1
    return q


def build_tensors():
    """Named tensors: triaxial, axial with the unique axis along z and along x of the PAS."""
    rng = np.random.default_rng(20231849)
    tensors = {}
    # triaxial; traceless ones are also used in NQR order
    for k in range(6):
        E = random_rotation(rng)
        ev = rng.uniform(-6, 6, size=3)
        while np.min(np.diff(np.sort(ev))) < 0.5 or np.min(np.diff(np.sort(np.abs(ev - ev.mean())))) < 0.5:
            ev = rng.uniform(-6, 6, size=3)
        if k % 2:
            ev = ev - ev.mean()
        tensors[f"tri{k}"] = E @ np.diag(ev) @ E.T
    # axial, with the unique eigenvalue larger (z in increasing order) or smaller (x in increasing order)
    for k in range(4):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        tensors[f"prolate{k}"] = -1.0 * np.eye(3) + 3.0 * np.outer(n, n)   # evals -1, -1, 2
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        tensors[f"oblate{k}"] = 1.0 * np.eye(3) - 3.0 * np.outer(n, n)     # evals -2, 1, 1
    # the ALA example tensors shipped with TensorView (Source/Examples), as used in nmr_tests.py
    tensors["ala_ms"] = np.array([[-5.9766, -60.302, -10.8928], [-65.5206, -23.0881, -25.2372],
                                  [-9.5073, -28.2399, 56.2779]])
    tensors["ala_efg"] = np.array([[-0.7806, 0.7215, 0.2987], [0.7215, 1.3736, 0.9829],
                                   [0.2987, 0.9829, -0.5929]])
    return tensors


def build_cases(tensors):
    singles, pairs = [], []
    for name in tensors:
        for order in ("i", "d"):
            singles.append((name, order))
        if abs(np.trace(tensors[name])) < 1e-9 and not name.startswith("ala"):
            singles.append((name, "n"))
    tri = [f"tri{k}" for k in range(6)]
    pro = [f"prolate{k}" for k in range(4)]
    obl = [f"oblate{k}" for k in range(4)]
    # (A, B, order): Soprano A.euler_to(B), TensorView RelativeAngles(Tensor1=B, Tensor2=A)
    combos = [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5]), ("ala_ms", "ala_efg"), ("ala_efg", "ala_ms"),
              (pro[0], tri[0]), (tri[1], pro[1]), (obl[0], tri[2]), (tri[3], obl[1]),
              (pro[2], pro[3]), (obl[2], obl[3]), (pro[0], obl[2]), (obl[3], pro[1])]
    for A, B in combos:
        for order in ("i", "d"):
            pairs.append((A, B, order))
    return singles, pairs


def matlab_matrix(M):
    return "[" + "; ".join(" ".join(f"{x:.17g}" for x in row) for row in M) + "]"


def run_tensorview(tensors, singles, pairs):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in FUNCTIONS:
            src = urllib.request.urlopen(TENSORVIEW_URL.format(f)).read().decode()
            src = re.sub(r'(\w+)\s*==\s*"([A-Za-z]+)"', r'strcmp(\1,"\2")', src)
            (tmp / f"{f}.m").write_text(src)
        (tmp / "round.m").write_text(ROUND_SHIM)
        lines = ["warning('off', 'all');"]
        for name, M in tensors.items():
            lines.append(f"T.{name} = {matlab_matrix(M)};")
        for i, (name, order) in enumerate(singles):
            for mode in MODES:
                lines.append(f'e = MFtoEuler(T.{name}, "{mode}", "{ORDERS[order]}");')
                lines.append(f'printf("S {i} {mode} %.15g %.15g %.15g\\n", e);')
        for i, (A, B, order) in enumerate(pairs):
            for mode in MODES:
                lines.append(f'e = RelativeAngles(T.{B}, T.{A}, "{mode}", "{ORDERS[order]}", "{ORDERS[order]}");')
                lines.append(f'printf("R {i} {mode} %.15g %.15g %.15g\\n", e);')
        (tmp / "driver.m").write_text("\n".join(lines) + "\n")
        out = subprocess.run(["octave-cli", "-q", "--no-gui", "driver.m"], cwd=tmp, capture_output=True,
                             text=True, check=True).stdout
        version = subprocess.run(["octave-cli", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    results = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in ("S", "R"):
            results[(parts[0], int(parts[1]), parts[2])] = [float(x) for x in parts[3:6]]
        elif line.strip() and not line.startswith("ans"):
            # TensorView prints these when its own consistency checks fail
            print("TensorView:", line)
    return results, version


def main():
    tensors = build_tensors()
    singles, pairs = build_cases(tensors)
    results, version = run_tensorview(tensors, singles, pairs)
    data = {
        "description": "Euler angles (radians) computed by TensorView for MATLAB; see tests/reference_generators/tensorview.py",
        "tensorview_commit": TENSORVIEW_COMMIT,
        "octave": version,
        "tensors": {k: np.asarray(v).tolist() for k, v in tensors.items()},
        "single": [{"tensor": name, "order": order,
                    "angles": {m: results[("S", i, m)] for m in MODES}}
                   for i, (name, order) in enumerate(singles)],
        "relative": [{"A": A, "B": B, "order": order,
                      "angles": {m: results[("R", i, m)] for m in MODES}}
                     for i, (A, B, order) in enumerate(pairs)],
    }
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUTPUT}: {len(data['single'])} single, {len(data['relative'])} relative cases")


if __name__ == "__main__":
    main()
