# Changelog

All notable changes to Soprano are documented here.

---

## v0.12.0 (unreleased)

### Breaking changes

- **`NMRTensor.euler_to` and `equivalent_euler_to` now return the inverse rotation** for
  triaxial and mixed axial/triaxial pairs, and the same call gives different numbers. This is a
  change of convention, not a bug fix. `A.euler_to(B)` is now the orientation of `B` in the PAS of
  `A` (`E_B = E_A R`), as the method name reads. It used to be the orientation of `A` in the PAS of
  `B`, which is TensorView's positional order for `RelativeAngles(Tensor1, Tensor2)`. The argument
  order is now the reverse of TensorView's: to compare with TensorView, put `B` first.
  To reproduce old values, call `B.euler_to(A)`; the result is equivalent but may be a different
  one of the 16 equivalent angle sets. Code that uses relative angles (for example to set up
  mrsimulator couplings) must be recomputed or have its arguments swapped. There is no deprecation
  period, because a warning cannot tell old callers from new ones. Details are under "Relative Euler
  angles" below.
- **Exported Euler angles change** (`to_simpson()` for axial tensors; `to_mrsimulator()` for
  every tensor). Re-export any spin systems generated with earlier versions. See below.
- `NMRTensor.euler_to` with two axial tensors, either of which has its unique axis along y,
  now raises `ValueError` as `euler_angles` does (this order is not reachable with the supported
  tensor orders).

### Fixes

#### Tensor orientations in the SIMPSON and mrsimulator exports

Several bugs made the exported Euler angles wrong whenever more than one oriented
interaction was present (single-interaction powder spectra do not depend on orientation, so
these went unnoticed). Re-export any spin systems you generated with earlier versions.

- **mrsimulator angles.** `Site.to_mrsimulator()`, `ISCoupling.to_mrsimulator()` and
  `DipolarCoupling.to_mrsimulator()` now write the angles in the convention of mrsimulator's
  own code (`E = Rz(gamma) Ry(beta) Rz(alpha)`, as in `mrsimulator.utils.cartesian_tensor`), via
  the new `NMRTensor.mrsimulator_euler_angles()`. Previously they wrote TensorView's active ZYZ angles,
  `E = Rz(alpha) Ry(beta) Rz(gamma)`. mrsimulator's code uses the same matrices with `alpha` and `gamma`
  in each other's places, so those numbers described a different relative orientation in mrsimulator.
  This is a difference between two conventions, not a bug in mrsimulator. These are *not* the numbers `to_simpson()` writes: SIMPSON uses passive
  angles in degrees.
- **Axial tensors (SIMPSON and mrsimulator).** For axial tensors (every dipolar coupling, and
  any CSA, EFG or J tensor with `eta = 0`) the passive angles had the wrong angle zeroed, so
  the unique axis kept its polar angle but lost its azimuth. A dipole at azimuth 40° was
  written as pointing at 180°. Spectra with an axial tensor tilted out of the
  xz-plane relative to another tensor were wrong in both programs.
- **mrsimulator `zeta` and referencing.** `zeta` now follows the same referencing as the
  isotropic shift and `to_simpson()` (`-shift_reduced_anisotropy`). With a custom `gradient`
  it was previously unscaled; with the default full formula it changes by a factor
  `1 / (1 - reference * 1e-6)` (about 3e-5 for ¹H, 1e-4 for ¹³C).
- **mrsimulator and tensor order.** `Site.to_mrsimulator()` now uses Haeberlen ordering for the
  shielding angles, as `to_simpson()` does, so they refer to the same axes as `zeta` and
  `eta` for tensors built with another `order`.
- **Coupling exports and tensor order.** `ISCoupling.to_simpson()` and `to_mrsimulator()` took
  the Euler angles in the tensor's own eigenvalue order, while `zeta` and `eta` are always
  Haeberlen, so the angles could refer to different axes. They now use Haeberlen order.
  `DipolarCoupling.coupling_constant`, `to_simpson()` and `to_mrsimulator()` assumed NQR order
  (the unique eigenvalue last). A tensor in another order gave a wrong coupling constant, for
  example `d/2` with the opposite sign in increasing order with `d < 0`, and wrong angles. The
  coupling constant is now half the tensor's reduced anisotropy, which doesn't depend on the order,
  and the angles use Haeberlen order like the other exports. The angles are taken from a copy, and
  the coupling's own tensor is not changed. Couplings from
  `JCouplingList`, `DipolarCouplingList` and `DipolarCoupling.from_distance_vector` already used
  these orders and are unaffected. Only couplings built from an `NMRTensor` in another order change.
- `Site.to_mrsimulator()` no longer changes the order of the site's own EFG tensor (it works on a copy, as `to_simpson()` does).
- `ElectricFieldGradient.__str__` and a comment gave the quadrupole moment units as barns; it is millibarn.

#### Relative Euler angles (`euler_to`) and axial tensors

- `NMRTensor.euler_to` and `equivalent_euler_to` returned the orientation of the first tensor in
  the PAS of the second for triaxial and for mixed axial/triaxial pairs (TensorView's positional
  order), i.e. both active and passive angles described the inverse of what they now return. They
  now return the rotation carrying the first tensor's PAS onto the second's
  (`E_other = E_self R(alpha, beta, gamma)`), the orientation of the second tensor in the PAS of
  the first (Eq. 31 of the TensorView paper). `A.euler_to(B)` therefore matches TensorView's
  `RelativeAngles(Tensor1, Tensor2)` with Tensor 1 = `B`, Tensor 2 = `A`. Anything computed from
  relative angles with an earlier version (for example, to set up mrsimulator couplings) should be
  recomputed. The docstring and `docs/simpson-conventions.md` now say which way round this is.
- Two axial tensors: the angle between the unique axes was wrong when either tensor has its
  unique axis along x (as for increasing order), and two tensors whose axes were not aligned could
  be reported as "perfectly aligned" for particular geometries.
- A tensor with its unique axis along x (e.g. eigenvalues `[1, 2, 2]` in increasing order) is now
  given TensorView's gauge, active `alpha = 0`, followed by the usual range normalisation (so
  `alpha` is 0 or pi), exactly as TensorView reports it; before, the code for this was never reached.
- When exactly one of the two tensors in `euler_to` is axial, its free angle is now fixed as for a
  single tensor (written in the PAS of the other), which also covers a unique axis along x.
- All angles, including the relative ones of two axial tensors (Table 1 of the paper), are now
  reported in TensorView's range: beta in [0, pi/2), then gamma (active) or alpha (passive) in
  [0, pi). Table 1's `(0, 90°, g)` is therefore returned as `(180°, 90°, 180° - g)`, and the passive
  ZYZ angles of two axial tensors as `(0, beta, 180°)`. These are equivalent angle sets; only the
  numbers change.
- The meaning of the angles is now defined in the `euler_angles` and `euler_to` docstrings and in
  `docs/simpson-conventions.md` in TensorView's terms. `tests/nmr_euler_convention_tests.py` checks
  the conventions against matrices and angle sets written out from the paper, and against output
  of TensorView itself (`tests/test_data/tensorview_reference.json`, made by running TensorView in
  Octave with `tests/reference_generators/tensorview.py`).

The exact angle convention is now documented in {doc}`simpson-conventions`, with a test
suite (`tests/nmr_export_orientation_tests.py`) that rebuilds each tensor from the written
angles, and checks the formulas it uses against stored SIMPSON and mrsimulator spectra of the
written spin systems, and single-crystal SIMPSON simulations that fix the absolute frame of the SIMPSON angles for a CSA with a dipolar or a J coupling (run in CI with SIMPSON 6.1.4 from the `jkshenton/simpson` container). Note that mrsimulator's default octant powder integration is wrong for
spin systems with tensors in general relative orientations; use
`sim.config.integration_volume = "hemisphere"`.

---

## v0.11.1 (2026-02-25)

### Changes

- Removed `bottleneck` as a dependency — it was never imported by Soprano.
- Widened the `click` version constraint from `>=8.2.1` to `>=8.0`, restoring
  compatibility with environments that ship older click releases (e.g. Pyodide
  0.27, which provides click 8.1.7).

---

## v0.11.0 (2026-02-25)

### New features

#### Pluggable symmetry backends: spglib and moyopy

Soprano now supports two interchangeable symmetry backends. spglib is no longer
a hard dependency. The new abstraction in `soprano.properties.symmetry.backend`
exposes two public functions:

- `get_symmetry_dataset(atoms, *, symprec=1e-5, backend="auto")` → `SpacegroupDataset`
- `get_symmetry_ops_from_hall(hall_no, *, backend="auto")` → `(rotations, translations)`

The `SpacegroupDataset` dataclass provides a unified interface regardless of
which backend performed the analysis:

```python
SpacegroupDataset(
    international,        # H-M symbol, e.g. "Fm-3m"
    hall_number,          # integer 1–530
    std_lattice,          # (3,3) ndarray, rows are lattice vectors
    transformation_matrix,
    origin_shift,
    rotations,            # (N,3,3)
    translations,         # (N,3)
    equivalent_atoms,     # (nat,) integer orbit labels
)
```

The `backend` keyword (`"auto"` / `"moyo"` / `"spglib"`) is accepted by:
`SymmetryDataset`, `WyckoffPoints`, `XRDCalculator`, `compute_asymmetric_distmat`,
and `UniqueSites`.

#### moyopy: faster Rust-based symmetry backend

[moyopy](https://github.com/spglib/moyo) is a Rust-based rewrite of spglib
approximately 4× faster. Install it with:

```sh
pip install soprano[moyo]
```

#### spglib moved to optional extras

```sh
pip install soprano[spglib]    # spglib only
pip install soprano[moyo]      # moyopy only (recommended)
pip install soprano[symmetry]  # both (useful for CI / validation)
```

If neither backend is installed, symmetry functions raise an `ImportError` with
installation instructions.

#### New helpers in `soprano.optional`

```python
from soprano.optional import has_spglib, has_moyopy, has_symmetry_backend
```

`requireSpglib` is still importable but emits a `DeprecationWarning`.

### Bug fixes

#### `UniqueSites`: replaced deprecated `ase.spacegroup.get_spacegroup`

`UniqueSites` previously used ASE's `get_spacegroup`, deprecated in ASE 3.24
and removed in ASE 3.25. The function was also **incorrect** for non-standard
cell settings: it returned symmetry operations for the *standard* setting of the
detected space group, so `tag_sites()` could silently produce wrong site
equivalences (see [ASE !3455](https://gitlab.com/ase/ase/-/merge_requests/3455)).

The replacement calls the symmetry backend directly on the input structure and
uses `equivalent_atoms` / `orbits` from the dataset, which are always computed
from the actual input cell.

#### scipy 1.15 compatibility: `sph_harm` removed

`scipy.special.sph_harm` was removed in scipy 1.15 and replaced by `sph_harm_y`,
which also swapped argument order and the θ/φ angle convention. A compatibility
shim is now in place for scipy ≥ 1.15 with a fallback to the original import on
older versions. Fixes `TestGenes::test_coordhist` and all `TestPhylogen` tests
on scipy ≥ 1.15.

### Notes

#### Wyckoff points and transformation-matrix conventions

`_find_wyckoff_points` internally prefers spglib's transformation-matrix
convention when spglib is available. The two backends may return different (but
equally valid) matrices for the same space group due to freedom in the choice of
Euclidean normalizer (see [spglib/moyo#198](https://github.com/spglib/moyo/issues/198)).
Moyo is only used for Wyckoff analysis when spglib is absent.

---

## v0.10.2 (2025-12-09) — NumPy 2.0 bugfix

- Fixed `average_quaternions` for NumPy 2.0 compatibility
  ([#42](https://github.com/CCP-NC/soprano/pull/42))

---

## v0.10.1 (2025-07-28)

- Added support for custom `ms` and `efg` array tags
  ([#40](https://github.com/CCP-NC/soprano/pull/40), @carlosbornes)
- Fixed trailing singleton dimensions in tensor `average` method
  ([#41](https://github.com/CCP-NC/soprano/pull/41))

---

## v0.10.0 (2025-07-24) — NumPy 2.0 compatibility and ASE upgrade

Full NumPy 2.0 compatibility while maintaining backward compatibility with
NumPy 1.x. The minimum ASE version is now 3.26.

- **Dependencies**: removed `numpy<2.0` restriction; now supports `numpy>=1.18.5`
  ([#39](https://github.com/CCP-NC/soprano/pull/39))
- **NumPy 2.0 fixes**: replaced deprecated `np.array(copy=False)` with
  `np.asarray()` throughout `soprano/utils.py` and `soprano/calculate/xrd/xrd.py`
- **Collection**: fixed ASE calculator import compatibility; optimised
  `get_array()` for NumPy 2.0 copy semantics
- **NMR tensor averaging**: new averaging functionality for tensors and other
  properties ([#34](https://github.com/CCP-NC/soprano/pull/34))
- **CLI**: better file-type error handling
  ([#27](https://github.com/CCP-NC/soprano/pull/27),
  [#30](https://github.com/CCP-NC/soprano/pull/30),
  [#38](https://github.com/CCP-NC/soprano/pull/38))
- **Simpson interface**: updates to the Simpson NMR simulation interface
  ([#28](https://github.com/CCP-NC/soprano/pull/28), @carlosbornes)

*New contributor: @carlosbornes ([#28](https://github.com/CCP-NC/soprano/pull/28)).*

---

## v0.9.2 (2024-10-03)

- Added `soprano` CLI ([#26](https://github.com/CCP-NC/soprano/pull/26))

---

## v0.8.14 (2024-07-04)

- Migrated documentation to JupyterBook
  ([#17](https://github.com/CCP-NC/soprano/pull/17),
  [#18](https://github.com/CCP-NC/soprano/pull/18))
- Defect generators: new `max_attempts` and `random` flags
  ([#22](https://github.com/CCP-NC/soprano/pull/22))
- NMR Euler angles and conventions
  ([#25](https://github.com/CCP-NC/soprano/pull/25))

---

For older releases see the [GitHub releases page](https://github.com/CCP-NC/soprano/releases).
