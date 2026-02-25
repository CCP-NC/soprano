# Changelog

All notable changes to Soprano are documented here.

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
