# Changelog

All notable changes to Soprano are documented here.

---

## v0.12.0 (unreleased)

### New features

- **`soprano.SiteMap`** records which atoms of a source structure each site of a derived
  structure (a selection, or a structure with equivalent positions merged) stands for. Maps
  compose (`SiteMap.compose`), so provenance survives chained steps such as symmetry reduction
  followed by group averaging, and expose each site's `multiplicity`. `NMRData2D` uses it to
  evaluate pair metrics on the source atoms.
- **`DipolarRSSByAtom`** (`soprano.properties.nmr`) is the per-atom counterpart of `DipolarRSS`:
  the root-sum-squared dipolar coupling of each atom in `sel_i` to a neighbour set `sel_j` and its
  periodic images within a cutoff, with `sel_j` optionally expanded to symmetry-equivalent atoms
  or atoms sharing a CIF label.
- **Tensor-based methyl dipolar averaging.** `averaged_dipolar_coupling`,
  `averaged_dipolar_coupling_list` (`soprano.properties.nmr.dipolar`) and
  `averaged_j_coupling_list` (`soprano.properties.nmr.isc`) average the full coupling tensor
  over the motion of a functional group, rather than averaging distances or scalar couplings.
  See {doc}`dipolar-averaging`.
- **2D NMR module refactor.** `soprano.calculate.nmr.nmr` is split into
  `soprano.calculate.nmr.data2d` (`NMRData2D`), `plot2d` (`NMRPlot2D`), `backends` (Matplotlib
  and Plotly drawing), `config` (`PlotSettings` and marker/colormap tables) and `export` (`ExportConfig` and the
  Bruker TopSpin, ssNake, SIMPSON, NPZ, CSV and plain text writers). `NMRData2D`, `NMRPlot2D`,
  `PlotSettings` and the marker/colormap tables are still importable from
  `soprano.calculate.nmr.nmr`.
- **New `soprano plotnmr` 2D options:**
  - `--weight-by` chooses the metric that weights each cross-peak: `fixed`, `distance`,
    `inversedistance`, `dipolar`, `dipolar2`, `jcoupling` or `dipolar_rss`. `--rss-cutoff` and
    `--rss-expand-j` (`periodic_images`, `symmetry`, `cif_labels`) control the sum for
    `dipolar_rss`. `--scale-markers/--fixed-markers` scales the marker sizes by the weight.
  - `--contour-levels`, `--contour-range`, `--heatmap-range`, `--intensity-range`,
    `--heatmap-levels` and `--heatmap-grid-size` set the contour and heatmap levels.
  - `--export-file` (repeatable), `--export-format` and `--overwrite` export the contour data;
    `--b0-field-tesla`, `--spectrometer-freq`, `--x-larmor-freq` and `--y-larmor-freq` give the
    Larmor frequencies that the exports need. See "2D NMR spectra and export" below.
- **NMR tutorials reorganised** into `05a-nmr-parameters`, `05b-nmr-2d-spectra` and
  `05c-nmr-tensors-spin-systems` (replacing `05-nmr` and `05a-nmr-spin-systems`).

### Breaking changes

- **`MagneticShielding(gradient=...)` defaults to `None`** (full-formula referencing); it was
  `-1.0`. With a reference and no gradient the shifts are the same as before, but an explicit
  `gradient=-1.0` is now a linear fit (δ = σ_ref − σ). See the referencing item below.
- **`write_spinsys` Python parameters renamed:** `ref` → `references` and `grad` → `gradients`
  (the latter now defaults to `None`, not `-1.0`). The old names still work and emit a
  `DeprecationWarning`; passing both an old and a new name raises `ValueError`.

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
- **`MSTensor` defaults to Haeberlen order** (`'h'`), as `MSEuler`, `MSQuaternion` and
  `MagneticShielding` do; it was increasing (`'i'`). Pass `order='i'` to get the old eigenvalue
  order. Euler angles and quaternions from `MSTensor`, `MSEuler().mean` and `MSQuaternion().mean`
  change accordingly; isotropy, span, skew and anisotropy do not.
- **Exported Euler angles change** (`to_simpson()` for axial tensors; `to_mrsimulator()` for
  every tensor). Re-export any spin systems generated with earlier versions. See below.
- `NMRTensor.euler_to` with two axial tensors, either of which has its unique axis along y,
  now raises `ValueError` as `euler_angles` does (this order is not reachable with the supported
  tensor orders).
- **Chemical shift referencing: an explicit gradient now means a linear fit, with no
  denominator.** The mode is chosen per element (see {doc}`chemical-shift-referencing`):
  - An element without a gradient uses the full formula δ = (σ_ref − σ)/(1 − σ_ref·10⁻⁶).
  - An element with a gradient m uses the linear fit δ = σ_ref + m·σ.

  In v0.11 every gradient, including the default −1, went through the denominator formula.
  `gradients=None` (now the default everywhere, including `get_spin_system`, `NMRSpinSystem`,
  `NMRSites`, `write_spinsys` and `soprano nmr`) gives the same numbers as the old default.
  An explicit `-1` now gives σ_ref − σ, which differs by about σ_ref·10⁻⁶·δ (≈0.02 ppm for ¹³C).

  ```python
  # v0.11: both lines used the denominator formula
  MSShift.get(atoms, references={'H': 31, 'C': 170})
  MSShift.get(atoms, references={'H': 31, 'C': 170}, gradients={'C': -0.98})  # H used m = −1
  # v0.12: first line unchanged; second line is a linear fit for C, full formula for H
  ```

  A `UserWarning` is emitted when a gradient dict leaves out referenced elements, and when a
  gradient is exactly −1. The warning for |σ_ref| > 500 ppm in linear mode is gone, since a
  fitted intercept has no neglected denominator. An INFO log line reports each element's mode.
- **Missing references raise.** `nmr_extract_multi` / `soprano nmr` used to fill elements
  without a reference with 0 ppm; they now raise a `ValueError` naming those elements.
- **CLI flags renamed with no aliases:** `--ref` → `--references`, `--grad` → `--gradients`,
  `--select-i`/`--select-j` → `--select_i`/`--select_j`, `--plot_type` → `--plot-type`,
  `-cmap` → `--colormap`.
- **`soprano plotnmr` defaults:** `--rcut` is now `None` (all pairs; was 10 Å), and contour and
  heatmap levels span 10–100 % of the maximum (was the full Z range).
- **`plotnmr` exit codes:** invalid option combinations are usage errors (exit 2) rather than
  tracebacks or a silent exit 0.
- **`NMRCalculator.spectrum_1d` window and defaults.** (a) A new `NMRCalculator` uses a powder
  average by default (`set_powder(N=8)`, hemisphere). Before, `spectrum_1d` raised
  `AttributeError` for isotropic spectra unless `set_powder()` or `set_single_crystal()` had been
  called first. (b) With a reference set, `min_freq` and `max_freq` are now a chemical-shift window,
  in the same units and scale as the returned axis. Before, they were applied to the shielding
  axis, so the window was reflected about the reference. To get the same spectrum, pass
  `ref - max_shielding` and `ref - min_shielding` as the window. The returned axis still runs from
  `min_freq` to `max_freq`. (c) `freq_broad` is now in `freq_units`. Before, it was always in ppm,
  so MHz spectra were broadened by the wrong amount. The `min_freq`/`max_freq` docstrings are
  also corrected, and the docstring now gives the true default for `bins` (100, not 500).
- **`NMRCalculator.spectrum_1d` referencing and signs.** (a) With a reference, shifts use the
  same formula as `MSShift`: the full formula δ = (σ_ref − σ)/(1 − σ_ref·10⁻⁶) by default, or a
  linear fit with the new `set_reference(ref, element, gradient=...)`. Before, it was always
  σ_ref − σ (≈0.02 ppm off for ¹³C). (b) The second-order quadrupolar shift (`Q_2_SHIFT`, and the
  `Q_2_ORIENT_*` and `Q_1_ORIENT` terms) had the wrong sign: frequency shifts were added on the
  shielding axis, so the central-transition shift moved peaks downfield. It now moves them upfield,
  matching −(3/40)(C_Q/(I(2I−1)))²(I(I+1)−¾)(1+η²/3)/ν₀. (c) MHz offsets are taken against |ν₀|, so a
  positive shift is a positive offset for negative-γ nuclei too. (d) A `multiplicity` array on the
  sample (as left by symmetry reduction or `average_group`) now weights each site. (e) Each 1Q
  transition m → m+1 of a quadrupolar nucleus is weighted by its transition probability
  I(I+1) − m(m+1) (5 : 8 : 9 : 8 : 5 for I = 5/2); all transitions used to count equally, which
  underweighted the central transition. `use_central=True` no longer averages the central
  transition 2I times over.
- **`NMRData2D` evaluates every pair metric on the source structure.** With `reduce=True`, a site
  pair (I, J) is expanded into the atom pairs (a, b), with a the representative atom of I and b every
  atom merged into J, each weighted by m_I. Distances, `rcut`, `dipolar`, `dipolar2`, `distance`,
  `inversedistance` and `jcoupling` therefore give the same peaks and weights with and without
  reduction. Before, they were evaluated between single representatives of the reduced structure,
  which is not a physical crystal: `rcut` dropped intermolecular contacts (36 of 108 peaks on
  EDIZUM at 3 Å), a site pair carried one representative coupling for all of its m_I·m_J atom pairs
  (2.2× the total dipolar weight on EDIZUM), and a DQ self-pair used only the nearest of its m−1
  partners. `NMRData2D.pairs`, `pair_distances` and `correlation_strengths` now hold one entry per
  atom pair, with the atoms in the new `src_pairs` and the weights in `pair_multiplicities`.
  User-supplied `pairs` are evaluated on the requested atoms rather than on the site
  representatives.
- **`dipolar_rss` peaks count each x atom once per j set.** RSS already sums over every atom of the
  j site, so its peak is weighted by m_I instead of m_I·m_J, and with `reduce=False` and
  `rss_expand_j='symmetry'`/`'cif_labels'` the duplicate entries for equivalent j atoms are dropped.
  The reduced and unreduced spectra now agree.
- **`MagneticShielding.shift_span`** (and the `herzfeldberger_shift` / `maryland_shift` Ω) is now
  δ11 − δ33, scaled by |gradient| or 1/(1 − σ_ref·10⁻⁶). It used to return the shielding span.
  Without a reference it still returns the shielding span.

### Fixes

- `NMRData2D` with `reduce=True` and `average_group` averaged a functional group of one molecule
  against the counter-atom of another, because every symmetry copy of the group mapped to the same
  sites and the last one won. The methyl C–H₃ residual coupling came out at −0.13 kHz instead of
  +7.61 kHz on alanine. Groups are now matched on source atoms, and the site representatives of a
  group all come from one physical group.
- `nmr_extract_atoms(reduce=True)` picked each orbit's representative independently, so for inputs
  not ordered molecule by molecule the reduced structure could hold atoms of different molecules,
  and `soprano spinsys` read intermolecular couplings as intramolecular ones. Representatives now
  form one compact unit, grown by nearest neighbour from the first atom.
- SIMPSON `.spe` export wrote its comment block before `DATA`, which SIMPSON rejects and which
  made `nmrglue.fileio.simpson.read` fail on a line with two `=` signs. The comments now follow
  `END`.
- ssNake JSON export, and SIMPSON export with frequency axes, now reject unreferenced data
  (`is_shift=False`), as Bruker export does, instead of writing carriers on the wrong side of ν₀.
- `DipolarCoupling`, `DipolarRSS`, `DipolarRSSByAtom` and `averaged_dipolar_coupling` honour
  `atoms.pbc`. A non-periodic structure with a bounding-box cell got couplings to images across the
  box (−120 kHz instead of −4.4 kHz for two H 3 Å apart in a 4 Å box), and slabs got images along
  their normal.
- `soprano plotnmr -p 1D` crashed with `--isotopes`, ignored `--gradients`, ignored site
  multiplicities (with `-g CH3` a merged methyl counted as one proton), and had an empty window
  for a single site.
- `MSShift`, `MSShielding` and the `JC*` properties (`JCTensor`, `JCAnisotropy`,
  `JCReducedAnisotropy`, `JCAsymmetry`, `JCSpan`, `JCSkew`, `JCQuaternion`) can now be imported
  from `soprano.properties.nmr`, as the other NMR properties can.
- `NMRCalculator.set_element_isotope` accepts isotope symbols (`'17O'`) as well as mass numbers
  (`17`); symbols used to raise "Invalid isotope".
- `NMRTensor.euler_to`, `equivalent_euler_to` and `rotation_to` warn when both tensors are of the
  same type (e.g. two shielding tensors) but use different eigenvalue orders, since the angles then
  differ (e.g. by 90° in γ). Mixing types, such as a Haeberlen-ordered shielding tensor with an
  NQR-ordered EFG tensor, stays silent. Returned values are unchanged.
- **`NMRCalculator.spectrum_1d` with `freq_units='MHz'` for negative-gamma nuclei** (e.g. ¹⁷O)
  returned an empty spectrum, because the powder average needs an ascending axis and the MHz axis
  is descending. The spectrum is now computed on an ascending axis and returned in the requested
  order.

- **`AtomsProperty.get` honours instance parameters when called on an instance.**
  `EFGQuadrupolarConstant(isotopes={'H': 2}).get(atoms)` used to ignore the isotopes and return
  zeros for H; it now matches `instance(atoms)`. Keyword arguments still override instance
  parameters, and calling `get` on the class is unchanged. `JCEuler` now also uses the parameters
  it passes to `JCTensor`, so its output changes when those parameters are not the defaults.
  `mean()` on the base class follows the same rule. The NMR `mean` overrides still ignore instance
  parameters.

- **`NMRData2D` `pairs` are always indices into the input structure.** Without `reduce=True`,
  pairs used to be read against the structure after filtering to the x and y elements, so a
  structure with O before C or H could pick the wrong atoms or raise. Pairs now refer to the atoms
  you passed in, with or without `reduce=True`. A pair whose atom is not of the x or y element
  raises `ValueError`, naming the index and its element.

#### NMR extraction and CLI

- New `--expand-group/--collapse-group` for `soprano spinsys`. An expanded group keeps each atom
  as a spin with the group-mean tensors and the couplings inside the group. It is the default when
  `--dip` or `--jcoupling` is given, and groups collapse otherwise. New functions
  `averaged_dipolar_coupling_list` and `averaged_j_coupling_list`, and the `return_site_sets` and
  `expand_groups` arguments of `nmr_extract_atoms`, support it.

- `EFG_Vzz` in `soprano nmr` output was wrong by a large factor; it is now in V m⁻²
  (ethanol O: −1.81 × 10²² V m⁻²). `EFG_quadrupolar_constant` was not affected.
- `soprano nmr --mean-merge` is available again.
- `--average-group` averaged symmetry-equivalent methyl groups together, because groups were
  matched by tag. Tensors from groups in different orientations were mixed, which distorted the
  anisotropy and Euler angles. Each group is now averaged on its own and equivalent groups merge
  afterwards. Output changes for structures with symmetry-equivalent methyls. `--mean-merge` still
  pools them.
- `soprano spinsys --average-group` took couplings from the merged structure, so a collapsed CH₃
  got the first hydrogen's static coupling. Couplings are now averaged over the group in the source
  structure (see {doc}`dipolar-averaging`).
- `SiteMap` keeps its read-only array through pickle on every numpy release.
- `nmr_extract_multi` no longer stops at the first unreadable file, and no longer crashes when
  `include` is left at its default.
- `soprano.nmr.extract` no longer imports from `soprano.scripts`. The dataframe helpers live in
  `soprano.nmr.dataframe`; the old import paths in `soprano.scripts` still work.
- `soprano spinsys` error paths now exit non-zero (`UsageError` / `ClickException`) instead of
  logging and returning exit code 0, and multi-frame files are rejected rather than silently
  reading the last frame.
- `soprano plotnmr` now requires exactly one input file (instead of silently plotting only the
  first), honours `--subset` in `-p 1D` while rejecting it in `-p 2D`, and rejects `--export-file`
  / `--export-format` in `-p 1D`.

#### 2D NMR spectra and export

Exporting 2D spectra (`export_contour_data`, `plotnmr --export-file`) to Bruker TopSpin,
ssNake, SIMPSON, NPZ, CSV and plain text is **experimental** in this release. Formats and
options may change without a deprecation period, and the first export in a session logs a
warning saying so.

- Bruker datasets open in TopSpin 5: the parameter files carry the JCAMP-DX header TopSpin
  requires, and `YMAX_p`/`YMIN_p` plus a `clevels` file give it usable default contours.
- Bruker export writes `<path>/1/pdata/1/...`: `path` is the TopSpin dataset name and the
  experiment is expno 1. Previously `path` was itself the experiment directory, which TopSpin
  refused to open (`BROWSE_CORRUPT`). `peaks.csv` is now `<path>/1/peaks.csv`.
  A `pdata/1/outd` file is also written; without it TopSpin's Plot tab fails with
  `getpar failed ... outd`.
- Larmor frequencies use the selected isotope (e.g. ¹⁵N rather than ¹⁴N).
- Bruker export: the indirect axis now round-trips exactly through nmrglue; an existing
  dataset is not overwritten unless `ExportConfig(overwrite=True)` / `--overwrite` is given;
  NaN, all-zero and int32-overflowing grids raise instead of writing corrupt data.
- SIMPSON `.spe` and ssNake JSON export: the spectral width is scaled by N/(N−1) to match the
  endpoint-inclusive grid, and the carrier (`SPECFREQ`, written at full precision) and
  0 ppm reference (`SPECREF`) are written so ssNake rebuilds the ppm axis. Previously the
  window was centred on 0 ppm and one grid step short.
- The 2D Lorentzian is the separable product of two 1D Lorentzians, so each linewidth no
  longer depends on the other coordinate and normalisation gives unit area.
- DQ-SQ with `reduce=True` keeps the self-pairs of merged sites, weighted m(m−1), so diagonal
  peaks from equivalent atoms (e.g. within a methyl group) are no longer lost.
- `merge_peaks` merges peaks within the tolerances of a group's first peak; merged labels no
  longer leak between groups that share a label.
- `NMRPlot2D` no longer modifies the caller's `PlotSettings`; Plotly labels stay on the
  correct edges when axes are inverted; an all-zero contour grid raises a clear error; any valid
  matplotlib colormap (including `_r` reversed variants such as `grey_r` and `viridis_r`) is
  accepted.
- NPZ export writes the exact path requested instead of silently appending `.npz`.

#### Core

- `SiteMap` rejects duplicate selection indices and non-integer maps, no longer freezes the
  caller's array, and gains a `multiplicity` property.
- `minimum_periodic` handles a zero cell again.
- `UniqueSites(override_cif=True)` now actually uses the symmetry reduction; before, both
  branches returned the CIF-label grouping. CIF labels that give the same grouping as symmetry
  are no longer reported as a mismatch.
- `averaged_dipolar_coupling` raises for coincident atoms instead of returning NaN.
  `DipolarRSSByAtom` caches symmetry tags per structure.

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
