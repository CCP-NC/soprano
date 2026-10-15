---
name: soprano-domain
description: Domain glossary for Soprano's crystal structure handling and NMR crystallography features
metadata:
  type: reference
---

# Soprano Domain Glossary

## Structure and Index Spaces

**Atom** — an entry in the structure as supplied by the user. The atoms of a structure are its canonical index space; everything else is derived from it.
_Avoid_: site (when you mean an atom of the input)

**Site** — an entry in a structure derived from the input, standing for one or more atoms. Selecting a subset yields sites that each represent a single atom; merging symmetry-equivalent or chemically equivalent positions yields sites that represent several. Broader than the crystallographic sense of "site": a Soprano site is any group of atoms treated as one entry, which includes chemically (not just symmetry) equivalent groups such as a rotating methyl.
_Avoid_: unique atom, reduced atom, merged atom

**Multiplicity** — the number of atoms a site represents. One for every site of an underived structure.
_Avoid_: degeneracy, weight, occupancy

**Site map** — the correspondence between the atoms of a structure and the sites of a structure derived from it. Composable: deriving twice yields a single map back to the original atoms.
_Avoid_: index map, reduce map, atoms_full

**Source** — the structure whose atoms a site map is expressed in terms of. Always the user's input, never an intermediate.
_Avoid_: full atoms, parent, original

## Core NMR Quantities

**Magnetic shielding (σ)** — the raw quantity Soprano computes from a crystal structure (e.g. via CASTEP). Higher σ = more shielded nucleus = upfield. Stored in ppm units. *Axis convention: ascending σ runs downfield-to-upfield, which happens to match the NMR display convention (downfield on the left).*

**Chemical shift (δ)** — the experimentally reported quantity, referenced against a standard: δ = σ_ref − σ. The subtraction *inverts the axis* relative to σ. Higher δ = less shielded = downfield. *Axis convention: descending δ runs downfield-to-upfield (high ppm on the left).*

**is_shift** — boolean flag on `NMRData2D` indicating whether `data` holds chemical shifts (True, references provided) or raw shieldings (False). Governs axis labelling and export behaviour. Export to experimental formats (e.g. Bruker) requires `is_shift=True`.

## 2D NMR Data Representations

**Contour grid** — the broadened 2D intensity map (X, Y, Z arrays at `grid_size × grid_size` resolution) produced by `get_contour_data()`. Stored in `ContourData`.

The *raw arrays* are always ascending in whatever units peaks are stored in (σ or δ) — a plain `np.linspace(lo, hi)`, with no branch on `is_shift`. Downfield-left is not a property of the array; it is imposed downstream, and each consumer does it differently:

- **Plotting** inverts the *displayed* axis when `is_shift=True` (matplotlib `invert_xaxis`, plotly `autorange="reversed"`). In shielding mode the axis is left un-inverted, which already gives downfield-left because ascending σ runs downfield-to-upfield.
- **Bruker export** reverses the *array itself* (`Z[::-1, ::-1]`), because Bruker stores processed data downfield-first.

**Grid limits (`ContourData.xlims`/`ylims`)** — the extent of the computed grid: the peak range plus 5 × FWHM padding on each side, possibly widened to cover requested limits. Distinct from `PlotSettings.xlim`/`ylim`, which only control the visible window and never clip the grid. Export parameters are derived from the *grid* limits, not the display limits.
_Avoid_: using "xlims" without saying which of the two you mean

**Peak list** — the discrete set of `Peak2D` objects (x_ppm, y_ppm, xlabel, ylabel, correlation_strength) before any broadening is applied. The scientifically meaningful output for assignment workflows.

**Correlation strength** — a scalar on each `Peak2D` encoding the relative intensity of the simulated cross-peak (e.g. distance-weighted dipolar coupling, J-coupling magnitude). Not a physical transfer efficiency.

## Experiment Types Supported

Soprano's 2D NMR is nucleus-agnostic via `xelement`/`yelement`. Targeted experiment classes for NMR crystallography:

- **Homonuclear (e.g. ¹³C–¹³C DARR/PDSD)** — both axes same nucleus, same Larmor frequency
- **Heteronuclear (e.g. ¹H–¹³C HETCOR, ¹³C–¹⁵N)** — axes are different nuclei with different Larmor frequencies
- **Double-quantum (DQ-SQ)** — y-axis is the DQ coherence frequency (δ_i + δ_j); handled via `yaxis_order='2Q'`. Self-pairs of single atoms are removed: a DQ diagonal peak comes from two *distinct* atoms of equal shift, not from a spin correlated with itself. Self-pairs of merged sites (`reduce=True`) are kept, weighted by the m(m−1) ordered pairs of distinct atoms they stand for, with distance/coupling taken from the nearest equivalent atom in the source structure.

## Export Concepts

**Larmor frequency** — nucleus-specific resonance frequency at a given B0 field (MHz). Required for any format that uses Hz-based spectral widths (SIMPSON, ssNake JSON, Bruker). Can be supplied directly or derived from B0 (Tesla) or spectrometer ¹H frequency (MHz).

**OFFSET (Bruker)** — ppm value of the downfield (leftmost) edge of a processed Bruker spectrum, written to `procs`/`proc2s`. Taken from the *contour grid* limits, not the display limits: `OFFSET = ContourData.xlims[1]` (largest δ) for the direct dimension, `ylims[1]` for the indirect one.

**SF (Bruker)** — the frequency in MHz that Bruker uses to convert between the Hz and processed-ppm scales of one dimension: it scales `SW_p` and fixes the frequency assigned to 0 ppm. Soprano writes the nominal Larmor frequency |γ|B₀/2π for the observed nucleus. Strictly that is the *bare-nucleus* frequency, not the frequency of the reference compound at 0 ppm — the two differ by σ_iso of the reference (~188 ppm for ¹³C/TMS). The discrepancy is irrelevant here because SF only rescales a ppm range that is already correct, but SF, BF1, SFO1 and the physical Larmor frequency are not synonyms and should not be written as if they were.

**Carrier frequency** — the transmitter offset during acquisition. Simulated spectra have no real carrier, but the Bruker export synthesises one anyway: `acqus`/`acqu2s` place it at the centre of the spectral window (`O1 = centre_ppm × SF` in Hz, `SFO1 = SF + O1 × 1e−6` in MHz), because TopSpin needs these files to open the dataset and nmrglue's `guess_udic` reconstructs the axes from `car = (SFO1 − SF) × 1e6`. The value carries no physical meaning; it is chosen so the axes round-trip.

**SW_p endpoint convention (Bruker)** — Bruker frequency axes are endpoint-*exclusive*: point *i* sits at `OFFSET − i × (SW_p/SF)/SI`. The contour grid is an endpoint-*inclusive* `linspace` with step `range/(SI−1)`. The export therefore scales the spectral width by `SI/(SI−1)` so the per-point step matches and the last point lands exactly on the upfield limit. Omitting this puts every point one grid step out.

**2Q indirect axis (Bruker)** — a DQ dimension carries δ_i + δ_j ppm values, but the export applies *no* multiple-quantum scaling to the indirect axis, and `y_larmor_freq_mhz` silently defaults to the direct-dimension value. So a DQ-SQ export is read back by TopSpin as a 1Q axis whose ppm values happen to be sums. Deliberate (it preserves the DQ ppm scale as plotted), but it means the indirect axis of an exported DQ spectrum is not a physical frequency axis.

## Target Workflow

Primary audience: **NMR crystallographers** comparing computational predictions against experimental solid-state NMR data. Primary use case: **visual overlay in Bruker TopSpin** — load simulated spectrum alongside experimental data using TopSpin's dual display. Assignment-assist workflows (importing labelled peak lists into CcpNmr Analysis or Sparky) are out of scope for the current export feature.
