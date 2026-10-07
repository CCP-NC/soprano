# ADR 0001: Bruker TopSpin export format via nmrglue

**Status:** Accepted  
**Date:** 2026-06-18

## Context

NMR crystallographers want to load Soprano-simulated 2D spectra directly into Bruker TopSpin for visual overlay against experimental data. No current Soprano export format (SIMPSON, NPZ, CSV, ssNake JSON, plain text) is natively readable by TopSpin for 2D data.

The Bruker processed 2D format (`pdata/1/2rr` + `procs`/`proc2s` parameter files) is the only format TopSpin reads natively. The binary encoding uses 32-bit big-endian integers scaled by a `NC_proc` exponent, with data stored **high-ppm → low-ppm** (descending δ) in both dimensions.

## Decision

Add Bruker format as a new export target (`fmt="bruker"`, extension `.bruker` or output as a directory), implemented via **`nmrglue`** as an optional dependency (imported at call time; not required at install).

**Constraints baked into this format:**
- Requires `is_shift=True` on `NMRData2D`; raises `ValueError` for shielding data — exporting unreferenced shieldings to an experimental format is scientifically meaningless.
- Requires Larmor frequency (same requirement as ssNake JSON export); raises `ValueError` if unavailable.
- Grid is flipped (ascending → descending δ) before serialisation; `OFFSET` = `ContourData.xlims[1]` (downfield edge in shift mode).
- `nmrglue`'s `write_pdata` writes only `procs`/`proc2s`. Soprano additionally writes `acqus`/`acqu2s` (plus the `acqu`/`acqu2` copies) at the experiment root: TopSpin will not open the dataset without them, and `guess_udic` reads axis information from them. This forces a synthetic carrier at the centre of the spectral window, which has no physical meaning for simulated data.

## Alternatives considered

**Hand-rolling the binary writer** — avoids the dependency but requires owning the binary format details (NC_proc scaling, byte order, JCAMP-style parameter file quoting). Bruker's format has undocumented quirks that nmrglue's implementation has already resolved against real TopSpin data. The risk of a silent scaling error that corrupts contour levels in a scientific tool outweighs the cost of the optional dependency.

## Consequences

- `nmrglue` added as an optional dependency (e.g. `pip install soprano[nmr-io]` or similar).
- Users without `nmrglue` get a clear `ImportError` with install instructions when they request Bruker format.
- The format produces a directory (not a single file), which differs from all other export formats. The public API should accept a `path` that is treated as the experiment directory root.
