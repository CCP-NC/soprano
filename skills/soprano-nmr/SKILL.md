---
name: soprano-nmr
description: Use when analysing NMR parameters from .magres or NMR-annotated .xyz files with Soprano — extracting magnetic shielding (MS), chemical shifts, EFG/quadrupolar couplings, J-couplings, dipolar coupling tables or RSS, computing 1D powder spectra, configuring ~/.soprano/config.ini, or using soprano nmr, soprano dipolar, soprano.properties.nmr, or NMRCalculator. For spin-system construction, NMRTensor Euler conventions, fast-exchange dipolar tensor averaging, or SIMPSON/mrsimulator export, use soprano-spin-systems instead. For general crystal collections, symmetry, molecules, generators, phylogen clustering, or XRD, use soprano-structures.
---

# Soprano NMR Analysis

Soprano provides both a Click CLI (`soprano nmr`, `soprano dipolar`) and a Python API (`soprano.properties.nmr`, `soprano.calculate.nmr`) for solid-state NMR crystallography from CASTEP/QE `.magres` files and extended `.xyz` files.

## Installation

Install this skill first, then make Soprano available in one of these ways.

**Install permanently (CLI and Python API):**

```bash
pip install soprano[moyo]   # includes the moyopy symmetry backend
```

The `[moyo]` extra is needed for the default symmetry reduction (`--reduce`). Without it, use `--no-reduce`. Check the install with `soprano --help`.

**Run without installing (via `uv`):**

```bash
uvx --from "soprano[moyo]" soprano nmr structure.magres -s H -g CH3 --references H:30
```

From a local checkout (for example, a development branch), point `uvx` at the directory:

```bash
uvx --from /path/to/soprano soprano plotnmr -x C -y H structure.magres
```

For Python code, `uv` can run a script with Soprano available: `uvx --from "soprano[moyo]" python script.py`.

## Choose CLI vs Python First

Do **not** write custom ASE loops when a Soprano CLI subcommand or `AtomsProperty` already handles the job:

| Task | Preferred entry point |
|---|---|
| Summary table (MS + EFG) from one or many `.magres` files | `soprano nmr` |
| Pairwise dipolar couplings or dipolar RSS table from any structure file | `soprano dipolar` |
| Programmatic extraction across an `AtomsCollection` or custom DataFrame | `soprano.properties.nmr` (`MS*`, `EFG*`, `JC*`, `Dipolar*`) |
| Referenced chemical shifts per atom in Python | `MSShift` from `soprano.properties.nmr.ms` |
| Fast 1D diluted-atom MAS/static powder spectrum approximation | `soprano.calculate.nmr.NMRCalculator` |
| Spin systems, Euler angle transforms, SIMPSON / mrsimulator input | Switch to `soprano-spin-systems` |

## Core CLI Workflows

### 1. Extract MS & EFG Summaries (`soprano nmr`)

```bash
# Full summary (MS + EFG, symmetry-reduced by default)
soprano nmr structure.magres

# Chemical shifts for C and H, written to CSV
soprano nmr -p ms -s C,H --references C:170,H:30 structure.magres -o shifts.csv

# Linear DFT calibration (explicit gradient switches to δ = ref + m * σ)
soprano nmr -p ms -s C,H --references C:170,H:30 --gradients C:-0.97,H:-1.0 structure.magres

# Custom isotopes for EFG (e.g. deuterium, 17O, 14N)
soprano nmr -p efg -s H --isotopes 2H structure.magres

# Batch multiple files into one merged table, filtered with pandas query
soprano nmr --merge -s H --query "MS_shielding > 20 and MS_asymmetry > 0.3" *.magres -o all_H.csv

# Extended XYZ files with custom array names
soprano nmr ml_preds.xyz --ms-tag pred_ms --efg-tag pred_efg -o summary.csv
```

### 2. Extract Dipolar Couplings (`soprano dipolar`)

Works on `.magres`, `.cif`, `POSCAR`, or any ASE-readable structure file:

```bash
# C-H dipolar couplings stronger than 5 kHz (note underscores in --select_i/--select_j)
soprano dipolar structure.magres --select_i C --select_j H --query "abs(D) > 5.0" -o ch_dip.csv

# Per-atom dipolar RSS within 5.0 Å (homonuclear only)
soprano dipolar structure.magres --rss --cutoff 5.0 --isonuclear --select_i H
```

## Core Python Workflows

### Extracting NMR Properties

```python
from ase.io import read
from soprano.properties.nmr import (
    MSIsotropy, MSAnisotropy, MSAsymmetry, MSTensor,
    EFGQuadrupolarConstant, EFGAsymmetry, EFGTensor,
    DipolarCoupling, DipolarRSS,
)
from soprano.properties.nmr.ms import MSShift  # NOTE: not re-exported in soprano.properties.nmr!

atoms = read("structure.magres")

# Magnetic shielding & chemical shifts
iso_shielding = MSIsotropy.get(atoms)                                  # ppm (all atoms)
c_shifts = MSShift.get(atoms, selection="C", references={"C": 170.0})  # ppm (C only)
ms_tensors = MSTensor.get(atoms, references={"C": 170.0, "H": 30.0, "O": 250.0})

# EFG / quadrupolar parameters (specify isotope if default isotope is spin-1/2!)
cq_2h = EFGQuadrupolarConstant.get(atoms, selection="H", isotopes={"H": 2})  # Hz
eta_q = EFGAsymmetry.get(atoms, selection="H")

# Dipolar couplings (dict of (i, j) -> (d_Hz, unit_vector))
dip = DipolarCoupling.get(atoms, sel_i=[0, 1], sel_j=[6, 7])
d_rss = DipolarRSS.get(atoms, cutoff=5.0, isonuclear=True)  # Hz
```

### Averaging Fast-Rotating Groups in Python (`-g CH3`)

The CLI `-g` option is two steps in Python. Both are needed. `tag_functional_groups` only tags the atoms, and `merge_tagged_sites` does the merge. If you skip the second step, the methyl hydrogens stay separate and nothing warns you.

```python
from ase.io import read
from soprano.scripts.nmr import label_atoms, tag_functional_groups, merge_tagged_sites
from soprano.properties.nmr.ms import MSShift

atoms = label_atoms(read("structure.magres"))
atoms = tag_functional_groups("CH3", atoms, vdw_scale=1.0)  # tag CH3 hydrogens
atoms = merge_tagged_sites(atoms)                           # merge each tagged group into one site

h_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
shifts = MSShift.get(atoms, selection="H", references={"H": 30.0})
labels = atoms.get_array("labels")
for i, d in zip(h_idx, shifts):
    print(labels[i], round(float(d), 2))  # merged methyls show as "H12,H13,H14"
```

Use `label_atoms` first so the merged sites carry readable labels. The CLI does the same.

## Critical Guardrails & Gotchas

1. **`--references` must cover every element in the selected atoms.**
   If a `.magres` file contains C, H, and O and you pass `--references C:170,H:30` without `-s C,H`, Soprano raises `ValueError: Mismatch between sites and references given. Missing elements: {'O'}`. Always either subset with `-s C,H` or supply references for every present element.
2. **Direct vs linear referencing (`--gradients` / `gradients=`):**
   - `gradients` omitted (`None`): full NMR formula $\delta = (\sigma_{\mathrm{ref}} - \sigma) / (1 - \sigma_{\mathrm{ref}} \times 10^{-6})$.
   - `gradients` provided: linear calibration $\delta = a + m\sigma$ (where $m \approx -1$ from regressing $\delta_{\mathrm{exp}}$ on $\sigma_{\mathrm{calc}}$). If your fit regressed $\sigma_{\mathrm{calc}} = m'\delta_{\mathrm{exp}} + a'$, invert it: `gradient = 1/m'`, `reference = -a'/m'`.
3. **Symmetry reduction:**
   - `soprano nmr` defaults to `--reduce` (merges equivalent sites via CIF labels or symmetry; disable with `--no-reduce`).
   - Symmetry reduction requires a symmetry backend (`pip install soprano[moyo]` recommended, or `soprano[spglib]`). If a crystal with slightly noisy/rounded coordinates is reduced to `P1` at the default `symprec=1e-4`, try `--symprec 1e-3` (or `symprec=1e-3`).
4. **Never use `--mean-merge` unless sites are related purely by translation.**
   Default reduction uses `merge_first` because averaging tensors across rotations/mirrors corrupts principal axes and Euler angles.
5. **Query bare column names, not unit-suffixed headers.**
   In `--query`, use `MS_shielding`, `MS_shift`, `EFG_quadrupolar_constant`, `D` (no units). Output tables/files automatically append units (`MS_shielding/ppm`, `EFG_quadrupolar_constant/MHz`, `D/kHz`).
6. **Watch units across CLI vs Python.**
   - CLI `soprano nmr` prints `EFG_quadrupolar_constant/MHz` and `soprano dipolar` prints `D/kHz`.
   - Python `EFGQuadrupolarConstant.get()`, `DipolarCoupling.get()`, and `DipolarRSS.get()` return **Hz**.
7. **Option naming & import traps:**
   - `soprano dipolar` uses `--select_i` and `--select_j` (**underscores**, not hyphens).
   - `MSShift` and `MSShielding` must be imported from `soprano.properties.nmr.ms` (not top-level `soprano.properties.nmr`).
   - Reading `.magres` files with CIF-style labels requires `ase>=3.26` (ASE 3.23–3.25 had a regression).
8. **Group averaging needs two steps in Python.** `-g CH3` in the CLI is `tag_functional_groups` followed by `merge_tagged_sites`. Calling only `tag_functional_groups` silently leaves the methyl hydrogens separate. In a multi-molecule cell, each site also appears once per molecule, so group the output by label if you want one value per environment. In `soprano nmr`, merging keeps only the first H position, so do not feed merged structures into downstream pair calculations.

## Detailed References

- For full CLI option tables, output formats, column aliases, `UNITS`, and `~/.soprano/config.ini` templates, read [references/cli-reference.md](references/cli-reference.md).
- For the full Python API (`soprano.properties.nmr` and `NMRCalculator` 1D powder spectra), read [references/python-api.md](references/python-api.md).
