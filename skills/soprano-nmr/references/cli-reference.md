# Soprano NMR CLI Reference

## Config File Defaults (`~/.soprano/config.ini`)

`soprano nmr` and `soprano dipolar` load default option values from an INI file in this precedence order:
1. CLI flag (highest priority)
2. `-c / --config FILE`
3. Environment variable `SOPRANO_CONFIG`
4. `~/.soprano/config.ini`

Section names follow `[soprano.<subcommand>]`, using Click parameter names (underscores):

```ini
[soprano.nmr]
references = C:170.0,H:30.5
gradients = C:-0.98,H:-1.0
precision = 2
symprec = 1e-4
```

## Output Formats & Table Options (`soprano nmr` & `soprano dipolar`)

Supported `-f / --output-format` values (also inferred from `-o` extension):
`csv`, `json`, `html`, `md` (requires `tabulate`), `tex`, `txt`, `tsv`, `dat`, `xls`, `xlsx`.

- `-m / --merge`: concatenate DataFrames from multiple input files into one output table (adds `file` column).
- `--sortby COL --sort-order [ascending|descending]`: sort rows by a bare column name.
- `--include` / `--exclude`: comma-separated bare column names or aliases.
  - `soprano nmr` aliases: `minimal` (`MS_shielding`, `MS_anisotropy`, `EFG_quadrupolar_constant`, `EFG_asymmetry`), `ms`, `efg`, `angles` (`alpha`, `beta`, `gamma`), `essential` (`labels`, `species`, `multiplicity`, `tags`, `file`).
  - `soprano dipolar` aliases: `minimal`, `essential`.
- `--query "EXPR"`: pandas DataFrame query using **bare column names without units**.

### Printed / Exported Column Units (`UNITS`)

| Bare name (for `--query` / `--include` / `--sortby`) | Printed header |
|---|---|
| `MS_shielding`, `MS_shift`, `MS_anisotropy`, `MS_reduced_anisotropy`, `MS_span` | `<name>/ppm` |
| `MS_alpha`, `MS_beta`, `MS_gamma`, `EFG_alpha`, `EFG_beta`, `EFG_gamma`, `alpha`, `beta`, `gamma` | `<name>/deg` |
| `EFG_Vzz` | `EFG_Vzz/Vm^-2` |
| `EFG_quadrupolar_constant`, `EFG_NQR` | `<name>/MHz` |
| `D` | `D/kHz` |

## Selection String Syntax (`-s`, `--select_i`, `--select_j`)

| Pattern | Meaning |
|---|---|
| `C` | All carbon atoms |
| `C,N` | All carbon and nitrogen atoms |
| `C.1-4` | Carbons 1 through 4 (1-based within species) |
| `C.1.3,H.2-5` | Specific element indices |
| `H1a,C2` | Specific CIF / MagresView site labels |

## `soprano nmr` Key Options

| Option | Default | Notes |
|---|---|---|
| `-p, --properties [efg\|ms]` | both | Repeatable: `-p ms -p efg` |
| `-s, --subset TEXT` | all | Selection string |
| `--references TEXT` | none | `C:170,H:30` — must cover every element in selection |
| `--gradients TEXT` | omitted | Omit for full formula; supply for linear $\delta = a + m\sigma$ |
| `-i, --isotopes TEXT` | most common NMR-active | e.g. `-i 2H,17O,13C` |
| `-g, --average-group TEXT` | none | Fast-exchange XHn groups: `-g CH3,NH3` |
| `--reduce / --no-reduce` | `--reduce` | Group equivalent sites (CIF labels override symmetry) |
| `--euler [zyz\|zxz]` | `zyz` | Euler angle convention |
| `--ms-tag`, `--efg-tag` | `ms`, `efg` | Custom array names (useful for ML-predicted `.xyz` files) |
| `--symprec FLOAT` | `1e-4` | Symmetry tolerance in Å |

## `soprano dipolar` Key Options

| Option | Default | Notes |
|---|---|---|
| `--select_i TEXT`, `--select_j TEXT` | all | **Underscores** in flag names; cannot use `--select_j` with `--rss` |
| `--rss` | off | Compute per-atom root-sum-squared dipolar coupling instead of pairs |
| `--cutoff FLOAT` | `5.0` | Cutoff radius (Å) for `--rss` |
| `--isonuclear` | off | Restrict to same-element pairs |
| `-i, --isotopes TEXT` | default | e.g. `-i 2H,13C` |
| `--query TEXT` | none | e.g. `--query "abs(D) > 10.0"` (D is in kHz in the CLI DataFrame) |
