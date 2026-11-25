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
gradients = C:-0.98
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
| `--gradients TEXT` | omitted | Chosen **per element**: an element without a gradient uses the full formula; one with a gradient (e.g. `C:-0.97`) uses the linear fit $\delta = a + m\sigma$. A partial dict or an explicit `-1` emits a warning |
| `-i, --isotopes TEXT` | most common NMR-active | e.g. `-i 2H,17O,13C` |
| `-g, --average-group TEXT` | none | Fast-exchange XHn groups: `-g CH3,NH3` |
| `--reduce / --no-reduce` | `--reduce` | Group equivalent sites (CIF labels override symmetry) |
| `--mean-merge / --no-mean-merge` | off | Average tensors (`merge_mean`) instead of keeping the first (`merge_first`) when reducing; only safe for sites related by translation. `-g` groups are still pooled by `--mean-merge` |
| `--euler [zyz\|zxz]` | `zyz` | Euler angle convention |
| `--ms-tag`, `--efg-tag` | `ms`, `efg` | Custom array names (useful for ML-predicted `.xyz` files) |
| `--symprec FLOAT` | `1e-4` | Symmetry tolerance in Å |
| `--precision INT` | `3` | Decimal places in the output |
| `-o, --output FILE`, `-f, --output-format` | stdout | See "Output Formats" above |
| `-m, --merge`, `--sortby`, `--sort-order`, `--include`, `--exclude`, `--query` | — | Table options, see above |
| `--view` | off | View the structure(s) in the ASE GUI (sites coloured by equivalence tag) |
| `-v` / `-vv` | warnings only | Info / debug logging; `-v` also reports each element's referencing mode |

Elements in the selection that have no reference raise a `ValueError` (they are not filled with 0 ppm).

## `soprano dipolar` Key Options

| Option | Default | Notes |
|---|---|---|
| `--select_i TEXT`, `--select_j TEXT` | all | **Underscores** in flag names; cannot use `--select_j` with `--rss` |
| `--rss` | off | Compute per-atom root-sum-squared dipolar coupling instead of pairs |
| `--cutoff FLOAT` | `5.0` | Cutoff radius (Å) for `--rss` |
| `--isonuclear` | off | Restrict to same-element pairs |
| `-i, --isotopes TEXT` | default | e.g. `-i 2H,13C` |
| `-g, --average-group TEXT` | none | Average fast-exchange XHn groups (`CH3`, ...) |
| `--symprec FLOAT` | `1e-4` | Symmetry tolerance in Å |
| `--query TEXT` | none | e.g. `--query "abs(D) > 10.0"` (D is in kHz in the CLI DataFrame) |

## `soprano plotnmr` Key Options

Plots a 1D spectrum (`-p 1D`) or a 2D correlation map (`-p 2D`, default) from a single `.magres` file. `-x/--xelement` is required; `-y/--yelement` defaults to the x element. It shares `-i`, `-g`, `--references`, `--gradients`, `--reduce`, `--symprec` and `--euler` with `soprano nmr` (`-s/--subset` works with `-p 1D` to filter sites of the `-x` element, and is rejected for 2D plots). Invalid option combinations exit with a usage error (code 2).

| Option | Default | Notes |
|---|---|---|
| `-p, --plot-type [2D\|1D]` | `2D` | Plot type |
| `--rcut FLOAT` | none (all pairs) | Cutoff distance (Å) for correlation pairs |
| `--yaxis-order [1Q\|2Q]` | `1Q` | `2Q` plots δ_i + δ_j on the y axis (DQ-SQ) |
| `--weight-by [fixed\|distance\|inversedistance\|dipolar\|dipolar2\|jcoupling\|dipolar_rss]` | `fixed` | Metric weighting each cross-peak (heatmap/contour intensity and, with `--scale-markers`, marker size). `--scale-marker-by` is a legacy alias |
| `--rss-cutoff FLOAT` | `5.0` | Cutoff (Å) for `--weight-by dipolar_rss` |
| `--rss-expand-j [periodic_images\|symmetry\|cif_labels]` | `periodic_images` | How the j-neighbour set is expanded for `dipolar_rss` |
| `--scale-markers / --fixed-markers`, `--marker-symbol`, `-ms/--max-marker-size`, `--marker-color`, `--marker-linewidth`, `--legend`, `--markers / --no-markers` | — | Marker styling |
| `--heatmap / --no-heatmap`, `--contour / --no-contour` | off | Add a broadened heatmap / contour lines |
| `--contour-levels INT` | `10` | Number of contour lines |
| `--intensity-range LO HI` | `10 100` | Percent-of-max range shared by contour and heatmap; override per layer with `--contour-range` / `--heatmap-range` |
| `--heatmap-levels INT`, `--heatmap-grid-size INT`, `--colormap TEXT`, `--contour-color`, `--contour-linewidth`, `--grid-max FLOAT` | — | Heatmap/contour styling and grid scaling |
| `--xbroadening FLOAT`, `--ybroadening FLOAT` | 5 % of range | FWHM (ppm); must be positive |
| `--xlim LO HI`, `--ylim LO HI`, `--grid / --no-grid`, `--diag / --no-diag`, `--connectors / --no-connectors`, `--ticklabels / --no-ticklabels`, `--shielding` | — | Axes and display; `--shielding` plots shielding even when references are given |
| `-o, --output FILE` | window | Plot file (svg or pdf recommended) |
| `--export-file PATH` (repeatable), `--export-format [simpson\|npz\|csv\|json\|plain\|ssnake\|bruker]`, `--overwrite` | — | **Experimental.** Export contour data; the format is inferred from the extension (`.spe`/`.sim`, `.npz`, `.csv`, `.json`) unless `--export-format` is given. A Bruker export writes `<PATH>/1/pdata/1/...`; `--overwrite` is needed to replace an existing dataset. Bruker, ssNake JSON and frequency-axis SIMPSON export require both referenced data (`--references`) and a Larmor frequency (`--spectrometer-freq`, `--b0-field-tesla`, or `--x-larmor-freq`) |
| `--b0-field-tesla T`, `--spectrometer-freq MHz`, `--x-larmor-freq MHz`, `--y-larmor-freq MHz` | — | Larmor frequencies for export only (the plot stays in ppm). `--b0-field-tesla` and `--spectrometer-freq` are mutually exclusive |
