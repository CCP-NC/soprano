# Soprano NMR Python API Reference

## 1. NMR `AtomsProperty` Classes (`soprano.properties.nmr`)

Every `AtomsProperty` works on either a single `ase.Atoms` or an `AtomsCollection` via `.get(s, store_array=False, selection=None, **kwargs)`.

### Magnetic Shielding & Chemical Shift (`soprano.properties.nmr` & `.ms`)

| Class | Import path | Returns | Key kwargs |
|---|---|---|---|
| `MSTensor` | `soprano.properties.nmr` | `list[MagneticShielding]` | `order`, `references`, `gradients`, `tag="ms"` |
| `MSIsotropy` | `soprano.properties.nmr` | `np.ndarray` (ppm, raw shielding) | `tag="ms"` |
| `MSShift` | `soprano.properties.nmr` | `np.ndarray` (ppm, chemical shift) | `references` (required), `gradients=None`, `tag="ms"` |
| `MSShielding` | `soprano.properties.nmr` | `np.ndarray` (ppm, raw shielding) | `tag="ms"` |
| `MSAnisotropy` | `soprano.properties.nmr` | `np.ndarray` (ppm, $\Delta = 3\zeta/2$) | `tag="ms"` |
| `MSReducedAnisotropy` | `soprano.properties.nmr` | `np.ndarray` (ppm, $\zeta$) | `tag="ms"` |
| `MSAsymmetry` | `soprano.properties.nmr` | `np.ndarray` ($\eta \in [0, 1]$) | `tag="ms"` |
| `MSSpan` / `MSSkew` | `soprano.properties.nmr` | `np.ndarray` ($\Omega$ ppm / $\kappa$) | `tag="ms"` |
| `MSDiagonal` | `soprano.properties.nmr` | `np.ndarray` of per-atom `dict` (`evals`, `evecs`) | `order`, `tag="ms"` |
| `MSEuler` / `MSQuaternion` | `soprano.properties.nmr` | Euler angles (rad) / quaternions | `convention="zyz"`, `passive=False`, `order` |

> **Note:** `MSShift`, `MSShielding` and the `JC*` properties (`JCTensor`, `JCAnisotropy`, `JCReducedAnisotropy`, `JCAsymmetry`, `JCSpan`, `JCSkew`, `JCQuaternion`) are re-exported from `soprano.properties.nmr`, like the other NMR properties. They are defined in `.ms` / `.isc`, and those paths still work.
>
> **Referencing:** `gradients=None` (default) uses the full formula for every element; an element present in `gradients` uses the linear fit $\delta = \sigma_{\mathrm{ref}} + m\sigma$ (chosen per element). `MagneticShielding(gradient=...)` also defaults to `None` (it was `-1.0` before v0.12). When `references` is passed to `MSTensor.get` or `MSShift.get`, it must cover every element present in `atoms` (or pass `selection=...` to restrict the sites).

### Electric Field Gradient (`soprano.properties.nmr`)

Default EFG eigenvalue ordering is NQR (`order="n"`, $|V_{xx}| \le |V_{yy}| \le |V_{zz}|$).

| Class | Returns | Key kwargs |
|---|---|---|
| `EFGTensor` | `list[ElectricFieldGradient]` | `order="n"`, `use_q_isotopes=False`, `isotopes={}`, `isotope_list=None`, `tag="efg"` |
| `EFGVzz` | `np.ndarray` (atomic units) | `tag="efg"` |
| `EFGQuadrupolarConstant` | `np.ndarray` ($C_q$ in **Hz**) | `use_q_isotopes=False`, `isotopes={}`, `isotope_list=None`, `tag="efg"` |
| `EFGQuadrupolarProduct` | `np.ndarray` ($P_Q$ in **Hz**) | `use_q_isotopes=False`, `isotopes={}`, `isotope_list=None`, `tag="efg"` |
| `EFGAsymmetry` | `np.ndarray` ($\eta_Q$) | `tag="efg"` |
| `EFGNQR` | `dict` (transition freqs in **Hz**) | `use_q_isotopes=False`, `isotopes={}`, `isotope_list=None`, `tag="efg"` |
| `EFGAnisotropy`, `EFGReducedAnisotropy`, `EFGSpan`, `EFGSkew`, `EFGDiagonal`, `EFGEuler`, `EFGQuaternion` | standard tensor invariants/orientations | `tag="efg"` |

> **Tip:** Default isotopes are the most common NMR-active ones (e.g. `1H`, `13C` are spin-$1/2$, so $C_q = 0$). Pass `isotopes={"H": 2}` or `use_q_isotopes=True` to get nonzero quadrupolar couplings for elements whose primary NMR isotope is spin-$1/2$.

### Dipolar & Indirect (J) Couplings (`soprano.properties.nmr`)

| Class / Function | Returns | Key kwargs |
|---|---|---|
| `DipolarCoupling` | `dict[(i, j)] -> (d_Hz, unit_vec)` | `sel_i`, `sel_j`, `isotopes={}`, `isotope_list=None`, `self_coupling=False`, `isonuclear=False` |
| `DipolarRSS` | `np.ndarray` (Hz, per atom) | `cutoff=5.0`, `isonuclear=False`, `isotopes={}`, `isotope_list=None` |
| `DipolarRSSByAtom` | `dict[(i, j)] -> float` (Hz) | `sel_i`, `sel_j`, `cutoff=5.0`, `expand_j="periodic_images"`, `symprec=1e-4` |
| `DipolarTensor` / `DipolarEuler` / `DipolarDiagonal` | tensor / Euler / eigenpairs per pair | `sel_i`, `sel_j`, `isotopes={}` |
| `averaged_dipolar_coupling(atoms, sel_i, sel_j, ...)` | `(d_eff_Hz, D_avg_3x3)` | Fast-exchange tensor average (see `soprano-spin-systems`) |
| `averaged_dipolar_coupling_list(atoms, site_sets, ...)` / `averaged_j_coupling_list(atoms, site_sets, ...)` | `list[DipolarCoupling]` / `list[ISCoupling]` | Per-site-pair motional averages for spin-system export (`site_sets[k]` = atom indices of site `k`) |
| `JCIsotropy` / `JCDiagonal` / `JCEuler` | `dict[(i, j)] -> ...` (Hz) | `sel_i`, `sel_j`, `isotopes={}`, `tag="isc"` |

## 2. Fast 1D Diluted-Atom Powder Spectra (`NMRCalculator`)

For a fast approximated 1D MAS or static powder spectrum (diluted-atom approximation, no spin-spin couplings):

```python
from ase.io import read
from soprano.calculate.nmr import NMRCalculator, NMRFlags

atoms = read("structure.magres")
calc = NMRCalculator(atoms, larmor_frequency=400, larmor_units="MHz")
calc.set_larmor_frequency(400.0, element="1H")       # sets B0 from any nucleus
b0_tesla = calc.B                                    # field in Tesla
freq_13c = calc.get_larmor_frequency("13C")          # Larmor frequency in MHz
calc.set_reference(170.0, "C")                       # gradient=None: full formula; gradient=m: linear fit
calc.set_powder(N=16, mode="hemisphere")             # optional: the default is already set_powder(N=8)

# Combine flags with bitwise OR: e.g. NMRFlags.CS (static CSA) or NMRFlags.MAS
spec, freqs = calc.spectrum_1d(
    "C",
    min_freq=0,
    max_freq=200,
    bins=500,
    freq_broad=1.0,
    freq_units="ppm",
    effects=NMRFlags.CS_ISO,
)
```

Behaviour to know:

- A new `NMRCalculator` uses a powder average by default (`set_powder(N=8)`, hemisphere), so isotropic spectra work without calling `set_powder()` or `set_single_crystal()` first.
- With a reference set, `min_freq` / `max_freq` bound the **chemical-shift** axis (same units and scale as the returned axis), not the shielding axis.
- `freq_broad` is in `freq_units` (ppm or MHz), not always ppm.
- `set_reference(ref, element, gradient=None)` follows the same referencing as `MSShift`: full formula by default, linear fit when `gradient` is given.
- `set_element_isotope` accepts isotope symbols (`'17O'`) as well as mass numbers (`17`).

Useful `NMRFlags`: `CS_ISO`, `CS_ORIENT`, `CS`, `Q_1_ORIENT`, `Q_2_SHIFT`, `Q_2_ORIENT_STATIC`, `Q_2_ORIENT_MAS`, `Q_2_STATIC`, `Q_2_MAS`, `Q_STATIC`, `Q_MAS`, `STATIC`, `MAS`.
For coupled multi-spin or exact lineshape simulations, export a spin system via `soprano-spin-systems`.

## 3. 2D Correlation Spectra & Experimental Export (`soprano.calculate.nmr`)

```python
from ase.io import read
from soprano.calculate.nmr import ExportConfig, NMRData2D, NMRPlot2D, PlotSettings

atoms = read("ethanol.magres")
data = NMRData2D(
    atoms,
    xelement="C",
    yelement="H",
    references={"C": 170.0, "H": 30.0},
    correlation_strength_metric="dipolar",
    rcut=3.0,
    reduce=True,
)
settings = PlotSettings(show_heatmap=True, show_contour=False, colormap="grey_r")
plotter = NMRPlot2D(data, settings=settings)
fig, ax = plotter.plot()  # returns (fig, ax) for matplotlib; fig for plotly
fig.savefig("ch_hetcor.png", dpi=150)

# Experimental export (Bruker TopSpin requires `pip install soprano[nmr-io]`):
data.export_contour_data(
    "ch_hetcor.json",  # or a Bruker directory with fmt="bruker"
    config=ExportConfig(spectrometer_freq_mhz=400.0),
)
```

Key notes:
- Import `NMRData2D`, `NMRPlot2D`, `PlotSettings`, and `ExportConfig` from `soprano.calculate.nmr` (not `soprano.properties.nmr`).
- `NMRPlot2D(...).plot()` returns `(fig, ax)` with the default matplotlib backend.
- Bruker, ssNake JSON, and frequency-axis SIMPSON exports require both referenced shifts (`references=...`) and a Larmor frequency (`spectrometer_freq_mhz`, `b0_field_tesla`, or `x_larmor_freq_mhz`).
