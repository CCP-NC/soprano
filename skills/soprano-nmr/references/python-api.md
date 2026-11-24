# Soprano NMR Python API Reference

## 1. NMR `AtomsProperty` Classes (`soprano.properties.nmr`)

Every `AtomsProperty` works on either a single `ase.Atoms` or an `AtomsCollection` via `.get(s, store_array=False, selection=None, **kwargs)`.

### Magnetic Shielding & Chemical Shift (`soprano.properties.nmr` & `.ms`)

| Class | Import path | Returns | Key kwargs |
|---|---|---|---|
| `MSTensor` | `soprano.properties.nmr` | `list[MagneticShielding]` | `order`, `references`, `gradients`, `tag="ms"` |
| `MSIsotropy` | `soprano.properties.nmr` | `np.ndarray` (ppm, raw shielding) | `tag="ms"` |
| `MSShift` | `soprano.properties.nmr.ms` | `np.ndarray` (ppm, chemical shift) | `references` (required), `gradients=None`, `tag="ms"` |
| `MSShielding` | `soprano.properties.nmr.ms` | `np.ndarray` (ppm, raw shielding) | `tag="ms"` |
| `MSAnisotropy` | `soprano.properties.nmr` | `np.ndarray` (ppm, $\Delta = 3\zeta/2$) | `tag="ms"` |
| `MSReducedAnisotropy` | `soprano.properties.nmr` | `np.ndarray` (ppm, $\zeta$) | `tag="ms"` |
| `MSAsymmetry` | `soprano.properties.nmr` | `np.ndarray` ($\eta \in [0, 1]$) | `tag="ms"` |
| `MSSpan` / `MSSkew` | `soprano.properties.nmr` | `np.ndarray` ($\Omega$ ppm / $\kappa$) | `tag="ms"` |
| `MSDiagonal` | `soprano.properties.nmr` | `np.ndarray` of per-atom `dict` (`evals`, `evecs`) | `order`, `tag="ms"` |
| `MSEuler` / `MSQuaternion` | `soprano.properties.nmr` | Euler angles (rad) / quaternions | `convention="zyz"`, `passive=False`, `order` |

> **Important:** `MSShift` and `MSShielding` are defined in `soprano.properties.nmr.ms` and are **not** re-exported in `soprano.properties.nmr.__init__`.

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
calc.set_reference(170.0, "C")
calc.set_powder(N=16, mode="hemisphere")

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

Useful `NMRFlags`: `CS_ISO`, `CS_ORIENT`, `CS`, `Q_1_ORIENT`, `Q_2_SHIFT`, `Q_2_ORIENT_STATIC`, `Q_2_ORIENT_MAS`, `Q_2_STATIC`, `Q_2_MAS`, `Q_STATIC`, `Q_MAS`, `STATIC`, `MAS`.
For coupled multi-spin or exact lineshape simulations, export a spin system via `soprano-spin-systems`.
