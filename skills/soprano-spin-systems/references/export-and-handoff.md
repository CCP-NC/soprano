# Spin System Export, Dipolar Averaging & Simulator Handoff

## 1. `soprano spinsys` CLI Options & Recipes

`soprano spinsys` accepts a **single** `.magres`, `.extxyz`, or `.xyz` file (`nargs=1`).

| Option | Default | Notes |
|---|---|---|
| `-o, --output FILE` | `None` | **Always specify `-o`** (without it, nothing is printed to stdout) |
| `-f, --format [simpson\|mrsimulator]` | `simpson` | Output format |
| `-s, --subset TEXT` | all | Selection string (e.g. `C,H`, `C.1-4`, `H1a`) |
| `--references TEXT` | none | Required when `--ms` is enabled; must include every species in subset |
| `--gradients TEXT` | omitted | Linear calibration slopes (e.g. `C:-0.97`); chosen per element, elements without one use the full formula |
| `-i, --isotopes TEXT` | default | e.g. `-i 2H,17O` (controls dipolar, EFG, and J; does not affect MS) |
| `--obs, --observed-nucleus TEXT` | first sorted | Sets first channel in SIMPSON `channels` line (e.g. `--obs 13C`) |
| `--ms / --no-ms`, `--efg / --no-efg` | on | Include MS / EFG interactions |
| `--dip / --no-dip`, `--jcoupling / --no-jcoupling` | off | Opt-in dipolar and J couplings |
| `--select_i TEXT`, `--select_j TEXT` | all | Restrict coupling pairs (**underscores**, e.g. `--select_i C --select_j H`) |
| `--q-order [1\|2]` | `2` if active | Quadrupolar perturbation order (SIMPSON) |
| `--cross-terms / --no-cross-terms` | on | Emit `quadrupole_x_dipole` / `quadrupole_x_shift` (SIMPSON) |
| `--ms-iso` | off | Drop CSA anisotropy/angles (isotropic shift only) |
| `--angles [all\|none\|default]` | `default` | Global angle toggle; fine-grained `--[no-]{ms,efg,dipolar,jcoupling}-angles` |
| `--reduce / --no-reduce` | `--reduce` | Reduce equivalent sites; use `--no-reduce` with `--average-group CH3` |
| `-g, --average-group TEXT` | none | Average fast-rotating XHn groups (`CH3`, `NH3`) |
| `--expand-group / --collapse-group` | auto | How an averaged group is exported. Expanded: each atom stays a spin with group-mean tensors and tensor-averaged couplings (including intra-group). Collapsed: one spin per group. Default is expand when `--dip` or `--jcoupling` is given, collapse otherwise |
| `--split` | off | Write one file per site (`base_0.in`, ...); only valid for 1 species & 0 couplings |
| `--ms-tag`, `--efg-tag` | `ms`, `efg` | Custom array names for `.extxyz`/`.xyz` inputs |

## 2. Programmatic `SpinSystem` Construction & Editing

```python
from ase.io import read
from soprano.properties.nmr import get_spin_system
from soprano.selection import AtomSelection

atoms = read("structure.magres")
sub = AtomSelection.from_selection_string(atoms, "C,H").subset(atoms)

ss = get_spin_system(
    sub,
    isotopes={"C": 13, "H": 1},
    references={"C": 170.0, "H": 30.0},
    gradients=None,
    include_shielding=True,
    include_efg=False,
    include_dipolar=True,
    include_j=False,
    coupling_kwargs={"sel_i": "C", "sel_j": "H"},
)

# Inspect sites and couplings
print(ss.n_sites, ss.n_couplings, ss.isotope_set)
site0 = ss.sites[0]
print(site0.label, site0.isotope, site0.ms.shift)

# Filter out weak dipolar couplings (< 1 kHz)
# Construct a new SpinSystem so Pydantic rebuilds coupling_indices (avoid .copy(update=...)!)
from soprano.nmr import SpinSystem
strong = [c for c in ss.couplings if abs(c.coupling_constant) >= 1000.0]
ss_pruned = SpinSystem(sites=ss.sites, couplings=strong)
pair_couplings = ss_pruned.get_couplings(0, 6)  # lookup by (site_i, site_j)

# Export strings or files
simpson_block = ss_pruned.to_simpson(observed_nucleus="13C", include_angles=True)
mrsim_dict = ss_pruned.to_mrsimulator(include_angles=True)
```

> **Notes:**
> - `get_spin_system` pops `sel_i`/`sel_j` from `coupling_kwargs` during dipolar extraction (so pass a fresh dict if reusing kwargs, and note `sel_i`/`sel_j` do not filter J-couplings).
> - All `site_i`/`site_j` indices are relative to the **subset** `Atoms` object passed in, not the parent crystal.
> - **J-couplings on subsets:** `.magres` `isc` tensors are keyed by original crystal indices, so `get_spin_system(sub, include_j=True)` on a subset `Atoms` raises `KeyError`. Extract J-couplings from the full `atoms` via `get_j_couplings(atoms, sel_i=sel, sel_j=sel)` and remap `site_i`/`site_j` to subset indices before combining into `SpinSystem(sites=..., couplings=...)`.

### Low-Level & Manual Construction (`get_sites`, `Site`, `DipolarCoupling`, `ISCoupling`)

Use the modular helpers (`get_sites`, `get_dipolar_couplings`, `get_j_couplings`) or build `Site` and `Coupling` objects directly when combining DFT tensors with manual/synthetic values:

```python
import numpy as np
from soprano.nmr import Site, SpinSystem, NMRTensor, MagneticShielding
from soprano.nmr.coupling import DipolarCoupling as SiteDipolarCoupling, ISCoupling
from soprano.properties.nmr import get_sites, get_dipolar_couplings, get_j_couplings

# Modular extraction from a subset Atoms object
sites = get_sites(sub, isotopes={"C": 13, "H": 1})
dips = get_dipolar_couplings(sub, sel_i=[0, 1], sel_j=[6, 7])
# js = get_j_couplings(sub)  # when .magres contains 'isc' J-coupling tensors

# Manual Site + ISCoupling (NOTE: ISCoupling is the canonical class name; JCoupling is only an alias for it;
# tensor is the reduced K tensor in units of 10^19 T^2 J^-1, scaled by gamma1*gamma2)
s0 = Site(index=0, label="C1", isotope="13C", ms=MagneticShielding(np.diag([120, 150, 180]), species="C", reference=170.0))
s1 = Site(index=1, label="H1", isotope="1H", ms=MagneticShielding(np.diag([25, 28, 31]), species="H", reference=30.0))
j_cpl = ISCoupling(site_i=0, site_j=1, species1="13C", species2="1H", tensor=NMRTensor(np.eye(3) * 1.0))
d_cpl = SiteDipolarCoupling(
    site_i=0, site_j=1, species1="13C", species2="1H",
    # coupling_constant is half the Haeberlen reduced anisotropy, so it does not depend on the tensor's order
    tensor=NMRTensor(np.diag([20000.0, 20000.0, -40000.0]), order="n"),  # axial (-d, -d, 2d) for d = -20 kHz
)
# Or build from a distance vector in Angstroms:
# d_cpl = SiteDipolarCoupling.from_distance_vector(np.array([0.0, 0.0, 1.09]), "13C", "1H", site_i=0, site_j=1)
# Or from a structure:
# t_dip = NMRTensor.make_dipolar(sub, i=0, j=6)
manual_ss = SpinSystem(sites=[s0, s1], couplings=[d_cpl, j_cpl])
```

## 3. Fast-Exchange Dipolar Tensor Averaging (`averaged_dipolar_coupling`)

For fast-rotating groups (such as CH₃ at almost all experimental NMR temperatures), average the full $3\times 3$ dipolar tensors on the **original unmerged structure**:

```python
from ase.io import read
from soprano.properties.nmr import averaged_dipolar_coupling

atoms = read("ethanol.magres")
methyl_h = [0, 1, 2]

# External proton (index 3) coupled to rotating CH3 group
d_eff_ext, D_avg_ext = averaged_dipolar_coupling(atoms, [3], methyl_h)
print(f"External H to CH3: {d_eff_ext:.1f} Hz")  # signed effective constant

# Intra-methyl residual coupling (P2(cos 90°) = -1/2 emerges automatically)
d_eff_intra, D_avg_intra = averaged_dipolar_coupling(atoms, methyl_h, methyl_h)
print(f"Intra-CH3 residual: {d_eff_intra:.1f} Hz")
```

Notes:
- Uses minimum-image periodic boundaries (honouring `atoms.pbc`) and skips self-pairs (`i == j`); coincident atoms raise.
- For whole spin systems, `averaged_dipolar_coupling_list(atoms, site_sets, ...)` and `averaged_j_coupling_list(atoms, site_sets, ...)` (`soprano.properties.nmr`) return per-site-pair averaged couplings, where `site_sets[k]` lists the atom indices site `k` stands for. `soprano spinsys --average-group ... --expand-group` and `nmr_extract_atoms(..., expand_groups=True)` use them.
- Passing the same group for both `sel_i` and `sel_j` computes the intra-group residual average.
- Only valid in the **fast-exchange** limit ($k_{\mathrm{ex}} \gg |d|$); do not apply to slow/intermediate exchange regimes.

## 4. Short Simulator Handoff Snippets

### Handing off to `mrsimulator`

```python
from mrsimulator import Simulator, SpinSystem as MrSpinSystem
from mrsimulator.method.lib import BlochDecaySpectrum

# Convert Soprano SpinSystem -> mrsimulator SpinSystem
mrsim_ss = MrSpinSystem.parse_dict_with_units(ss.to_mrsimulator())
# Or from JSON exported by `soprano spinsys -f mrsimulator -o spinsys.json`:
# import json; mrsim_ss = MrSpinSystem.parse_dict_with_units(json.load(open("spinsys.json")))

method = BlochDecaySpectrum(
    channels=["13C"],
    magnetic_flux_density=9.4,  # Tesla (400 MHz 1H)
    rotor_frequency=0,          # Hz (static; set >0 for MAS)
    spectral_dimensions=[{"count": 2048, "spectral_width": 5e4, "reference_offset": 0}],
)

sim = Simulator(spin_systems=[mrsim_ss], methods=[method])
# CRITICAL for multi-tensor systems with arbitrary relative orientations:
sim.config.integration_volume = "hemisphere"
sim.run()
```

### Handing off to SIMPSON

Save the `spinsys { ... }` block to `spinsys.in` (via `soprano spinsys ... -o spinsys.in` or `ss.write("spinsys.in", format="simpson", observed_nucleus="13C")`), then wrap or `source` it in a SIMPSON Tcl script:

```tcl
source spinsys.in

par {
    proton_frequency 400e6
    spin_rate        0
    crystal_file     rep168
    gamma_angles     1
    sw               50000
    np               1024
    start_operator   I1x
    detect_operator  I1p
    method           direct
}

proc pulseq {} {
    global par
    acq_block {
        delay [expr 1.0e6/$par(sw)]
    }
}

proc main {} {
    global par
    set f [fsimpson]
    faddlb $f 100 0
    fft $f
    fsave $f spectrum.spe
}
```
