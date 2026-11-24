---
name: soprano-spin-systems
description: Use when building NMR spin systems in Soprano, exporting to SIMPSON or mrsimulator (soprano spinsys, get_spin_system, SpinSystem, Site, Coupling), working with NMRTensor / MagneticShielding / ElectricFieldGradient conventions (Haeberlen, Mehring, Maryland, NQR), converting or comparing Euler angles (passive ZYZ, active ZYZ, mrsimulator_euler_angles, euler_to), or computing fast-exchange dipolar tensor averages (averaged_dipolar_coupling). For routine .magres summary tables or dipolar tables, use soprano-nmr instead. For general crystal structures, collections, or symmetry, use soprano-structures.
---

# Soprano Spin Systems, Tensor Conventions & Simulation Export

Soprano builds structured `SpinSystem` models (`soprano.nmr`) from `.magres` / extended `.xyz` files and exports orientation-consistent spin systems for **SIMPSON** and **mrsimulator** via both `soprano spinsys` (CLI) and `get_spin_system()` (Python).

## Quick-Start Workflows

### CLI (`soprano spinsys`)

> **CLI traps to avoid:** `soprano spinsys` takes a **single** input file and requires `-o <file>` (without `-o`, it does **not** print to stdout). The flags are `--references` (not `--ref`) and `--select_i` / `--select_j` (**underscores**, not hyphens).

```bash
# 13C/1H spin system with C-H dipolar couplings -> SIMPSON
soprano spinsys structure.magres -s C,H --references C:170,H:30 \
  --dip --select_i C --select_j H --obs 13C -o spinsys.in

# Quadrupolar 2H spin system (2nd order) -> SIMPSON
soprano spinsys structure.magres -s H -i 2H --references H:30 \
  --q-order 2 -o spinsys_2H.in

# 17O quadrupolar only (suppress shielding) -> SIMPSON
soprano spinsys structure.magres -s O -i 17O --no-ms --q-order 2 -o spinsys_17O.in

# Export to mrsimulator JSON (with dipolar couplings)
soprano spinsys structure.magres -s C,H --references C:170,H:30 \
  --dip --select_i C --select_j H -f mrsimulator -o spinsys.json

# Average fast-rotating CH3 groups (use --no-reduce so equivalent methyls stay separate)
soprano spinsys structure.magres -s H --references H:30 \
  --average-group CH3 --no-reduce -o spinsys_ch3.in
```

### Python (`get_spin_system` & `SpinSystem`)

```python
from ase.io import read
from soprano.properties.nmr import get_spin_system
from soprano.selection import AtomSelection

atoms = read("structure.magres")
sub = AtomSelection.from_selection_string(atoms, "C,H").subset(atoms)

ss = get_spin_system(
    sub,
    references={"C": 170.0, "H": 30.0},
    gradients=None,  # None = full NMR formula; pass float/dict for linear δ = ref + m*σ
    include_shielding=True,
    include_efg=True,
    include_dipolar=True,
    include_j=False,
    coupling_kwargs={"sel_i": "C", "sel_j": "H"},
)

# SIMPSON spinsys block (string or file)
simpson_str = ss.to_simpson(observed_nucleus="13C", q_order=2)
ss.write("spinsys.in", format="simpson", observed_nucleus="13C")

# mrsimulator dict or JSON file
mrsim_dict = ss.to_mrsimulator()
ss.write("spinsys.json", format="mrsimulator")
```

## Non-Negotiable Physics & Convention Guardrails

Multi-tensor solid-state NMR simulations silently produce wrong spectra when Euler conventions, anisotropy signs, or powder integration settings are mixed. Follow these rules strictly:

1. **SIMPSON and mrsimulator use DIFFERENT Euler angle conventions — never copy angles between them.**
   - **SIMPSON** uses **passive ZYZ** in **degrees**: $E = R_z(-\gamma)R_y(-\beta)R_z(-\alpha)$ → `tensor.euler_angles(passive=True, degrees=True)`.
   - **mrsimulator** uses **extrinsic ZYZ** in **radians**: $E = R_z(\gamma)R_y(\beta)R_z(\alpha)$ → `tensor.mrsimulator_euler_angles()` (or `SpinSystem.to_mrsimulator()`).
   - **Do NOT implement mrsimulator paper Appendix B equations** (they swap $\alpha$ and $\gamma$ relative to mrsimulator's actual code and give wrong multi-tensor spectra). Always use Soprano's `mrsimulator_euler_angles()` / `to_mrsimulator()`.
2. **Shielding vs shift anisotropy sign differs between programs.**
   - Soprano stores raw **magnetic shielding** ($\sigma$). Exporting chemical shifts requires `references` (otherwise `to_simpson()` / `to_mrsimulator()` raises `ValueError` unless shielding is disabled).
   - SIMPSON `shift` `aniso` is **shift-sense** reduced anisotropy ($\zeta_\delta$ in ppm with `p` suffix).
   - mrsimulator `zeta` is **shielding-sense** reduced anisotropy ($\zeta_\sigma = -\zeta_\delta$). Soprano converts both automatically.
3. **Always set `integration_volume = "hemisphere"` in mrsimulator for multi-tensor systems.**
   mrsimulator defaults to `"octant"`, which is only exact when all tensors share the first tensor's symmetry. For CSA + dipolar, CSA + J, or CSA + EFG in general relative orientations, `"octant"` gives wrong lineshapes regardless of Euler angles.
4. **SIMPSON-specific scalings (handled automatically by Soprano — do NOT double-apply):**
   - **CSA:** `aniso` is reduced anisotropy $\zeta$ (NOT full anisotropy $\Delta = 3\zeta/2$).
   - **Dipolar:** `d` is in Hz ($d = -\frac{\mu_0\hbar\gamma_1\gamma_2}{8\pi^2 r^3}$, negative for like-sign $\gamma$) with **NO extra $2\pi$ factor**.
   - **Anisotropic J-coupling (SIMPSON v6.0.2):** `aniso` is $\zeta / 2$ in Hz due to SIMPSON's internal Hamiltonian scaling.
   - **Second-order cross terms:** `to_simpson(..., include_cross_terms=True)` (default) automatically writes `quadrupole_x_dipole` and `quadrupole_x_shift` lines when quadrupolar sites are present.
5. **Fast-exchange dipolar averaging (e.g. rotating CH₃ / NH₃):**
   - Never take one representative H or average scalar dipolar constants $|d_{ij}|$ (overestimates by 17–30%+).
   - Average the full $3\times 3$ dipolar tensors on the **unmerged** structure with `soprano.properties.nmr.averaged_dipolar_coupling(atoms, sel_i, sel_j)`.
6. **Tensor averaging & `--reduce`:**
   - Default `--reduce` uses `merge_first` (safe). Do not pass `--mean-merge` for non-translation symmetries (rotations/mirrors corrupt tensor orientations).
   - `--split` in `soprano spinsys` only works for single-species spin systems with **no couplings**.
7. **Relative angles & default eigenvalue orders:**
   - `A.euler_to(B)` gives the orientation of tensor **B** in the PAS of **A** ($E_A^{\mathsf T}E_B$), matching TensorView's `RelativeAngles(Tensor1=B, Tensor2=A)`.
   - `euler_to` depends on the eigenvalue ordering of **both** tensors, and Soprano's defaults differ (`MSTensor.get()` defaults to `order="i"`, `MagneticShielding`/`MSEuler` default to `order="h"`, `EFGTensor`/`EFGEuler` default to `order="n"`). Always set explicit `order=` before comparing orientations.
8. **Low-level spin-system assembly & J-coupling class name:**
   For custom spin systems, use `get_sites`, `get_dipolar_couplings`, `get_j_couplings` (`soprano.properties.nmr`), and `Site`, `DipolarCoupling`, `ISCoupling` (`soprano.nmr.coupling` — note the class is `ISCoupling`, not `JCoupling`). Site indices in `Coupling` are relative to the **subset** `Atoms` object passed in, not the original unit cell.

## Working Directly with `NMRTensor` Objects

```python
from ase.io import read
from soprano.properties.nmr import MSTensor, EFGTensor

atoms = read("structure.magres")
ms_list = MSTensor.get(atoms, references={"C": 170.0, "H": 30.0, "O": 250.0})
efg_list = EFGTensor.get(atoms, isotopes={"H": 2})

t_ms = ms_list[0]
t_efg = efg_list[0]

# Conventions (Haeberlen, Mehring, Maryland)
h_shift = t_ms.haeberlen_shift        # isotropy, reduced_anisotropy, asymmetry, ...
m_shield = t_ms.mehring_shielding

# Orientation in lab/magres frame
ang_simpson = t_ms.euler_angles(convention="zyz", passive=True, degrees=True)
ang_mrsim = t_ms.mrsimulator_euler_angles(degrees=False)

# Relative orientation of EFG PAS in MS PAS
rel_euler = t_ms.euler_to(t_efg, convention="zyz", passive=False)
```

## Detailed References

- For full mathematical definitions of Haeberlen/Mehring/Maryland/NQR conventions, active vs passive ZYZ Euler matrices, axial gauges, referencing formulas, and the SIMPSON v6.0.2 vs mrsimulator parameter table, read [references/conventions.md](references/conventions.md).
- For complete `soprano spinsys` CLI recipes, programmatic `SpinSystem` / `Site` / `Coupling` construction, `averaged_dipolar_coupling`, and runnable SIMPSON / mrsimulator handoff snippets, read [references/export-and-handoff.md](references/export-and-handoff.md).
