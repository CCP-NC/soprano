# SIMPSON Tensor Conventions

This page documents the conventions used by [SIMPSON](https://inano.au.dk/about/research-centers-and-projects/nmr/software/simpson) for representing NMR interaction tensors in `.spinsys` files, and how Soprano maps its internal tensors to them.

The conventions described below have been verified for **[SIMPSON v6.0.2](https://gitlab.au.dk/nmr/simpson)** through source-code inspection and numerical validation.  Internal implementation details may change in future SIMPSON releases; users should revalidate version-specific behaviour where noted.

## Tensor and Sign Conventions

Unless otherwise stated, anisotropy parameters are expressed using the **Haeberlen convention**, with reduced anisotropy

$$
\zeta = \delta_{ZZ} - \delta_{\mathrm{iso}}
$$

and asymmetry parameter

$$
\eta = \frac{\delta_{YY}-\delta_{XX}}{\zeta},
$$

where $|\delta_{ZZ}-\delta_{\mathrm{iso}}| \ge |\delta_{XX}-\delta_{\mathrm{iso}}| \ge |\delta_{YY}-\delta_{\mathrm{iso}}|$.

SIMPSON uses **right-handed passive ZYZ Euler angles** (in degrees) to specify the orientation of the interaction principal-axis system ($X$, $Y$, $Z$) relative to the molecular/crystal frame.

## Overview

| Interaction | SIMPSON keyword | Physical quantity | SIMPSON parameter | Soprano output |
|-------------|-----------------|-------------------|-------------------|----------------|
| Isotropic shift | `shift` | $\delta_\text{iso}$ (ppm) | `iso` (ppm, `p` suffix) | $\delta_\text{iso}$ with `p` suffix |
| CSA anisotropy | `shift` | Reduced anisotropy $\zeta$ (ppm) | `aniso` (ppm, `p` suffix) | $\zeta$ with `p` suffix |
| Quadrupolar | `quadrupole` | $C_q$ (Hz) | `Cq` (Hz) | $C_q$ (Hz) |
| Dipolar coupling | `dipole` | Coupling constant $d$ (Hz) | `aniso` (Hz) | $d$ (Hz) |
| Isotropic J-coupling | `jcoupling` | $J_\text{iso}$ (Hz) | `iso` (Hz) | $J_\text{iso}$ (Hz) |
| Anisotropic J-coupling | `jcoupling` | Reduced anisotropy $\zeta$ (Hz) | `aniso` (Hz) | **$\zeta/2$** (Hz, SIMPSON v6.0.2) |

**Note:** 

For J-coupling anisotropy, the `aniso` parameter must be **half the reduced anisotropy** ($\zeta/2$).  This seems to be an implementation-specific scaling of how SIMPSON assembles the bilinear Hamiltonian (see {ref}`jcoupling-aniso-section`).

---

## Chemical Shift

### Spinsys format

```
shift n iso_ppm aniso_ppm eta alpha beta gamma
```

* `iso_ppm` and `aniso_ppm` are given in **ppm with a `p` suffix** (e.g. `100.0p`).
* SIMPSON converts ppm values to frequency offsets using the **Larmor frequency of the observed nucleus**, derived from the specified proton reference frequency and nuclear gyromagnetic ratio.

### ppm → Hz conversion

For a nucleus with gyromagnetic ratio $\gamma$ on a spectrometer with proton frequency $\nu_\text{ref}$:

$$
\nu_L = \frac{|\gamma|}{\gamma_\text{H}} \, \nu_\text{ref}
$$

A shift of $\delta$ ppm corresponds to a frequency offset of:

$$
\Delta\nu = \delta \times 10^{-6} \times \nu_L
$$

**Example:** For $^{13}$C at a 400 MHz proton spectrometer:

$$
\nu_C = \frac{\gamma_C}{\gamma_H} \times 400\,\text{MHz} \approx 100.60\,\text{MHz}
$$

So 100 ppm for $^{13}$C = $100 \times 10^{-6} \times 100.60\,\text{MHz} \approx$ **10,060 Hz**.

### CSA anisotropy

The `aniso` parameter corresponds to the **reduced anisotropy** ($\zeta$).  It is **not** the full span-like anisotropy $\Delta = 3\zeta/2$.

The frequency shift for a single crystal is:

$$
\delta(\beta,\gamma) = \delta_\text{iso} + \frac{\zeta}{2}\bigl(3\cos^2\beta - 1 - \eta\sin^2\beta\cos 2\gamma\bigr)
$$


### Magnetic shielding vs. chemical shift

Soprano stores NMR tensors as **magnetic shielding tensors** ($\sigma$), where larger values correspond to greater electronic shielding. In contrast, programs such as SIMPSON use **chemical shifts** ($\delta$).

The chemical shift is defined as the fractional difference between the resonance frequency of a nucleus and that of a reference compound:

$$
\delta = \frac{\nu - \nu_{\mathrm{ref}}}{\nu_{\mathrm{ref}}}
$$

Here, $\nu_{\mathrm{ref}}$ is the resonance frequency of the chosen reference.

Because the resonance frequency depends on the shielding according to

$$
\nu = \nu_0 (1 - \sigma),
$$

where $\nu_0$ is the bare Larmor frequency of the nucleus, the chemical shift can be written as

$$
\delta = \frac{\sigma_{\mathrm{ref}} - \sigma}{1 - \sigma_{\mathrm{ref}}}.
$$

Since $\sigma_{\mathrm{ref}}$ is typically only tens to hundreds of ppm ($10^{-4}$–$10^{-3}$), the denominator differs from 1 by less than 0.1%. In practice, the usual approximation is therefore

$$
\delta \approx \sigma_{\mathrm{ref}} - \sigma.
$$

This means that chemical shift is inversely related to shielding:

* Larger shielding ($\sigma$) corresponds to smaller chemical shift ($\delta$) (more upfield).
* Smaller shielding ($\sigma$) corresponds to larger chemical shift ($\delta$) (more downfield).

If the shielding principal values are ordered as

$$
\sigma_{11} \le \sigma_{22} \le \sigma_{33},
$$

then the corresponding chemical shift values are ordered as

$$
\delta_{11} \ge \delta_{22} \ge \delta_{33}.
$$

The tensor axes are unchanged; only the numerical ordering is reversed.

---

#### `references` and `gradients` arguments

##### `references`

The `references` argument specifies the constant term in the conversion from magnetic shielding to chemical shift.

Its interpretation depends on whether `gradients` is supplied:

* **Direct referencing** (`gradients` omitted):
  `reference` is the shielding of the reference compound, $\sigma_{\mathrm{ref}}$, and the expression is:

   $$
   \delta = \frac{\sigma_{\mathrm{ref}} - \sigma}{1 - \sigma_{\mathrm{ref}}}.
   $$


* **DFT calibration** (`gradients` explicitly supplied):
  `reference` is the intercept $a$ of a linear regression of **experimental chemical shifts** ($\delta_{\mathrm{exp}}$, y-axis) against **computed shieldings** ($\sigma_{\mathrm{calc}}$, x-axis):

  $$
  \delta_{\mathrm{exp}} = a + m\,\sigma_{\mathrm{calc}}.
  $$

  In this form:

  * `reference = a`
  * `gradient = m`

  The intercept $a$ is the predicted chemical shift for a nucleus with zero calculated shielding.

  If your regression was performed in the opposite direction (i.e. with experimental shifts on the x-axis),

  $$
  \sigma_{\mathrm{calc}} = m'\,\delta_{\mathrm{exp}} + a',
  $$

  then the equivalent Soprano parameters are

  $$
  \text{gradient} = \frac{1}{m'}, \qquad \text{reference} = -\frac{a'}{m'}.
  $$

Typical values are provided per element, for example:

```python
references = {"C": 170.0, "H": 30.0}
```

If `references` is not provided, the `MagneticShielding` object contains only raw shielding values. Any attempt to export chemical shifts (for example, with `to_simpson()`) will raise a `ValueError`.

---

##### `gradients`

The `gradients` argument specifies the slope $m$ in the conversion

$$
\delta = \text{reference} + m\,\sigma.
$$

Its default value is $-1$, corresponding to the ideal relationship between shielding and chemical shift. Note the above discussion on the convention of the calibration plot assumed by Soprano. 

When calibrating against experiment, the fitted slope may deviate slightly from $-1$ because of systematic errors in the computational method. Providing per-element gradients allows these deviations to be corrected.

---

Both `references` and `gradients` can be provided as:

* A single numeric value
* A dictionary keyed by element
* A list of per-site values

Examples:

```python
references = 170.0
references = {"C": 170.0, "H": 30.0}
```

If a single value is supplied for a system containing multiple nuclear species, Soprano issues a `UserWarning`, since different nuclei generally require different reference shieldings. For example:

* $^{13}\mathrm{C}$: approximately 170 ppm
* $^{1}\mathrm{H}$: approximately 30 ppm

Applying one value to all species is usually incorrect.

---

## Quadrupolar Coupling

### Spinsys format

```
quadrupole n order Cq_hz eta alpha beta gamma
```

* `Cq_hz` is the quadrupolar coupling constant in **Hz**.
* Internally, SIMPSON converts $C_q$ from Hz to angular frequency units by multiplication with $2\pi$.

### First-order splitting ($I = 1$)

For a spin-1 nucleus (e.g. $^2$H), the first-order quadrupolar frequency shifts of the $m=\pm1 \leftrightarrow 0$ transitions are:

$$
\nu_\pm = \pm\frac{3C_q}{8}\bigl(3\cos^2\beta - 1\bigr)
$$

At $\beta = 0^\circ$: splitting = $3C_q/2$ (two peaks at $\pm 3C_q/4$).
At $\beta = 90^\circ$: splitting = $3C_q/4$ (two peaks at $\pm 3C_q/8$).

---

(dipolar-section)=
## Dipolar Coupling

### Spinsys format

```
dipole i j d_hz alpha beta gamma
```

* `d_hz` is the dipolar coupling constant in **Hz**.
* **Important:** SIMPSON does **not** multiply $d$ by $2\pi$ when storing it in the `Dipole` object.  Instead, SIMPSON applies the necessary scaling during spatial tensor construction via an internal prefactor in the dipolar Hamiltonian assembly.

Here $d$ denotes the conventional secular dipolar coupling constant in frequency units:

$$
d = -\frac{\mu_0}{4\pi}\frac{\gamma_1\gamma_2\hbar}{2\pi r^3}.
$$

**Do not apply any extra $2\pi$ factor to the dipolar coupling constant when exporting to SIMPSON.**

---

(jcoupling-aniso-section)=
## J-Coupling

### Spinsys format

```
jcoupling i j Jiso_hz Janiso_hz eta alpha beta gamma
```

* `Jiso_hz` is the isotropic J-coupling in Hz.  SIMPSON multiplies this by $2\pi$ internally.
* `Janiso_hz` is the anisotropic part.  SIMPSON also multiplies this by $2\pi$ internally.

### Isotropic part

The isotropic Hamiltonian is assembled as:

```cpp
jptr->iso(iso_Jcoupling * 2.0*M_PI);
// ...
s->Hiso().multiply_add(jptr->Tiso(), jptr->iso() * corrfac);
```

where `corrfac = 1.0` for homonuclear and `corrfac = 1/sqrt(2/3)` for heteronuclear.  This correctly produces $H_\text{iso} = J_\text{iso} \, \mathbf{I}_1 \cdot \mathbf{I}_2$ (homonuclear) or $H_\text{iso} = J_\text{iso} \, I_{1z} I_{2z}$ (heteronuclear).

### Anisotropic part — the $\zeta/2$ factor

The anisotropic J-coupling implementation is less transparently documented than other interactions and is therefore described here in implementation-specific detail.  SIMPSON builds the anisotropic J Hamiltonian via three steps (`readsys.cpp`, `wigner.cpp`, `spinsys.cpp`):

1. **`Rmol_c(2.0)`** (`readsys.cpp:453`):
   ```cpp
   void Rmol_c(double c) { Rmol(Dtensor2(c*delta(), eta())); ... }
   ```
   The anisotropy is multiplied by **2** before entering the spatial tensor.

2. **`Dtensor2`** (`wigner.cpp:53`):
   ```cpp
   v[2] = Complex(deltazz * sqrt(3.0/2.0), 0.0);
   ```
   The $q=0$ component is scaled by $\sqrt{3/2}$.

3. **Spin operators** (`spinsys.cpp`):
   * Heteronuclear: `IzIz_sqrt2by3 = sqrt(2/3) * Iz1 ⊗ Iz2`
   * Homonuclear: `T20 = sqrt(1/6) * (3 Iz1 Iz2 − I1·I2)`

#### Net scaling (heteronuclear)

Combining these factors:

$$
\begin{aligned}
\text{physical coefficient}
  &= \underbrace{\sqrt{\tfrac{2}{3}}}_{\text{spin op.}}
   \times \underbrace{\sqrt{\tfrac{3}{2}}}_{\text{Dtensor2}}
   \times \underbrace{2}_{\text{Rmol}_{c}}
   \times J_\text{aniso} \\
  &= 2 \, J_\text{aniso}
\end{aligned}
$$

The secular heteronuclear Hamiltonian at angle $\beta$ is therefore:

$$
H_\text{aniso} = 2 J_\text{aniso} \cdot I_{1z} I_{2z} \cdot \frac{3\cos^2\beta - 1}{2}
$$

For the two transitions of the observed spin, the splitting is $2 J_\text{aniso}$ at $\beta = 0^\circ$.

#### Net scaling (homonuclear)

$$
\begin{aligned}
\text{physical coefficient}
  &= \underbrace{\sqrt{\tfrac{1}{6}}}_{\text{spin op.}}
   \times \underbrace{\sqrt{\tfrac{3}{2}}}_{\text{Dtensor2}}
   \times \underbrace{2}_{\text{Rmol}_{c}}
   \times J_\text{aniso} \\
  &= J_\text{aniso}
\end{aligned}
$$

The secular homonuclear Hamiltonian is:

$$
H_\text{aniso} = J_\text{aniso} \cdot \bigl(3 I_{1z} I_{2z} - \mathbf{I}_1\cdot\mathbf{I}_2\bigr) \cdot \frac{3\cos^2\beta - 1}{2}
$$

Again, the observable splitting at $\beta = 0^\circ$ is $2 J_\text{aniso}$.

#### Conclusion

In both homonuclear and heteronuclear cases, the physical splitting equals $2 \times J_\text{aniso}$.  For **SIMPSON v6.0.2**, the effective scaling implies that a physical reduced anisotropy $\zeta$ must be supplied as:

$$
\boxed{J_\text{aniso}^\text{(SIMPSON)} = \frac{\zeta}{2}}
$$

**Example:** A J-coupling tensor with reduced anisotropy $\zeta = 200$ Hz must be entered as `aniso = 100.0` in SIMPSON's `jcoupling` line.

This scaling has been validated numerically for SIMPSON v6.0.2 and should not be assumed for other versions without verification.

---

## Validation

All conventions described above were validated by single-crystal SIMPSON simulations using explicit orientation selection (`crystal_file alpha0beta0`) and comparison of simulated peak positions against analytical predictions.  Validation tolerances were set to numerical agreement within simulation resolution.

The validation suite lives in `tests/simpson_validation/` and covers:

| Test | Interaction | Parameter | Expected | Result |
|------|-------------|-----------|----------|--------|
| 01 | Isotropic shift | $\delta_\text{iso}$ = 100 ppm | Peak at +10,060 Hz | ✓ Pass |
| 02a | CSA ($\beta=0^\circ$) | $\zeta$ = 200 ppm | Peak at +20,120 Hz | ✓ Pass |
| 02b | CSA ($\beta=90^\circ$) | $\zeta$ = 200 ppm | Peak at −10,060 Hz | ✓ Pass |
| 02c | CSA (magic angle) | $\zeta$ = 200 ppm | Peak at 0 Hz | ✓ Pass |
| 03a | CSA ($\eta=0.5$, $\beta=90^\circ$) | $\zeta$ = 300 ppm | Peak at −22,635 Hz | ✓ Pass |
| 03b | CSA ($\eta=0.5$, $\gamma=90^\circ$) | $\zeta$ = 300 ppm | Peak at −7,545 Hz | ✓ Pass |
| 04a | Quadrupolar ($\beta=0^\circ$) | $C_q$ = 100 kHz | Peaks at ±75 kHz | ✓ Pass |
| 04b | Quadrupolar ($\beta=90^\circ$) | $C_q$ = 100 kHz | Peaks at ±37.5 kHz | ✓ Pass |
| 05a | Dipolar ($\beta=0^\circ$) | $d$ ≈ 30.25 kHz | Splitting ≈ 60.5 kHz | ✓ Pass |
| 05b | Dipolar ($\beta=90^\circ$) | $d$ ≈ 30.25 kHz | Splitting ≈ 30.25 kHz | ✓ Pass |
| 06 | J-coupling (isotropic) | $J_\text{iso}$ = 100 Hz | Splitting = 100 Hz | ✓ Pass |
| 07a | J-aniso ($\beta=0^\circ$) | $\zeta$ = 200 Hz | Splitting = 200 Hz | ✓ Pass |
| 07b | J-aniso ($\beta=90^\circ$) | $\zeta$ = 200 Hz | Splitting = 100 Hz | ✓ Pass |

### Running the validation suite

```bash
cd tests/simpson_validation
python generate_tests.py      # regenerate .spinsys and .in files
# run each .in file with SIMPSON
python verify.py              # check peak positions
```

---

## References

* M. Bak, J. T. Rasmussen, N. C. Nielsen, [*J. Magn. Reson.* **2000**, *147*, 296–330](https://doi.org/10.1006/jmre.2000.2179). (SIMPSON original paper)
* D. L. Goodwin, J. P. Carvalho, A. B. Nielsen, N. Wili, T. Vosegaard, Z. Tosner, and N. C. Nielsen, [arXiv:2602.15793](https://doi.org/10.48550/arXiv.2602.15793) (2026). (SIMPSON next-generation paper)
* SIMPSON source code (v6.0.2), particularly [`src/readsys.cpp`](https://gitlab.au.dk/nmr/simpson/-/blob/main/src/readsys.cpp?ref_type=heads), [`src/wigner.cpp`](https://gitlab.au.dk/nmr/simpson/-/blob/main/src/wigner.cpp?ref_type=heads), [`src/spinsys.cpp`](https://gitlab.au.dk/nmr/simpson/-/blob/main/src/spinsys.cpp?ref_type=heads), [`src/ham.cpp`](https://gitlab.au.dk/nmr/simpson/-/blob/main/src/ham.cpp?ref_type=heads), and [`include/sim.h`](https://gitlab.au.dk/nmr/simpson/-/blob/main/include/sim.h?ref_type=heads).

---

## Implementation Notes

SIMPSON internally converts all input frequencies (Hz) to angular frequencies (rad s⁻¹) by multiplying by $2\pi$ before assembling the Hamiltonian.  The dipolar coupling constant is a notable exception — see below.

### Why no $2\pi$ for dipole?

Looking at SIMPSON's source (`readsys.cpp`):

```cpp
s->DD[(s->nDD)] = new Dipole(n1,n2,asym_param,orientation);
s->DD[(s->nDD)]->delta(dipole_aniso);   // stored in Hz, NOT multiplied by 2π
s->DD[(s->nDD)]->Rmol_c(4.0*M_PI);      // 4π prefactor
```

The Hamiltonian is assembled as `HQ_multod` with `T = IzIz_sqrt2by3` (heteronuclear) or `T20` (homonuclear).  Combined with `Dtensor2`'s $\sqrt{3/2}$ scaling, the net result is that the input $d$ Hz reproduces the conventional secular dipolar splitting $d|3\cos^2\beta - 1|$.

### Version-specific behaviour

Several conventions described here—particularly the treatment of dipolar and anisotropic J-coupling scaling—are inferred from SIMPSON v6.0.2 source code and validated numerically.  These behaviours reflect the implementation of that version and may change in future releases.
