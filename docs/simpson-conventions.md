# SIMPSON Tensor Conventions

This page documents the conventions used by [SIMPSON](https://inano.au.dk/about/research-centers-and-projects/nmr/software/simpson) for representing NMR interaction tensors in `.spinsys` files, and how Soprano maps its internal tensors to them.

The conventions described below were read from the source code of **[SIMPSON v6.0.2](https://gitlab.au.dk/nmr/simpson)** and checked numerically with SIMPSON 6.0.1 and 6.1.3 (the stored spectrum used by the tests was produced with 6.0.1).  Internal implementation details may change in future SIMPSON releases; users should revalidate version-specific behaviour where noted.

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

SIMPSON uses **passive ZYZ Euler angles** (in degrees) to specify the orientation of the interaction principal-axis system ($X$, $Y$, $Z$) relative to the molecular/crystal frame.  The precise meaning of "passive" here, and why mrsimulator needs different numbers, is given in {ref}`euler-angles-section`.

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

(euler-angles-section)=
## Euler Angles (Tensor Orientation)

Every interaction line ends with three Euler angles `alpha beta gamma` that orient its principal-axis system (PAS) in a common frame. For a Soprano tensor that frame is the one the tensor's Cartesian components are given in (the magres frame, for tensors read from a `.magres` file).

**SIMPSON and mrsimulator do not take the same numbers.** Soprano writes the angles in each program's own convention, summarised here and defined below.

| | SIMPSON | mrsimulator |
|---|---|---|
| Units | degrees | radians |
| PAS axes in the common frame, $E$ | $R_z(-\gamma)\,R_y(-\beta)\,R_z(-\alpha)$ | $R_z(\gamma)\,R_y(\beta)\,R_z(\alpha)$ |
| Unique axis of an axial tensor | polar $\beta$, azimuth $180^\circ-\gamma$ | polar $\beta$, azimuth $\gamma$ |
| Free angle of an axial tensor | $\alpha$ | $\alpha$ |
| Soprano | `euler_angles(passive=True, degrees=True)` | `mrsimulator_euler_angles()` |

Here $R_z(\theta)$ and $R_y(\theta)$ are ordinary right-handed rotation matrices (counter-clockwise for positive $\theta$), and $E$ is the tensor's eigenvector matrix, whose columns are the PAS axes expressed in the common frame, so that

$$
\mathbf{T} = E\,\mathrm{diag}(T_{XX}, T_{YY}, T_{ZZ})\,E^{\mathsf T}.
$$

### Terminology (TensorView)

The names *active* and *passive* follow TensorView for MATLAB (Svenningsson and Mueller, *Solid State Nucl. Magn. Reson.* **123**, 101849, 2023), which defines them for a tensor $A_\mathrm{MF}=R\,A_\mathrm{PAS}R^{-1}$ as

$$
R_\mathrm{active}(\alpha,\beta,\gamma)=R_z(\alpha)R_y(\beta)R_z(\gamma)\ \text{(Eq. 65)},\qquad
R_\mathrm{active}(\alpha,\beta,\gamma)=R_\mathrm{passive}(-\gamma,-\beta,-\alpha)\ \text{(Eq. 9)}.
$$

In those terms the three conventions that appear here are:

| Convention | $E$ | TensorView |
|---|---|---|
| SIMPSON | $R_z(-\gamma)R_y(-\beta)R_z(-\alpha)$ | **passive ZYZ** |
| mrsimulator, Appendix B of its paper | $R_z(\alpha)R_y(\beta)R_z(\gamma)$ | **active ZYZ** |
| mrsimulator, its code | $R_z(\gamma)R_y(\beta)R_z(\alpha)$ | active ZYZ with $\alpha$ and $\gamma$ interchanged |

`NMRTensor.euler_angles(passive=False)` returns TensorView's active angles, and `passive=True` its passive angles. The same four orientation-equivalent sets (a half turn about each PAS axis) apply as in the paper (Eqs. 16-23 for a single tensor, 48-63 for two). The paper allows $0\le\beta\le\pi$; Soprano reports the member that TensorView's code reports (`MFtoEuler.m`, `RelativeAngles.m`): $0\le\beta<\pi/2$, then $0\le\gamma<\pi$ for active angles or $0\le\alpha<\pi$ for passive ones. For a single tensor this member is unique, so Soprano's numbers are TensorView's numbers; `tests/test_data/tensorview_reference.json` holds TensorView's output for a set of triaxial and axial tensors in every order and mode, and the tests check they agree.

`euler_to` returns the angles of $R_\mathrm{rel}=E_A^{\mathsf T}E_B$ (Eq. 31), the orientation of tensor $B$ in the PAS of $A$. TensorView's `RelativeAngles(Tensor1, Tensor2)` decomposes `U2^-1 U1`, the orientation of Tensor 1 in the PAS of Tensor 2, so **`A.euler_to(B)` corresponds to TensorView with Tensor 1 = B and Tensor 2 = A**. For relative angles the two programs can report different members of the 16 equivalent sets, because the choice depends on the signs of the eigenvectors their eigensolvers return.

### Axial tensors and the free angle

For an axial tensor (every dipolar coupling, and any CSA, EFG or J tensor with $\eta=0$) the rotation about the PAS $Z$ axis has no effect, so one angle is free. Soprano sets it to zero (the active $\gamma$ or the passive $\alpha$, as in TensorView), which for a bond at polar angle $\theta$ and azimuth $\phi$ gives `0  θ  180-φ` for SIMPSON (e.g. `dipole 1 2 d 0 70 140` for $\theta=70^\circ$, $\phi=40^\circ$) and $(0, \theta, \phi)$ in radians for mrsimulator.

If the unique axis is along $X$ instead (eigenvalues $[a,b,b]$, as in increasing order) there is no single free angle, and TensorView chooses the frame with active $\alpha=0$; Soprano does the same, by turning the PAS about $X$ until the third column of $E$ lies in the $xz$ plane. Both then normalise the range as above, so the active $\alpha$ is $0$ or $\pi$ (it is $\pi$ when that frame has $\beta\ge\pi/2$) and the passive $\gamma$ is $0$ or $\pi$.

For the relative angles of two tensors of which exactly one is axial, the axial tensor is written in the PAS of the other and given the single-tensor gauge, as TensorView does; for a unique axis along $Z$ this sets the first active angle to zero if the first tensor is axial and the last if the second is. Two axial tensors give $(0,\beta,0)$ in active ZYZ, where $\beta$ is the angle between the unique axes (or its complement if exactly one of them is along $X$), and Table 1 of the paper for ZXZ; these too are then normalised, so Table 1's $(0, 90^\circ, \gamma)$ is reported as $(180^\circ, 90^\circ, 180^\circ-\gamma)$, as TensorView displays it.

Soprano deliberately departs from TensorView's software (`tests/reference_generators/tensorview.py` runs the pinned TensorView source under Octave and stores its output) in three cases, where its output does not reproduce the input tensors or does not follow the paper. (i) For two axial tensors when the first one ($A$, TensorView's Tensor 2) has its unique axis along $X$, its formula reads $B$ along one of $A$'s arbitrary perpendicular axes, and the angles it returns do not reproduce the tensors. (ii) Its ZXZ branch for two axial tensors tests degeneracy with exact equality, so eigenvalues that differ in the 8th digit make it skip Table 1. (iii) Its test for two aligned axial tensors (a signed sum of three off-diagonal elements vanishing) also passes for some non-aligned geometries, for example unique axes along $(1,1,-\tfrac12)$ and $z$, $70.5^\circ$ apart, which it reports as zero angles; Soprano tests the angle between the axes instead.

### SIMPSON

SIMPSON's angles are the *passive* ones (TensorView's passive ZYZ): with $M=R_z(\alpha)R_y(\beta)R_z(\gamma)$, the tensor is $M^{\mathsf T}\,\mathrm{diag}\,M$, so $E=M^{\mathsf T}$ and $(\alpha,\beta,\gamma)$ are the intrinsic ZYZ angles of $E^{-1}$. This is consistent with SIMPSON's `wigner2`, whose comments describe the angles as passive and in which the spherical components of an axial tensor depend on $\beta$ and $\gamma$ only.

```{warning}
Do not read SIMPSON's angles as $R_z(\gamma)R_y(\beta)\hat z$ with ordinary active matrices: that gives the wrong azimuth ($\gamma$ rather than $180^\circ-\gamma$) for the unique axis of an axial tensor.
```

### mrsimulator

mrsimulator's code builds the PAS axes as $E=R_z(\gamma)R_y(\beta)R_z(\alpha)$: `mrsimulator.utils.cartesian_tensor` calls `Rotation.from_euler("zyz", [alpha, beta, gamma])`, where scipy's lower-case "zyz" is *extrinsic*, so $\alpha$ acts first. Soprano's `NMRTensor.mrsimulator_euler_angles()` returns exactly these angles; they are `(-alpha - pi, beta, pi - gamma)` of the passive angles above (with the free angle of an axial tensor set to zero).

```{warning}
Appendix B of the mrsimulator paper (J. Chem. Phys. **161**, 212501, 2024) uses TensorView's active ZYZ convention and writes $L(\alpha,\beta,\gamma)=L_z(\alpha)L_y(\beta)L_z(\gamma)$ and extracts the angles from the eigenvector matrix with $\beta=\arccos l_{33}$, $\alpha=\arctan2(l_{23},l_{13})$, $\gamma=\arctan2(l_{32},-l_{31})$ (Eqs. B13-B16). That is the code's frame with $\alpha$ and $\gamma$ in each other's places, so the paper's equations and the code describe different frames. Angles obtained from those equations give wrong spectra in mrsimulator 1.0.0 whenever more than one oriented interaction is present, whereas angles from `mrsimulator.utils.cartesian_tensor` give correct ones. Soprano follows the code. This is a mismatch between the paper and the code, not a fault in how mrsimulator simulates; it is worth raising upstream.

mrsimulator's own documentation describes its angles as passive. That is not TensorView's meaning: the code's matrix $R_z(\gamma)R_y(\beta)R_z(\alpha)$ is neither of TensorView's two sets. A formula that only flips the signs of the passive angles does not reproduce it (it is off by order one for generic angles in a check against `from_haeberlen_params`), which is why `mrsimulator_euler_angles()` converts them as $(-\alpha-\pi, \beta, \pi-\gamma)$.
```

The two programs also differ in the sign of the anisotropy: mrsimulator's `zeta` is the *shielding*-sense anisotropy ($\zeta_\sigma = -\zeta_\delta$), whereas SIMPSON's `aniso` is the *shift*-sense one. Both are derived from the same referenced shift tensor, so a custom `gradient` or the full-formula conversion scales them identically (see {ref}`references and gradients <references-and-gradients>`).

### How the two conventions relate

The same tensor written in the two conventions gives different numbers. Reading one program's numbers with the other's formula does not give a garbage orientation, but the tensor rotated by $\pi$ about $x$ (the two formulas differ by a mirror conjugation, which for a symmetric tensor is a proper $\pi$ rotation). If it is applied to every tensor in a spin system, as when all angles come from one convention, it has no effect on a powder spectrum. It would matter for a single-crystal simulation, or if angles from the two conventions (or from outside Soprano) were mixed in one spin system.

```{note}
For SIMPSON the absolute frame was checked with single-crystal simulations (see below): for a crystallite $(\alpha, \beta)$ the field direction in the frame the angles are written in is $R_z(\alpha)R_y(\beta)\hat z$, and the two lines of a CSA + dipolar pair match those of the lab-frame tensors Soprano started from to within 0.3 Hz at six generic orientations, whereas the mirrored frame is off by kHz. The absolute frame of the mrsimulator output has not been checked by a single-crystal simulation: what has been verified for mrsimulator is the relative orientation of all interactions in a spin system, via powder spectra.
```

### Verification

Powder spectra from SIMPSON v6.1.3 and mrsimulator 1.0.0 were compared with an analytic spectrum computed directly from the lab-frame tensors (no Euler-angle convention involved), for CSA + dipolar, CSA + J, $^2$H CSA + EFG, and axial EFG, CSA and J tensors, all with tensors rotated out of the $xz$-plane. After the fix, every case agrees with the analytic spectrum to the level of the powder-averaging error.

The tests do not need either program. `tests/nmr_export_orientation_tests.py` rebuilds each tensor from the written angles with each program's formula, and checks those formulas against stored output of the programs themselves (regenerated by `tests/reference_generators/simulators.py`):

* `tests/test_data/mrsimulator_reference.json`: tensors from mrsimulator's own `from_haeberlen_params` for arbitrary angles, and the static powder spectrum of a $^{13}$C CSA coupled to $^{1}$H (both tensors tilted, the bond off the $xz$-plane) simulated by mrsimulator's engine from the spin system Soprano writes;
* `tests/test_data/simpson_reference.json`: the same spectrum simulated by SIMPSON from Soprano's spinsys lines.

Both stored spectra agree with the analytic spectrum to about 2% of the maximum; reading the same numbers with TensorView's active convention (Appendix B, for mrsimulator) gives a spectrum about 25% off. A CI job installs mrsimulator and runs the tests that call it directly.

The single-crystal check for SIMPSON (`TestSimpsonSingleCrystal` in `tests/nmr_export_orientation_tests.py`) uses the same spin system with six crystallites chosen away from the $xz$-plane (`crystal_file` with one orientation, `gamma_angles 1`). The predicted lines are $\nu_\mathrm{L}\,10^{-6}\,\mathbf n^{\mathsf T}\boldsymbol\delta\,\mathbf n \pm \tfrac12\,\mathbf n^{\mathsf T}\mathbf D\,\mathbf n$ with $\mathbf n = R_z(\alpha)R_y(\beta)\hat z$, and are stored in `simpson_reference.json` (SIMPSON 6.0.1). A second CI job runs SIMPSON itself, from the `jkshenton/simpson` container image pinned by digest, on two of the crystallites and on the powder spectrum.

```{warning}
mrsimulator integrates powders over an octant by default (`integration_volume="octant"`), which is only exact when every tensor in the spin system has the symmetry of the first. With CSA, dipolar, J or EFG tensors in general relative orientations, set `sim.config.integration_volume = "hemisphere"`; with the default the spectrum of the system above is wrong regardless of the Euler angles.
```

```{note}
In the single-crystal validation suite below, test 03b puts the field along the PAS $Y$ axis through SIMPSON's $\alpha$ (the written line is `90 90 0`) and SIMPSON returns the $Y$ frequency, which is what the passive convention predicts (with an active reading it would be $\gamma$ that matters). A powder spectrum cannot separate the SIMPSON and mrsimulator conventions from each other (see above); the single-crystal lines described above fix the absolute frame of the SIMPSON angles, and the mrsimulator convention rests on the program's code and on its own `from_haeberlen_params` output.
```

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

For a static single crystal with the crystallite at $\alpha=\beta=\gamma=0$ (`crystal_file alpha0beta0`), the field direction in the PAS is the third row of $E=R_z(-\gamma)R_y(-\beta)R_z(-\alpha)$, $(\sin\beta\cos\alpha, \sin\beta\sin\alpha, \cos\beta)$, so with the angles written on the `shift` line the frequency shift is:

$$
\delta(\alpha,\beta) = \delta_\text{iso} + \frac{\zeta}{2}\bigl(3\cos^2\beta - 1 - \eta\sin^2\beta\cos 2\alpha\bigr)
$$

The $\gamma$ of the passive angles does not enter (it is the familiar $\cos 2\gamma$ of the *active* angles that $\alpha$ replaces here).


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

(references-and-gradients)=
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
| 03a | CSA ($\eta=0.5$, $\alpha=0^\circ$, $\beta=90^\circ$) | $\zeta$ = 300 ppm | Peak at −22,635 Hz | ✓ Pass |
| 03b | CSA ($\eta=0.5$, $\alpha=90^\circ$, $\beta=90^\circ$) | $\zeta$ = 300 ppm | Peak at −7,545 Hz | ✓ Pass |
| 04a | Quadrupolar ($\beta=0^\circ$) | $C_q$ = 100 kHz | Peaks at ±75 kHz | ✓ Pass |
| 04b | Quadrupolar ($\beta=90^\circ$) | $C_q$ = 100 kHz | Peaks at ±37.5 kHz | ✓ Pass |
| 05a | Dipolar ($\beta=0^\circ$) | $d$ ≈ 30.25 kHz | Splitting ≈ 60.5 kHz | ✓ Pass |
| 05b | Dipolar ($\beta=90^\circ$) | $d$ ≈ 30.25 kHz | Splitting ≈ 30.25 kHz | ✓ Pass |
| 06 | J-coupling (isotropic) | $J_\text{iso}$ = 100 Hz | Splitting = 100 Hz | ✓ Pass |
| 07a | J-aniso ($\beta=0^\circ$) | $\zeta$ = 200 Hz | Splitting = 200 Hz | ✓ Pass |
| 07b | J-aniso ($\beta=90^\circ$) | $\zeta$ = 200 Hz | Splitting = 100 Hz | ✓ Pass |

The orientation of tensors (the Euler angles) is covered separately by `tests/nmr_export_orientation_tests.py`; see {ref}`euler-angles-section`.

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
