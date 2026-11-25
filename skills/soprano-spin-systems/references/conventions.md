# NMR Tensor, Sign & Euler Angle Conventions in Soprano

## 1. Eigenvalue Ordering & Tensor Conventions

Soprano's `NMRTensor` supports four eigenvalue orderings via `TensorConvention` (`order=`):
- `"h"` (`ORDER_HAEBERLEN`): $|\lambda_{zz} - \lambda_{\mathrm{iso}}| \ge |\lambda_{xx} - \lambda_{\mathrm{iso}}| \ge |\lambda_{yy} - \lambda_{\mathrm{iso}}|$
- `"n"` (`ORDER_NQR`): $|\lambda_{xx}| \le |\lambda_{yy}| \le |\lambda_{zz}|$
- `"i"` (`ORDER_INCREASING`): $\lambda_{11} \le \lambda_{22} \le \lambda_{33}$
- `"d"` (`ORDER_DECREASING`): $\lambda_{11} \ge \lambda_{22} \ge \lambda_{33}$

### Default Eigenvalue Orders Across APIs (Watch Out!)

| API | Default `order` |
|---|---|
| `MagneticShielding(...)` | `"h"` (Haeberlen) |
| `MSTensor.get(...)` / `MSEuler.get(...)` / `MSAnisotropy.get(...)` | `"h"` (Haeberlen; `MSTensor` was `"i"` before v0.12) |
| `ElectricFieldGradient(...)` / `EFGTensor.get(...)` / `EFGEuler.get(...)` | `"n"` (NQR) |

> **Warning:** `A.euler_to(B)` depends on the PAS axis ordering of **both** tensors. Pass explicit `order="h"` (or `"n"`) to `MSTensor.get(..., order="h")` / `EFGTensor.get(...)` before computing relative Euler angles. `euler_to`, `equivalent_euler_to` and `rotation_to` warn when both tensors are of the same type (e.g. two shielding tensors) but use different orders, since the angles then differ (e.g. by 90° in $\gamma$); mixing types (Haeberlen shielding with NQR EFG) stays silent.

In the **Haeberlen** convention:
- Isotropic value: $\lambda_{\mathrm{iso}} = \frac{1}{3}(\lambda_{xx} + \lambda_{yy} + \lambda_{zz})$
- Reduced anisotropy: $\zeta = \lambda_{zz} - \lambda_{\mathrm{iso}}$
- Full anisotropy: $\Delta = \frac{3}{2}\zeta$
- Asymmetry: $\eta = (\lambda_{yy} - \lambda_{xx}) / \zeta \in [0, 1]$
- **Maryland / Herzfeld-Berger** span & skew: $\Omega = \lambda_{33} - \lambda_{11} \ge 0$, $\kappa = 3(\lambda_{\mathrm{iso}} - \lambda_{22})/\Omega$ (for shielding; sign reverses for shift).

`MagneticShielding` provides convention views: `.haeberlen_shielding` / `.haeberlen_shift`, `.mehring_shielding` / `.mehring_shift`, `.maryland_shielding` / `.maryland_shift`, `.iupac_shielding` / `.iupac_shift`, and `.herzfeldberger_shielding` / `.herzfeldberger_shift` (all `*_shift` views require `reference` to be set).

## 2. Referencing Formulas (`references` & `gradients`)

- **Direct referencing (`gradients=None`):**
  $$\delta = \frac{\sigma_{\mathrm{ref}} - \sigma}{1 - \sigma_{\mathrm{ref}} \times 10^{-6}}$$
- **Linear DFT calibration (`gradients` provided, e.g. from regressing $\delta_{\mathrm{exp}} = a + m\sigma_{\mathrm{calc}}$):**
  $$\delta = a + m\,\sigma$$
  If your fit was $\sigma_{\mathrm{calc}} = m'\delta_{\mathrm{exp}} + a'$, convert via `gradient = 1 / m'`, `reference = -a' / m'`.

> **Per-element mode:** `get_spin_system()`, `MSShift`, `MagneticShielding`, `write_spinsys` and `soprano spinsys` all default to `gradients=None` (full formula for every element). The mode is chosen **per element**: an element absent from `gradients` uses the full formula, an element present uses the linear fit. An explicit `-1` is therefore a linear fit ($\delta = \sigma_{\mathrm{ref}} - \sigma$, no denominator) and emits a warning, as does a `gradients` dict that omits referenced elements.

## 3. Euler Angle Conventions (SIMPSON vs mrsimulator vs TensorView)

Let $E$ be the tensor's eigenvector matrix whose columns are the PAS axes $(X, Y, Z)$ expressed in the common/magres frame, so $\mathbf{T} = E\,\mathrm{diag}(T_{XX}, T_{YY}, T_{ZZ})\,E^{\mathsf T}$.

| Target | Units | Formula for $E$ | Axial unique axis | Soprano method |
|---|---|---|---|---|
| **TensorView active ZYZ** | rad (or deg) | $R_z(\alpha)R_y(\beta)R_z(\gamma)$ | polar $\beta$, azimuth $\alpha$ | `tensor.euler_angles(passive=False)` |
| **SIMPSON (passive ZYZ)** | **degrees** | $R_z(-\gamma)R_y(-\beta)R_z(-\alpha)$ | polar $\beta$, azimuth $180^\circ - \gamma$ | `tensor.euler_angles(passive=True, degrees=True)` |
| **mrsimulator (code)** | **radians** | $R_z(\gamma)R_y(\beta)R_z(\alpha)$ | polar $\beta$, azimuth $\gamma$ | `tensor.mrsimulator_euler_angles()` |

- **Canonical gauge:** Soprano normalizes to TensorView's unique member: $0 \le \beta < \pi/2$, and $0 \le \gamma < \pi$ (active) or $0 \le \alpha < \pi$ (passive).
- **Axial tensors ($\eta = 0$, e.g. all static dipolar couplings):** The free rotation about the unique $Z$ axis is fixed to $0$ (passive $\alpha = 0$ in SIMPSON → `0 θ 180-φ`; $\alpha = 0$ in mrsimulator → `(0, θ, φ)`).
- **Relative angles & equivalent sets:**
  - `A.euler_to(B)` decomposes $R_{\mathrm{rel}} = E_A^{\mathsf T}E_B$ (orientation of **B** in the PAS of **A**), corresponding to TensorView `RelativeAngles(Tensor1=B, Tensor2=A)`.
  - `tensor.equivalent_euler_angles(convention="zyz", passive=False)` returns all **4** orientation-equivalent Euler sets for a single tensor; `A.equivalent_euler_to(B, ...)` returns all **16** equivalent relative angle sets for a pair.
- **Why mixing conventions matters:** Reading one program's angles with the other's formula rotates a tensor by $\pi$ about $x$. Applying that uniformly to every tensor leaves a powder spectrum unchanged, but **mixing** conventions across tensors in one spin system (or using mrsimulator paper Appendix B equations, which swap $\alpha$ and $\gamma$) corrupts multi-tensor spectra.

## 4. SIMPSON (v6.0.2) vs mrsimulator Parameter Mapping

| Interaction | SIMPSON keyword & units | mrsimulator field & units | Key sign / scaling note |
|---|---|---|---|
| Isotropic shift | `shift n iso_p ...` (ppm with `p` suffix) | `isotropic_chemical_shift` (ppm) | SIMPSON converts ppm→Hz using **observed nucleus** $\nu_L = (|\gamma|/\gamma_{\mathrm{H}})\nu_{\mathrm{ref}}$ |
| CSA anisotropy | `shift n ... aniso_p eta a b g` ($\zeta_\delta$, ppm with `p`) | `shielding_symmetric.zeta` ($\zeta_\sigma = -\zeta_\delta$, ppm) | **Opposite sign** between SIMPSON (shift-sense) and mrsimulator (shielding-sense); both use reduced $\zeta$, NOT $\Delta = 3\zeta/2$ |
| Quadrupolar | `quadrupole n order Cq eta a b g` ($C_q$ in Hz) | `quadrupolar.Cq` (Hz), `eta` | `order` is 1 or 2 in SIMPSON; cross-terms `quadrupole_x_dipole` / `quadrupole_x_shift` emitted by default |
| Dipolar | `dipole i j d a b g` ($d$ in Hz) | `dipolar.D` (Hz) | $d = -\frac{\mu_0\hbar\gamma_1\gamma_2}{8\pi^2 r^3}$ (negative for like-sign $\gamma$); **no extra $2\pi$** in SIMPSON |
| Isotropic J | `jcoupling i j Jiso ...` (Hz) | `isotropic_j` (Hz) | — |
| Anisotropic J | `jcoupling i j ... Janiso eta a b g` (Hz) | `j_symmetric.zeta` (Hz) | **SIMPSON v6.0.2 takes $\zeta/2$** (Soprano divides by 2 automatically) |
