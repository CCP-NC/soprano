# Expected Results for SIMPSON Validation Tests

All tests use `crystal_file alpha0beta0` (single crystal) so the spectrum consists of
sharp delta-function lines at analytically predictable frequencies.

Tests were run against **SIMPSON v6.0.2**.

**SIMPSON conventions used:**
- `proton_frequency = 400e6` Hz (400 MHz spectrometer)
- Euler angles: ZYZ passive, degrees, right-handed
- `shift` anisotropy = reduced anisotropy ζ (ppm with `p` suffix)
- `jcoupling` anisotropy = ζ/2 (half the reduced anisotropy, in Hz)
- `dipole` anisotropy = dipolar coupling constant d (Hz)
- `quadrupole` anisotropy = Cq (Hz)

**Important note on ppm → Hz conversion:**
SIMPSON converts chemical shift ppm to Hz using the **observed nucleus's Larmor frequency**, not the proton frequency directly. For ¹³C at a 400 MHz proton spectrometer:

```
ν_C = (γ_C / γ_H) × 400 MHz ≈ 100.60 MHz
```

Therefore, 100 ppm for ¹³C = 100 × 10⁻⁶ × 100.60 MHz ≈ **10,060 Hz**.

---

## Test 01 — Isotropic Shift

**System**: Single ¹³C, δ_iso = +100 ppm

**Expected**: Single peak at **+100 ppm** = **+10,060 Hz** from carrier.

**Current Soprano output**:
```
shift 1 100.0p 0.0p 0.0 0.0 0.0 0.0
```
✓ Correct.

---

## Test 02 — CSA Anisotropy + Euler β

**System**: Single ¹³C, δ_iso = 0 ppm, ζ = +200 ppm, η = 0

**Formula** (single crystal, β = angle between z_PAS and B₀):
```
δ = δ_iso + (ζ/2)(3cos²β − 1)
```

### 02a — β = 0°
```
δ = 0 + (200/2)(3·1 − 1) = +200 ppm = +20,120 Hz
```
**Expected**: Single peak at **+200 ppm** (**+20,120 Hz**).

**Current Soprano output**:
```
shift 1 0.0p 200.0p -0.0 0.0 0.0 0.0
```
✓ Correct.

### 02b — β = 90°
```
δ = 0 + (200/2)(3·0 − 1) = −100 ppm = −10,060 Hz
```
**Expected**: Single peak at **−100 ppm** (**−10,060 Hz**).

**Current Soprano output**:
```
shift 1 0.0p 200.0p -0.0 90.0 89.99999999999999 0.0
```
✓ Correct (β ≈ 90°).

### 02c — β = 54.7356° (magic angle)
```
δ = 0 + (200/2)(3·(1/3) − 1) = 0 ppm
```
**Expected**: Single peak at **0 ppm** (**0 Hz**).

**Current Soprano output**:
```
shift 1 4.7e-15p 200.0p 7.1e-17 0.0 54.7356 0.0
```
✓ Correct (δ_iso ≈ 0 within numerical noise).

---

## Test 03 — CSA Asymmetry + Euler γ

**System**: Single ¹³C, δ_iso = 0 ppm, ζ = +300 ppm, η = 0.5

**Formula**:
```
δ = δ_iso + (ζ/2)(3cos²β − 1 − η sin²β cos 2γ)
```

### 03a — β = 90°, γ = 0°
```
δ = 0 + (300/2)(0 − 1 − 0.5·1·1) = −225 ppm = −22,635 Hz
```
**Expected**: Single peak at **−225 ppm** (**−22,635 Hz**).

**Current Soprano output**:
```
shift 1 0.0p 300.0p 0.5 0.0 89.99999999999999 180.0
```
✓ Correct (γ = 180° is equivalent to γ = 0° for this tensor because cos(2·180°) = cos(0°) = 1).

### 03b — β = 90°, γ = 90°
```
δ = 0 + (300/2)(0 − 1 − 0.5·1·(−1)) = −75 ppm = −7,545 Hz
```
**Expected**: Single peak at **−75 ppm** (**−7,545 Hz**).

**Current Soprano output**:
```
shift 1 0.0p 300.0p 0.5 90.0 90.00000000000001 0.0
```
✓ Correct (α = 90°, γ = 0° is equivalent to α = 0°, γ = 90° for this tensor;
cos(2·90°) = −1 gives the right shift).

---

## Test 04 — Quadrupolar Coupling (First-Order)

**System**: Single ²H (I = 1), Cq = 100 kHz, η = 0, order = 1

**Formula** (first-order quadrupolar splitting for I = 1):
```
ν± = ±(3Cq/8)(3cos²β − 1)
```

### 04a — β = 0°
```
ν± = ±(3·100/8)(3·1 − 1) = ±75 kHz
```
**Expected**: Two peaks at **+75 kHz** and **−75 kHz** relative to carrier.

**Current Soprano output**:
```
quadrupole 1 1 100000.0 0.0 90.0 0.0 0.0
```
✓ Correct (Cq = 100 kHz; α arbitrary for η = 0).

### 04b — β = 90°
```
ν± = ±(3·100/8)(0 − 1) = ∓37.5 kHz
```
**Expected**: Two peaks at **−37.5 kHz** and **+37.5 kHz** relative to carrier.

**Current Soprano output**:
```
quadrupole 1 1 100000.0 0.0 90.0 89.99999999999999 0.0
```
✓ Correct (β ≈ 90°).

---

## Test 05 — Dipolar Coupling

**System**: ¹H–¹³C pair, r = 1.0 Å, only dipolar coupling

**Analytical d**:
```
d = −(μ₀/4π) · (ħ γ_H γ_C)/(2π r³)
  ≈ −30.21 kHz   (negative because both γ > 0)
```

For heteronuclear dipolar coupling, the ¹³C transition frequencies are:
```
ν_α = ν₀ + d(3cos²θ − 1)/2    (¹H in |α⟩ state)
ν_β = ν₀ − d(3cos²θ − 1)/2    (¹H in |β⟩ state)
```
Splitting = |d(3cos²θ − 1)|

### 05a — β = 0° (tensor z-axis || B₀)
```
Splitting = |2d| ≈ 60.4 kHz
```
**Expected**: ¹³C doublet with splitting **≈ 60.4 kHz** (peaks at ±30.2 kHz).

**Current Soprano output**:
```
dipole 1 2 -30210.667268 90.000000 0.000000 0.000000
```
✓ Correct (d in Hz).

### 05b — β = 90° (tensor z-axis ⊥ B₀)
```
Splitting = |d| ≈ 30.2 kHz
```
**Expected**: ¹³C doublet with splitting **≈ 30.2 kHz** (peaks at ±15.1 kHz).

**Current Soprano output**:
```
dipole 1 2 -30210.667268 90.000000 90.000000 0.000000
```
✓ Correct.

---

## Test 06 — Isotropic J-Coupling

**System**: ¹H–¹³C pair, J_iso = +100 Hz, no anisotropy

**Expected**: ¹³C doublet with splitting **100 Hz** (peaks at ±50 Hz).

**Current Soprano output**:
```
jcoupling 1 2 100.000000 0.000000 0.000000 0.000000 0.000000 0.000000
```
✓ Correct.

---

## Test 07 — Anisotropic J-Coupling

**System**: ¹H–¹³C pair, J_iso = 0 Hz, ζ = +200 Hz, η = 0

For J-coupling, SIMPSON v6.0.2's `aniso` parameter is **ζ/2** (half the reduced anisotropy).
The effective splitting is:
```
Splitting = |aniso × (3cos²β − 1)|
```

### 07a — β = 0°
```
Splitting = |(200/2) × (3·1 − 1)| = 200 Hz
```
**Expected**: ¹³C doublet with splitting **200 Hz** (peaks at ±100 Hz).

**Current Soprano output**:
```
jcoupling 1 2 0.000000 100.000000 0.000000 90.000000 0.000000 0.000000
```
✓ Correct (aniso = ζ/2 = 100 Hz).

### 07b — β = 90°
```
Splitting = |(200/2) × (0 − 1)| = 100 Hz
```
**Expected**: ¹³C doublet with splitting **100 Hz** (peaks at ±50 Hz).

**Current Soprano output**:
```
jcoupling 1 2 0.000000 100.000000 0.000000 90.000000 90.000000 0.000000
```
✓ Correct (aniso = ζ/2 = 100 Hz).

---

## Test 08 — Quadrupole-Dipole Cross-Term (13C–14N under MAS)

**System**: 13C–14N pair, quadrupole on 14N (Cq = 1.18 MHz, η = 0.54, β = 5°), dipolar coupling d = −660.2 Hz. MAS at 12 kHz with powder averaging (`rep30`).

**Physics**: Under fast MAS the first-order dipolar and quadrupolar (I = 1) terms both average to zero. The only surviving effect on the observed 13C spectrum is the second-order quadrupole–dipole cross-term, which produces a characteristic broadened centreband.

**Verification**: The test checks that:
1. `to_simpson()` automatically emits `quadrupole_x_dipole 2 1`
2. Simpson accepts the keyword and runs without error
3. The powder-averaged spectrum is broadened (width > 10 Hz) rather than a single delta peak at 0 Hz

Without `quadrupole_x_dipole` the powder-averaged spectrum collapses to a spike at 0 Hz.

**Current Soprano output**:
```
spinsys {
channels 13C 14N
nuclei 13C 14N
quadrupole 2 2 1180000.0 0.54 0.0 5.0 180.0
dipole 1 2 -660.2 90.0 0.0 0.0
quadrupole_x_dipole 2 1
}
```
✓ Correct — cross-term keyword is present and Simpson runs successfully.

---

## Summary of Fixes Applied

| Interaction | Original Bug | Fix | Status |
|-------------|-------------|-----|--------|
| Dipolar | Multiplied d by 2π | Removed 2π factor | ✓ Fixed |
| J-coupling anisotropy | Output Δ (full anisotropy) | Output ζ/2 (half reduced anisotropy) | ✓ Fixed |
| Shift anisotropy | Output ζ (ppm) | Output ζ (ppm) | ✓ Correct |
| Quadrupolar Cq | Output Cq (Hz) | Output Cq (Hz) | ✓ Correct |
| Euler angles | ZYZ passive, degrees | ZYZ passive, degrees | ✓ Correct |
