# Chemical shift referencing

Soprano converts magnetic shieldings σ (ppm) to chemical shifts δ (ppm) with
a reference shielding σ_ref per element. There are two formulas, and which one
an element gets depends on whether you give it a gradient.

## Full formula

An element with a reference and no gradient uses

$$
\delta = \frac{\sigma_{ref} - \sigma}{1 - \sigma_{ref}\times 10^{-6}}
$$

The denominator is there because ppm is a frequency ratio, not a linear
offset. Use this formula when you reference against a single compound, for
example TMS for ¹H and ¹³C.

## Linear fit

An element with a gradient m uses

$$
\delta = \sigma_{ref} + m\,\sigma
$$

with no denominator. Here σ_ref is the intercept and m the slope of a
calibration line: plot computed shieldings against experimental shifts for a
set of reference compounds and fit a straight line. Slopes usually come out
close to -1 (for example -0.98 for ¹³C). Any denominator correction is already
absorbed into the fitted intercept and slope, which is why the linear form
leaves it out.

## Choosing the mode

The choice is made per element, not for the whole structure:

- `gradients=None` or `{}`: every element uses the full formula.
- An element with a gradient uses the linear fit.
- An element with a reference but no gradient uses the full formula, even when
  other elements have gradients. Soprano warns when this happens.

So `references={'H': 31, 'C': 170}` with `gradients={'C': -0.98}` gives a linear
fit for carbon and the full formula for hydrogen.

## Warnings

Soprano issues a `UserWarning` when

- some referenced elements have no gradient while others do. The warning names
  the elements that fall back to the full formula;
- a gradient is exactly -1. That is still a linear fit, δ = σ_ref − σ, without
  the denominator. Scripts written for Soprano 0.11 often pass -1 expecting the
  full formula; if that is what you want, leave the gradient out;
- a gradient is outside -1.5 to -0.5, which usually means a typo.

Every shift calculation also logs one INFO line with the mode of each element,
for example `H=full formula, C=linear fit (m=-0.98)`.

## Python

```python
from ase.io import read
from soprano.properties.nmr import MSShift

atoms = read("ethanol.magres")
delta = MSShift.get(
    atoms,
    references={"H": 31, "C": 170},
    gradients={"C": -0.98},
)
```

The hydrogens use the full formula with σ_ref = 31 ppm. The carbons use the
linear fit with intercept 170 ppm and slope -0.98.

## Command line

```bash
soprano nmr file.magres --references C:170,H:31 --gradients C:-0.98
```

Leave out `--gradients` to use the full formula for every element.

## Changed in 0.12.0

In 0.11 there was a single formula, δ = (σ_ref + m σ)/(1 − σ_ref × 10⁻⁶), and the
gradient m defaulted to -1. From 0.12 a gradient selects the linear fit, which
has no denominator.

- If you never passed gradients, nothing changes: the default is now
  `gradients=None`, which gives the same numbers as the old default.
- If you passed gradients, including an explicit -1, the shifts for those
  elements change by roughly σ_ref × 10⁻⁶ × δ (about 0.02 ppm for ¹³C at
  100 ppm). To reproduce the 0.11 numbers exactly, divide both the reference
  and the gradient by (1 − σ_ref × 10⁻⁶) before passing them.
- Elements missing from a partial `gradients` dict used to get m = -1. They now
  use the full formula, which is what m = -1 gave in 0.11, so their numbers do
  not change.
