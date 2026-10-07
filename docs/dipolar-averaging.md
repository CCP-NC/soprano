# Dipolar couplings and methyl group rotation

Methyl groups rotate fast on the NMR timescale at essentially any temperature
where spectra are recorded. The three-fold hop rate exceeds 10⁶ s⁻¹ well below
100 K, while the ¹H-¹H dipolar couplings at stake are of order 10⁴ Hz. A
simulation built from a static DFT geometry therefore reports couplings that
the experiment never sees: the measurement averages over the rotation, and the
simulation must do the same if the two are to be compared. This document sets
out the theory of that average, the regimes in which different averages apply,
what Soprano now does, and what it deliberately does not attempt.

## The static coupling

For two nuclei i and j at distance r, the dipolar coupling constant is

$$
d_{ij} = -\frac{\mu_0 \hbar \gamma_i \gamma_j}{8\pi^2 r_{ij}^3}
$$

in Hz, and the full interaction is the traceless symmetric tensor

$$
D_{ij} = d_{ij}\,(3\,\hat{r}_{ij}\otimes\hat{r}_{ij} - \mathbb{1})
$$

with eigenvalues $(-d, -d, 2d)$ and the unique axis along the internuclear
vector. Note the sign: for two protons, $\gamma_i\gamma_j > 0$ and $d$ is
negative. Soprano keeps this sign throughout; only contour intensities take
the absolute value at plot time.

## Motional averaging

When a group of spins exchanges positions faster than the inverse of the
coupling it modulates, the observed interaction is the time average of the
coupling *tensor*, not of the coupling constant. That is, one must average the
full $3\times 3$ matrices, orientation and all, and only then extract an
effective constant. The distinction matters because tensors pointing in
different directions partially cancel; averaging the scalar magnitudes ignores
this cancellation entirely and always overestimates the residual coupling.

Three regimes are worth separating.

**Rigid limit.** If the hop rate is slow compared with the coupling (deep
cryogenic temperatures, or heavier rotors such as ND₃ near their tunnelling
regime), each conformer contributes its static tensor and no averaging is
appropriate. Soprano's default behaviour, with no `average_group` requested,
corresponds to this limit.

**Fast exchange.** When the rate exceeds the coupling by an order of magnitude
or more, which for CH₃ is the case at nearly all practically relevant
temperatures, the observed tensor is

$$
\bar{D} = \frac{1}{N}\sum_{(i,j)} D_{ij}
$$

where the sum runs over the exchanging site pairs. The effective constant is
half the eigenvalue of $\bar{D}$ with the largest magnitude, sign retained.
Three cases follow from the same formula:

1. *External spin k to a methyl proton.* The proton visits three sites, so
   $\bar{D} = \tfrac{1}{3}\sum_i D_{ki}$. The residual tensor is in general
   not axial ($\eta \neq 0$) and its effective constant is always smaller in
   magnitude than the mean of the three static constants.
2. *Two protons within the same methyl.* The H-H vectors are perpendicular to
   the C₃ axis, and rotation scales the coupling by
   $P_2(\cos 90°) = -\tfrac{1}{2}$: a static coupling of $-21.5$ kHz becomes
   $+10.7$ kHz. The tensor average over the three edge vectors reproduces
   this factor without special-casing, since the three edges at 120° to one
   another average to an axial tensor along the rotation axis.
3. *Methyl to methyl.* With uncorrelated hops the average runs over all
   $3\times 3 = 9$ site pairs.

**Intermediate exchange.** Between the two limits the lineshape is genuinely
dynamic and no single residual coupling describes it. Soprano does not model
this regime; if the hop rate is comparable to the coupling, a lineshape
simulation is needed rather than a rescaled constant.

## Why not one representative hydrogen, or a scalar mean?

Two shortcuts suggest themselves and both fail quantitatively. Keeping a
single methyl hydrogen (as a structure-reduction step does for positions)
makes the coupling depend on which hydrogen happens to survive. Averaging the
scalar constants ignores orientational cancellation. Measured on the ethanol
test structure (`tests/test_data/ethanol.magres`), for the three non-methyl
protons against the CH₃ group, in kHz:

| external H | static couplings to H0, H1, H2 | scalar mean | tensor average |
|-----------|-------------------------------|-------------|----------------|
| H3 | $-7.60$, $-7.52$, $-4.16$ | 6.43 | 5.09 |
| H4 | $-7.57$, $-4.17$, $-7.69$ | 6.47 | 5.06 |
| H5 | $-2.52$, $-7.50$, $-4.58$ | 4.86 | 4.17 |

The single-hydrogen shortcut lands anywhere from 40% below to 50% above the
tensor average, depending on an arbitrary ordering choice. The scalar mean
overestimates by 17-28% here; the discrepancy grows as the external spin
approaches the group, precisely the couplings that dominate a dipolar-weighted
spectrum.

## What Soprano does

The function `soprano.properties.nmr.averaged_dipolar_coupling` computes the
fast-exchange average for two groups of sites:

```python
from ase import io
from soprano.properties.nmr import averaged_dipolar_coupling

atoms = io.read("ethanol.magres")
methyl = [0, 1, 2]

# External proton to the methyl group
d_eff, D_avg = averaged_dipolar_coupling(atoms, [3], methyl)
print(d_eff)   # -5091 Hz: signed effective constant

# Intra-methyl residual: the -1/2 factor emerges from the geometry
d_eff, _ = averaged_dipolar_coupling(atoms, methyl, methyl)
print(d_eff)   # +10766 Hz, versus -21500 Hz static
```

Internuclear vectors use the minimum image convention, matching
`DipolarCoupling`. Identical index pairs are skipped and each unordered pair
counts once; passing the same group twice therefore gives the intra-group
average, as in the second example.

In the 2D spectrum pipeline, requesting functional-group averaging with the
dipolar weighting

```python
from soprano.calculate.nmr import NMRData2D

data = NMRData2D(
    atoms,
    xelement="H",
    references={"H": 30.0},
    correlation_strength_metric="dipolar",
    average_group="CH3",
)
```

now evaluates merged peak strengths through the tensor average, computed on
the original structure so that every internuclear vector is a true geometric
distance. Peak positions remain the mean of the member shifts, and
multiplicities still record the group size. The same applies on the command
line:

```bash
soprano plotnmr -p 2D -x H --weight-by dipolar --average-group CH3 structure.magres
```

Other weighting metrics (`distance`, `jcoupling`, `dipolar_rss`) keep the
multiplicity-weighted arithmetic mean, since tensor averaging has no meaning
for them; `dipolar_rss` already sums over all group members by construction.

One path is deliberately left alone. When a structure is *merged* during
extraction (`--average-group` in `soprano nmr`), the merged site keeps the
position of its first hydrogen, and any pair quantity computed downstream from
that merged structure inherits the single-hydrogen error described above.
Merging is inherently lossy for pair properties; the fix is not to patch the
merged structure but to compute dipolar quantities against the source
structure, which is what the 2D pipeline does through its site map.

## Caveats

The average implemented here is a rigid three-site hop on a fixed DFT
geometry. Vibrational averaging, which shortens effective H-H distances and
can change couplings by several percent, is not included; nor are librations
of the methyl axis, which scale the residual coupling by a further factor
slightly below one. Both effects are smaller than the factor-of-two errors
the tensor average corrects, but they set the accuracy floor. We would also
caution against applying `average_group` to groups that are not in fast
exchange at the temperature of interest: the flag asserts a dynamical regime,
and Soprano has no way to check that assertion against reality.
