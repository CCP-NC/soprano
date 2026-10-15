# Dipolar couplings and methyl group rotation

Methyl groups are usually in fast rotation on the NMR timescale at the
temperatures where spectra are recorded. Whether the three-fold hop rate exceeds
10⁶ s⁻¹ at a given temperature depends on the rotational barrier, which varies
widely between compounds. The ¹H-¹H dipolar couplings at stake are of order
10⁴ Hz. A
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
this cancellation entirely and never underestimates the residual coupling.

Three regimes are worth separating.

**Rigid limit.** If the hop rate is slow compared with the coupling (deep
cryogenic temperatures, or heavier rotors such as ND₃ near their tunnelling
regime), each conformer contributes its static tensor and no averaging is
appropriate. Soprano's default behaviour, with no `average_group` requested,
corresponds to this limit.

**Fast exchange.** When the rate exceeds the coupling by an order of magnitude
or more, the observed tensor is

$$
\bar{D} = \frac{1}{N}\sum_{(i,j)} D_{ij}
$$

where the sum runs over the exchanging site pairs. For ideal three-fold hops
between equivalent sites (120° jumps) this average is exact within the rigid
three-site model; the approximations that matter in practice are the fixed
geometry and the neglect of vibration and libration (see Caveats). The
effective constant is half the eigenvalue of $\bar{D}$ with the largest
magnitude, sign retained. When $\bar{D}$ is axial, $d_{\text{eff}}$ is the
coupling constant of the equivalent axial interaction. When it is not
($\eta \neq 0$), $\lambda_{\max}/2$ is a convention: it summarises the
averaged tensor but does not describe a single coupling. Three cases follow from
the same formula:

1. *External spin k to a methyl proton.* The proton visits three sites, so
   $\bar{D} = \tfrac{1}{3}\sum_i D_{ki}$. The residual tensor is in general
   not axial ($\eta \neq 0$). Its effective constant is never larger in
   magnitude than the mean of the three static constants' magnitudes, with
   equality when the three vectors are parallel.
2. *Two protons within the same methyl.* The H-H vectors are perpendicular to
   the C₃ axis, and rotation scales the coupling by
   $P_2(\cos 90°) = -\tfrac{1}{2}$: a static coupling of about $-21.5$ kHz
   (between $-21.4$ and $-21.7$ kHz for the three edges of the ethanol test
   structure) becomes about $+10.8$ kHz. The three edges of the H₃ triangle lie in a
   plane perpendicular to the C₃ axis, so their tensor average is axial along
   that axis, and the averaging reproduces this factor without
   special-casing.
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
| H5 | $-2.52$, $-7.50$, $-4.57$ | 4.86 | 4.17 |

Distances are minimum-image, as in `DipolarCoupling`. The single-hydrogen
shortcut lands anywhere from 40% below to 52% above the tensor average,
depending on an arbitrary ordering choice. The scalar mean
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
print(d_eff)   # ≈ -5091 Hz: signed effective constant

# Intra-methyl residual: the -1/2 factor emerges from the geometry
d_eff, _ = averaged_dipolar_coupling(atoms, methyl, methyl)
print(d_eff)   # ≈ +10766 Hz, versus about -21.5 kHz static
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

### Spin-system export

`soprano spinsys --average-group CH3` follows the same rule. The shielding and
EFG tensors of a group are the mean of its member tensors, and its couplings to
every other site are the pair tensors averaged over the group's members in the
*source* structure (`averaged_dipolar_coupling_list` and
`averaged_j_coupling_list`), never read off a structure with the group merged
away. Symmetry copies of an ordinary site are not averaged over, since no motion
exchanges them. For a methyl and its own carbon this gives the expected residual
of $P_2(\cos 109.5°) = -\tfrac{1}{3}$ times the static C-H coupling, with the
sign reversed, where the first hydrogen alone would give the full static value.

There are two ways to turn a group into spins.

**Expanded** (the default whenever `--dip` or `--jcoupling` is given). Each
atom of the group stays a spin. All three hydrogens of a methyl get the same
group-mean shielding and EFG (what a three-site hop produces), keep their own
labels, and couple to every other spin through the group-averaged tensor. Spins
in the same group couple to each other through the intra-group average: about
$+10.8$ kHz for the methyl in the ethanol test structure. This is the faithful
representation. Three coupled spin-½ nuclei have the A₃ spin-state structure
that one spin cannot reproduce, which matters for ¹H line shapes, spin
diffusion, homonuclear recoupling and coupled ¹³C multiplets. The cost is a
larger spin system. With symmetry reduction, one physical group stands for each
class of equivalent groups, and `multiplicity` is the number of groups in the
class.

**Collapsed** (the default without couplings; `--collapse-group` forces it).
The group is one spin. For per-site shifts and tensors this loses nothing, and
it avoids writing the same site three times, which is why it is the default
there. With couplings it is an approximation (see below).

```python
merged, site_sets = nmr_extract_atoms(
    atoms, average_group="CH3", expand_groups=True, return_site_sets=True
)
couplings = averaged_dipolar_coupling_list(atoms, site_sets)
```

Collapsing a group that carries couplings loses three things:

- **Couplings inside a group are dropped.** A collapsed CH₃ is one spin, so the
  intra-methyl H-H term (the $+10.8$ kHz residual above) is absent. That does not
  matter for ¹³C- or ¹⁵N-detected simulations with ¹H decoupling, but it does for
  ¹H line shapes, spin diffusion or homonuclear recoupling.
- **The second moment is short by the group size.** A collapsed A$_n$ spin
  coupled to X carries $\bar d^2$ where the $n$ real spins carry $n\bar d^2$.
  Scaling the constant by $\sqrt{n}$ would restore the second moment, but not the
  line shape or the REDOR-type dephasing curve, so Soprano does not do it.
- **The averaged tensor is generally not axial.** This one applies to both
  modes. SIMPSON's `dipole` and mrsimulator's `dipolar` take one constant, so the
  writers keep the largest-magnitude eigenvalue and drop the asymmetry. The
  command warns when that is more than 1%.

### What is averaged, and what is not

The rule is the fast three-site jump average: the lab-frame mean of the member
tensors, with the central atom left alone.

- Each hydrogen visits all three sites, and the three sites have equal energy,
  so the mean is exact for a hop and nothing needs to be projected onto the C₃
  axis. If the crystal field breaks C₃ symmetry the mean is genuinely
  non-axial, and that is physical.
- The methyl carbon, or the nitrogen of an NH₃⁺, is unchanged by a hop (the
  configuration maps onto itself), so its static tensor already is the hopped
  average. Soprano does not average it.
- Each physical group is averaged on its own. Symmetry-equivalent methyls share
  a site only afterwards, by taking the first copy (or the mean, with
  `--mean-merge`). Averaging the lab-frame tensors of differently oriented
  methyls together would not be a C₃ average.

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
