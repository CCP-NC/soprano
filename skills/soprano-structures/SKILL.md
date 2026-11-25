---
name: soprano-structures
description: Use when working with crystal structure ensembles or general crystallographic analysis in Soprano — AtomsCollection, AtomsProperty, AtomSelection, SiteMap, space-group/Wyckoff symmetry analysis (SymmetryDataset, WyckoffPoints, moyo/spglib backends), molecules/bonds/H-bonds (Molecules, Bonds, HydrogenBonds, soprano splitmols), periodic minimum-image distances (minimum_periodic), structure generators (rattleGen, defectGen, substitutionGen, additionGen, linspaceGen, transformGen, molecularNeighbourhoodGen, airssGen), phylogenetic clustering (PhylogenCluster, Gene), or X-ray diffraction (XRDCalculator). For .magres NMR extraction, use soprano-nmr instead. For spin systems, tensor Euler conventions, or SIMPSON/mrsimulator export, use soprano-spin-systems.
---

# Soprano Crystal Structures, Collections & Analysis

Soprano extends ASE's `Atoms` to large ensembles of periodic crystal structures (e.g. AIRSS searches, polymorph sets, MD snapshots) with periodic-boundary-safe properties, symmetry backends, structure generators, phylogenetic clustering, and powder XRD simulation.

## Core Workflows

### 1. Ensembles with `AtomsCollection` & `AtomsProperty`

Avoid manual Python loops over ASE `Atoms` lists when `AtomsCollection` and `AtomsProperty` do it natively:

```python
import glob
from soprano.collection import AtomsCollection
from soprano.properties.basic import LatticeABC, CalcEnergy
from soprano.properties.symmetry import SymmetryDataset

# Load from file paths, glob, generator, or list of ase.Atoms
coll = AtomsCollection(glob.glob("structures/*.res"), progress=True)

# Compute & store properties across the entire collection
abc = LatticeABC.get(coll, store_array=True)
coll.set_array("volume", coll.all.get_volume())
sg = [ds.international for ds in SymmetryDataset.get(coll)]
coll.set_array("spacegroup", sg)

# Sort (by a scalar per-structure array), filter, and split into classes
coll_sorted = coll.sorted_byarray("volume", reverse=False)
large = coll.filter(lambda s: len(s) > 10)
by_sg = coll.classify(coll.get_array("spacegroup"))

# Portable persistence (prefer save_tree/load_tree over pickle .save/.load)
coll.save_tree("saved_coll", save_format="cif")
coll_reloaded = AtomsCollection.load_tree("saved_coll", load_format="cif")
```

### 2. Selecting Atoms (`AtomSelection`) & Tracking Derived Indices (`SiteMap`)

```python
from ase.io import read
from soprano.selection import AtomSelection
from soprano.sitemap import SiteMap

atoms = read("mol_crystal.cif")

# Selection strings, elements, set algebra (+ union, - diff, * intersection)
sel_c = AtomSelection.from_element(atoms, "C")
sel_sub = AtomSelection.from_selection_string(atoms, "C.1-4,N")
sel_sphere = AtomSelection.from_sphere(atoms, center=[0, 0, 0], r=4.0, periodic=True)
sel_combined = (sel_c + sel_sub) * sel_sphere

# Track where derived sites came from in the original Atoms object
smap = SiteMap.from_selection(sel_c, atoms)
sub = sel_c.subset(atoms)
orig_indices = smap.members(0)  # source atom index(es) behind derived site 0
```

### 3. Splitting Molecular Crystals (`Molecules` & `soprano splitmols`)

**CLI** (works on any ASE-readable format — `.cif`, `.magres`, `.res`, `POSCAR`, etc.):

```bash
# Split into individual extended-XYZ files (reassembled across periodic boundaries)
soprano splitmols crystal.cif -o mols_out

# Output CIF files centered with 10 Å vacuum (note: `-f cell` requires a local CASTEP install for ASE's writer)
soprano splitmols crystal.cif -o mols_out -f cif -c --vacuum 10.0

# Zeolite + guest molecule: scale vdW radii and keep framework within the cell
soprano splitmols ZSM-5_withH2O.cif --vdw-scale 1.3 --no-cell-indices -o split_zeolite
```

**Python:**

```python
from ase.io import read
from soprano.properties.linkage import Molecules

atoms = read("mol_crystal.cif")
mols = Molecules.get(atoms)  # list[AtomSelection] (cell shifts stored internally in _arrays)

# CRITICAL: pass use_cell_indices=True so boundary-crossing molecules stay intact!
mol0_atoms = mols[0].subset(atoms, use_cell_indices=True)
```

### 4. Symmetry, Generators, Clustering & XRD

```python
import numpy as np
from ase.io import read
from soprano.collection import AtomsCollection
from soprano.collection.generate import rattleGen
from soprano.properties.symmetry import SymmetryDataset
from soprano.analyse.phylogen import PhylogenCluster, Gene
from soprano.calculate.xrd import XRDCalculator

atoms = read("mol_crystal.cif")

# Symmetry (auto prefers moyo, falls back to spglib)
ds = SymmetryDataset.get(atoms, symprec=1e-4, backend="auto")
print(ds.international, ds.hall_number)

# Generate perturbed ensemble & cluster phylogenetically
coll = AtomsCollection(rattleGen(atoms, amplitude=0.05, n=10))
pc = PhylogenCluster(coll, genes=[Gene("latt_abc_len"), Gene("linkage_list")])
cluster_ids, slices = pc.get_hier_clusters(t=0.5, method="single")

# Simulate powder XRD pattern
xrd = XRDCalculator(lambdax=1.54056, backend="auto")
peaks = xrd.powder_peaks(atoms)
th2 = np.linspace(5, 50, 500)
spec, peaks_used = xrd.spec_simul(peaks, th2)  # spec.theta2, spec.intensity
```

## Critical Guardrails & Gotchas

1. **Symmetry backends are optional since v0.11.0.**
   `SymmetryDataset`, `WyckoffPoints`, `UniqueSites`, `XRDCalculator`, and symmetry-based site reduction require either `moyopy` or `spglib`. Recommend `pip install soprano[moyo]` (Rust-based, ~4× faster) or `pip install soprano[spglib]`. Override per call with `backend="moyo"` or `backend="spglib"`.
2. **Always use `use_cell_indices=True` when subsetting molecules.**
   `Molecules.get(atoms)` returns `AtomSelection` objects with periodic cell shifts stored internally (`_arrays`, not a `.cell_indices` attribute). Calling `mol.subset(atoms)` without `use_cell_indices=True` leaves atoms broken across periodic cell boundaries.
3. **Never compute periodic distances by naive Cartesian subtraction.**
   Use `soprano.utils.minimum_periodic(v, latt_cart)` (or built-in properties like `Bonds`, `LinkageList`, `HydrogenBonds`) so closest periodic images are always accounted for.
4. **Prefer `save_tree` / `load_tree` over `save` / `load`, and watch collection indexing.**
   - `AtomsCollection.save()` uses Python pickle and is fragile across Soprano/ASE versions. Use `save_tree()` for portable structure folders + metadata; pass `safety_check=2` in non-interactive scripts to overwrite an existing collection folder without prompting.
   - Indexing `coll[0]` returns a 1-structure `AtomsCollection`, **not** an `ase.Atoms` — use `coll.structures[0]` (or `list(coll)[0]`) to get a single `Atoms` object.
5. **Custom `AtomsProperty` parameters:**
   You can either pass kwargs directly to `.get(s, param=val)` or instantiate a reusable property `prop = Bonds(vdw_scale=1.1); prop(atoms)` (or `prop(coll)`).
   Called on an **instance**, `.get` honours the instance's parameters (`Bonds(vdw_scale=1.1).get(atoms)` matches `prop(atoms)`); keyword arguments still override them, and `Bonds.get(atoms)` on the class uses the defaults. `mean()` on the base class follows the same rule.
   Watch out: if a property has its own `selection` parameter, passing `selection=` to `.get()` delegates to that parameter rather than subsetting `s` first.
6. **`UniqueSites(override_cif=True)`** now really uses the symmetry reduction (it used to return the CIF-label grouping in both branches); CIF labels that agree with the symmetry grouping are not reported as a mismatch.
7. **Optional undeclared extras for graph/ML features:**
   `BondGraph` requires `networkx`; `PhylogenCluster.get_sklearn_clusters()` requires `scikit-learn`. Neither is installed by default with `pip install soprano`.
8. **Avoid legacy standalone entry points.**
   Do not recommend deprecated standalone scripts (`phylogen`, `vasp2cell`, `extract_nmr`, `magresaverage`) or the legacy GULP interface unless the user specifically asks for them; prefer the Python API and the unified `soprano` CLI.

## Detailed References

- For full `AtomsCollection` methods, `AtomsProperty` catalog (`basic`, `labeling`, `linkage`, `symmetry`, `order`, `transform`), `AtomSelection` constructors, `SiteMap`, `minimum_periodic`, and `soprano splitmols` options, read [references/collections-and-properties.md](references/collections-and-properties.md).
- For all structure generators, `PhylogenCluster` & built-in `GeneDictionary` genes, and `XRDCalculator` (peak calculation, spectrum simulation, Le Bail fitting), read [references/generators-phylogen-xrd.md](references/generators-phylogen-xrd.md).
