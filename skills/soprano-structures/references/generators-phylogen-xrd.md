# Structure Generators, Phylogenetic Clustering & XRD Reference

## 1. Structure Generators (`soprano.collection.generate`)

All generators return Python iterators of `ase.Atoms` — pass them directly to `AtomsCollection(...)`:

| Generator | Signature & purpose |
|---|---|
| `rattleGen` | `rattleGen(struct, amplitude=0.01, n=100, method="uniform")` — random atomic displacements (`"uniform"` or `"normal"`) |
| `defectGen` | `defectGen(struct, defect, poisson_r=None, avoid_atoms=True, vdw_set="csd", vdw_scale=1, max_attempts=30)` — insert a point defect atom using Poisson-sphere sampling |
| `additionGen` | `additionGen(struct, add, to_addition=None, n=1, add_r=1.2, accept=None)` — add atom(s) near selected host sites |
| `substitutionGen` | `substitutionGen(struct, subst, to_replace=None, n=1, accept=None, max_attempts=0, random=False)` — substitute selected atoms with another species |
| `linspaceGen` | `linspaceGen(struct_0, struct_1, steps=10, periodic=False)` — linear interpolation between two structures |
| `transformGen` | `transformGen(struct_0, transform, steps=10)` — apply a `Translate`/`Rotate`/`Mirror` transform in steps |
| `molecularNeighbourhoodGen` | `molecularNeighbourhoodGen(struct, mols, central_mol=0, max_R=10, method="com", use_supercell=False)` — build molecular shells around a central molecule |
| `airssGen` | `airssGen(input_file, n=100, ...)` — wrap AIRSS `buildcell` (requires external `buildcell` executable on `PATH`) |

```python
from ase.io import read
from soprano.collection import AtomsCollection
from soprano.collection.generate import rattleGen, substitutionGen
from soprano.selection import AtomSelection

atoms = read("mol_crystal.cif")

# 10 rattled copies
rattled = AtomsCollection(rattleGen(atoms, amplitude=0.05, n=10))

# All single C -> N substitutions on carbon sites
c_sel = AtomSelection.from_element(atoms, "C")
subs = AtomsCollection(substitutionGen(atoms, subst="N", to_replace=c_sel, n=1))
```

## 2. Phylogenetic Clustering (`soprano.analyse.phylogen`)

Cluster crystal ensembles (e.g. polymorphs, AIRSS structures) by combining structural "genes" with weights:

```python
from soprano.analyse.phylogen import PhylogenCluster, Gene, GeneDictionary

# Inspect available built-in gene names
# print(sorted(GeneDictionary._gene_dictionary.keys()))

genes = [
    Gene("latt_abc_len", weight=1.0),
    Gene("linkage_list", weight=2.0),
    Gene("coord_histogram", weight=1.0, params={"s1": "C", "s2": "H"}),
]

pc = PhylogenCluster(coll, genes=genes)

# Hierarchical clustering (SciPy linkage)
# Returns (cluster_labels_array, list_of_index_arrays)
labels, slices = pc.get_hier_clusters(t=0.5, method="single")

# K-means clustering
k_labels, k_slices = pc.get_kmeans_clusters(n=3)

# 2D dimensionality reduction for visualization ("total-principal", "clafique", ...)
x, y = pc.create_mapping(method="total-principal")  # shape (2, N)

# Cluster summary statistics & elbow analysis
stats = pc.get_cluster_stats((labels, slices))
```

### Built-in Genes (`GeneDictionary`)

- **Lattice & energy:** `energy`, `latt_abc_len`, `latt_abc_ang`, `latt_cart`
- **Coordination & geometry:** `linkage_list`, `coord_histogram`, `bond_order_pars`
- **Hydrogen bonds:** `hbonds_tot_n`, `hbonds_length`, `hbonds_angle`, `hbonds_fingerprint`, `hbonds_site_compare`, `hbonds_site_reference`
- **Molecules:** `molecule_number`, `molecule_mass`, `molecule_com_linkage`, `molecule_rot_linkage`
- **Defects:** `defect_asymmetric_fpos`, `defect_asymmetric_fdist`

> `pc.get_sklearn_clusters(method, params={})` supports scikit-learn clustering algorithms (e.g. DBSCAN, HDBSCAN) when `scikit-learn` is installed.

## 3. Powder XRD Simulation (`soprano.calculate.xrd`)

Requires a symmetry backend (`pip install soprano[moyo]` or `[spglib]`) when computing peaks directly from an `ase.Atoms` structure.

```python
import numpy as np
from ase.io import read
from soprano.calculate.xrd import XRDCalculator

atoms = read("mol_crystal.cif")

# Default wavelength is Cu K-alpha (1.54056 Å)
xrd = XRDCalculator(lambdax=1.54056, backend="auto")

# Compute theoretical Bragg peaks (XraySpectrum named tuple: theta2, hkl, hkl_unique, invd, intensity, lambdax)
peaks = xrd.powder_peaks(atoms)

# Simulate continuous broadened powder pattern (XraySpectrumData: theta2, intensity)
th2_axis = np.linspace(5.0, 50.0, 1000)
spec, peaks_in_range = xrd.spec_simul(peaks, th2_axis)

# Optional: Le Bail fit against an experimental 2-theta / intensity dataset
# exp_spec = xrd.exp_dataset(exp_th2, exp_intensity)
# fitted_spec, fitted_peaks, rwp = xrd.lebail_fit(peaks_in_range, exp_spec)
```

Use `xrd.set_peak_func(peak_func, peak_f_args)` to customize peak broadening profiles (Gaussian, Lorentzian, pseudo-Voigt).
