# Collections, Properties, Selections & SiteMap Reference

## 1. `AtomsCollection` (`soprano.collection.AtomsCollection`)

| Method / Property | Description |
|---|---|
| `AtomsCollection(structures, info={}, cell_reduce=False, progress=False)` | Initialize from list of file paths, list of `ase.Atoms`, or a generator |
| `coll.all.<method>(...)` / `coll.all.map(fn, *args, **kwargs)` | Vectorized method calls or function mapping across all structures |
| `coll.set_array(name, a)` / `coll.get_array(name)` / `coll.has(name)` | Attach/retrieve per-structure arrays |
| `coll.sorted_byarray(name, reverse=False)` | Return a new collection sorted by a stored array |
| `coll.filter(filter_func)` | Keep structures where `filter_func(atoms)` is truthy |
| `coll.classify(classes)` | Split into `dict[class_label, AtomsCollection]` |
| `coll.chunkify(chunk_size=..., chunk_n=...)` | Split into a list of smaller `AtomsCollection` chunks |
| `coll1 + coll2` / `coll[i:j]` / `coll.structures[i]` / `len(coll)` | Concatenate, slice (`coll[0]` returns a 1-item `AtomsCollection`; use `coll.structures[0]` for an `ase.Atoms`), iterate |
| `coll.save_tree(path, save_format="cif", safety_check=3)` / `AtomsCollection.load_tree(path, load_format="cif")` | Portable directory-tree persistence (`safety_check=2` overwrites an existing collection folder without prompting) |
| `coll.save(fname)` / `AtomsCollection.load(fname)` | Pickle persistence (version-sensitive) |

## 2. `AtomsProperty` Catalog (`soprano.properties.*`)

Every property supports `Prop.get(atoms_or_coll, store_array=False, selection=None, **kwargs)` and reusable instantiation `p = Prop(name="custom", **kwargs); p(atoms_or_coll)`. `p.get(...)` called on an instance honours the instance parameters (keyword arguments still override them), so it matches `p(...)`; called on the class it uses the defaults.

- **`soprano.properties.basic`**: `CalcEnergy`, `LatticeABC`, `LatticeCart`, `NumAtoms`
- **`soprano.properties.symmetry`**:
  - `SymmetryDataset(symprec=1e-5, backend="auto")` → `SpacegroupDataset` with fields `international`, `hall_number`, `std_lattice`, `transformation_matrix`, `origin_shift`, `rotations`, `translations`, `equivalent_atoms`.
  - `WyckoffPoints(symprec=1e-5, backend="auto")` → list of `WyckoffPoint(fpos, pos, operations, hessian)`.
- **`soprano.properties.labeling`**:
  - `UniqueSites(symprec=1e-4, backend="auto", override_cif=False)` (with `override_cif=True` the symmetry reduction is used instead of CIF labels), `SiteLabels`, `MagresViewLabels`, `MoleculeSites`, `HydrogenBondTypes`, `CarbonHybridationState`.
- **`soprano.properties.linkage`**:
  - `Bonds(vdw_set="csd", vdw_scale=1.0, default_vdw=2.0, vdw_custom={}, return_matrix=False)`
  - `Molecules(vdw_set="csd", vdw_scale=1.0, default_vdw=2.0, vdw_custom={})` → `list[AtomSelection]` with `cell_indices`
  - `HydrogenBonds(hbond_elems=["O", "N"], max_length=3.5, max_angle=45.0, ...)` & `HydrogenBondsNumber`
  - `BondGraph` (requires `networkx`), `LinkageList`, `CoordinationHistogram`, `ElementPairs`, `DihedralAngleList`
  - Molecule-level: `MoleculeNumber`, `MoleculeMass`, `MoleculeCOM`, `MoleculeCOMLinkage`, `MoleculeQuaternion`, `MoleculeRelativeRotation`, `MoleculeSpectralSort`
- **`soprano.properties.order`**: `BondOrder` (Steinhardt bond-order invariants)
- **`soprano.properties.transform`**: `Translate`, `Rotate`, `Mirror` (return transformed `Atoms` objects)

## 3. `AtomSelection` (`soprano.selection.AtomSelection`)

Constructors:
- `AtomSelection.all(atoms)`
- `AtomSelection.from_element(atoms, "C")`
- `AtomSelection.from_selection_string(atoms, "C.1-4,H1a,N")`
- `AtomSelection.from_input(atoms, sel)` — accepts `AtomSelection`, `list[int]`, `int`, `str`, or `None`
- `AtomSelection.from_sphere(atoms, center, r, periodic=False, scaled=False)`
- `AtomSelection.from_box(atoms, abc0, abc1, periodic=False, scaled=False)`
- `AtomSelection.from_bonds(atoms, center, n, op="le")` — atoms within `n` bonds of index `center`
- `AtomSelection.from_array(atoms, name, value, op="eq")` — filter by an `atoms` array (`eq`, `ne`, `lt`, `le`, `gt`, `ge`)
- `AtomSelection.unique(atoms, symprec=1e-4)` — one atom per symmetry-equivalent site

Set operations & subsetting:
- `sel1 + sel2` (union), `sel1 - sel2` (difference), `sel1 * sel2` (intersection)
- `sel.subset(atoms, use_cell_indices=False)` — set `use_cell_indices=True` for `Molecules.get()` selections so periodic copies stay together.
- `sel.indices` — underlying numpy index array.

## 4. `SiteMap` (`soprano.SiteMap` / `soprano.sitemap.SiteMap`)

Lightweight, immutable index map between a source structure's **atoms** and a derived structure's **sites** (from selection or site merging):

```python
from soprano import SiteMap
from soprano.selection import AtomSelection

# Selection -> SiteMap
sel = AtomSelection.from_element(atoms, "H")
smap_sel = SiteMap.from_selection(sel, atoms)

# Tag-based merge -> SiteMap
smap_tags = SiteMap.from_tags(atoms.get_tags())

# Chain derivations: original atoms -> merged sites -> filtered sites
combined = smap_tags.compose(smap_sub)
combined.validate(atoms)       # raises ValueError if length mismatches
source_atoms = combined.members(0)  # np.ndarray of original atom indices
mult = combined.multiplicity        # atoms per site, np.ndarray
```

`SiteMap` rejects duplicate selection indices and non-integer maps, and does not freeze the caller's array. `NMRData2D` uses it to evaluate pair metrics on source atoms.

## 5. Periodic Minimum-Image Distances (`soprano.utils.minimum_periodic`)

```python
import numpy as np
from soprano.utils import minimum_periodic

# Shortest periodic displacement vectors between atom 0 and all atoms
raw_vecs = atoms.positions - atoms.positions[0]
pbc_vecs, cell_shifts = minimum_periodic(raw_vecs, atoms.get_cell()[:])
distances = np.linalg.norm(pbc_vecs, axis=1)
```

## 6. `soprano splitmols` CLI Quick Reference

| Flag | Default | Notes |
|---|---|---|
| `-o, --output-dir PATH` | `.` | Output directory |
| `-s, --seedname TEXT` | input stem | Base name for output files |
| `-f, --format TEXT` | `extxyz` | Any ASE output format (e.g. `cell`, `cif`, `xyz`) |
| `-c, --center` / `--vacuum FLOAT` | off | Center molecule and pad with vacuum (Å); `--vacuum` requires `-c` |
| `--cell TEXT` | original | Redefine cell: `"15"`, `"10 10 20"`, `"10 10 10 90 90 90"`, or 9 floats |
| `--cell-indices / --no-cell-indices` | `--cell-indices` | Reassemble boundary-crossing molecules; use `--no-cell-indices` for frameworks |
| `--vdw-set [ase\|jmol\|csd]` | `csd` | Base van der Waals radii set |
| `--vdw-scale FLOAT` / `--vdw-custom TEXT` | `1.0` / none | Scale radii or override per element (`H:1.1,Si:2.2`); use `-vvv` to inspect radii |
| `--view` / `--no-write` | off | Preview in ASE GUI without writing files |
