# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Utility functions for extracting and processing NMR data from ASE Atoms objects.

These helpers sit at the boundary between raw ASE structures and the pandas
DataFrames that the CLI and other tools consume.  They are intentionally
decoupled from any Click machinery so that they can be reused outside the CLI.

Typical usage::

    from soprano.nmr.extract import nmr_extract_multi, build_nmr_df

    dfs, images = nmr_extract_multi(["file.magres"])
    df = dfs[0]
"""

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from ase import Atoms, io
from ase.units import Bohr, Ha

from soprano.data.nmr import _get_isotope_list
from soprano.nmr.dataframe import (
    apply_df_filtering,
    average_quaternions_by_tags,
    expand_aliases,
    sortdf,
    units_rename,
)
from soprano.properties.labeling import MagresViewLabels, UniqueSites
from soprano.properties.linkage import Bonds, ElementPairs, Molecules
from soprano.properties.nmr import (
    EFGAsymmetry,
    EFGQuadrupolarConstant,
    EFGQuaternion,
    EFGNQR,
    EFGVzz,
    MSAnisotropy,
    MSAsymmetry,
    MSIsotropy,
    MSQuaternion,
    MSReducedAnisotropy,
    MSSkew,
    MSSpan,
)

from soprano.properties.nmr.ms import MSShift
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels, merge_first, merge_sites

# ---------------------------------------------------------------------------
# Module-level logger (callers can silence/configure via logging.getLogger)
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)


class MissingReferenceError(ValueError):
    """Raised when shifts are requested but some elements have no reference."""

# ---------------------------------------------------------------------------
# Column constants – also re-exported from soprano.scripts.nmr for compat
# ---------------------------------------------------------------------------
MS_MINIMAL_COLUMNS: List[str] = ["MS_shielding", "MS_anisotropy"]
EFG_MINIMAL_COLUMNS: List[str] = ["EFG_quadrupolar_constant", "EFG_asymmetry"]
NMR_COLUMN_ALIASES: dict = {
    "minimal": MS_MINIMAL_COLUMNS + EFG_MINIMAL_COLUMNS,
    "ms": MS_MINIMAL_COLUMNS,
    "efg": EFG_MINIMAL_COLUMNS,
    "angles": ["alpha", "beta", "gamma"],
    "essential": ["labels", "species", "multiplicity", "tags", "file"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_XHn_groups(
    atoms: Atoms,
    pattern_string: str,
    tags: Optional[np.ndarray] = None,
    vdw_scale: float = 1.0,
) -> list:
    """Find groups of atoms matching a functional-group pattern such as CH₃.

    The pattern is a string like ``"CH3"`` or ``"CH2"``.  Multiple patterns
    can be given comma-separated (e.g. ``"CH3,CH2"``).

    Args:
        atoms: Atoms object to search.
        pattern_string: Comma-separated functional-group patterns, e.g.
            ``'CH3'`` for methyl.  Each pattern must contain an element
            symbol, ``H`` and the H count (e.g. ``'CH3'``, ``'NH2'``).
        tags: Optional integer tag array (one per atom).  Defaults to
            ``np.arange(len(atoms))``.
        vdw_scale: Scale factor for van-der-Waals radii when determining
            connectivity.  Default ``1.0``.

    Returns:
        A list of lists of index groups, one outer list per pattern in
        *pattern_string*.
    """
    if tags is None:
        tags = np.arange(len(atoms))

    bcalc = Bonds(vdw_scale=vdw_scale, return_matrix=True)
    bonds, bmat = bcalc(atoms)
    all_groups = []
    for group_pattern in pattern_string.split(","):
        if "H" not in group_pattern:
            raise ValueError(
                f"{group_pattern} is not a valid group pattern "
                "(must contain an element symbol, H, and the number of H atoms. e.g. CH3)"
            )
        X, n = group_pattern.split("H")
        n = int(n)
        symbs = np.array(atoms.get_chemical_symbols())
        hinds = np.where(symbs == "H")[0]
        groups = []
        xinds = np.where(symbs == X)[0]
        xinds = xinds[np.where(np.sum(bmat[xinds][:, hinds], axis=1) == n)[0]]
        seen_tags = []
        for ix, xind in enumerate(xinds):
            bonded_hinds = np.where(bmat[xind][hinds] == 1)[0]
            group = list(hinds[bonded_hinds])
            assert len(group) == n
            match = []
            if len(seen_tags) > 0:
                match = np.where((seen_tags == tags[group]).all(axis=1))[0]
            if len(match) == 1:
                groups[match[0]] += group
            elif len(match) == 0:
                seen_tags.append(tags[group])
                groups.append(group)
            else:
                raise ValueError(f"Found multiple matches for {group_pattern}")
        all_groups.append(groups)
    return all_groups


# TODO: write guide for this on website...
NO_CIF_LABEL_WARNING = """
## Protip: ##
This .magres file doesn't seem to have CIF-stlye labels.
Using these is considered a good idea, but is not required.
You can export these automatically from a cif file using 
cif2cell. e.g. for CASTEP:

cif2cell mystructure.cif --export-cif-labels -p castep

"""

def reload_as_molecular_crystal(atoms: Atoms, force=False) -> Atoms:
    """
    If the atoms object is a molecular crystal, reload it with the correct
    connectivity.

    We check if there are any C-H bonds, and if so, we assume it's a molecular
    crystal and reload it as such. If there are no C-H bonds, we assume it's
    not and return the original atoms object, unless force=True.


    Args:
        atoms (ASE Atoms object): the atoms object to be reloaded.
        force (bool): if True, force the reload even if there are no C-H bonds.

    Returns:
        atoms (ASE Atoms object): the atoms object with the correct connectivity.
    """
    # save initial order
    atoms.set_array("order_tag", np.arange(len(atoms)))
    if has_CH_bonds(atoms) or force:
        # temporarily translate the atoms to the COM
        com = atoms.cell.T.dot([0.5, 0.5, 0.5]) - atoms.get_center_of_mass()
        atoms.translate(com)
        # let's assume this is an organic molecule/crystal
        # and try to reload the atoms object with the correct
        # connectivity:
        mols = Molecules.get(atoms)
        if len(mols) == 0:
            return atoms

        temp = mols[0].subset(atoms, use_cell_indices=True)
        if len(mols) > 1:
            for mol in mols[1:]:
                temp.extend(mol.subset(atoms, use_cell_indices=True))
        # restore original order
        temp = temp[temp.get_array("order_tag").argsort()]
        # restore original centering
        temp.translate(-com)
        atoms = temp

    return atoms

def has_CH_bonds(atoms: Atoms, rcut: float = 1.5) -> bool:
    """Check if the atoms object has bonds with CH atoms
    Does this simply by comparing the labels with the element symbols.
    If they're all the same, then so special labels are present.
    """
    # First check if it contains C and H elements
    symbols = atoms.get_chemical_symbols()
    if not ("C" in symbols and "H" in symbols):
        return False

    # Test the return_pairs=False option
    dists = ElementPairs.get(
        atoms, element1="C", element2="H", return_pairs=False, rcut=rcut
    )
    return len(dists) > 0


def label_atoms(atoms: Atoms, logger: Optional[logging.Logger] = None) -> Atoms:
    """Ensure *atoms* has CIF-style atom labels.

    If the structure already carries CIF labels nothing is changed.  Otherwise
    MagresView-style labels are computed and stored (warning the user via
    *logger* about best-practice).

    Args:
        atoms: The Atoms object to label.
        logger: Logger to use.  Falls back to the module logger when *None*.

    Returns:
        The (possibly modified) Atoms object.
    """
    log = logger or _logger

    if has_cif_labels(atoms):
        return atoms

    log.debug(NO_CIF_LABEL_WARNING)

    if atoms.has("magresview_labels"):
        labels = atoms.get_array("magresview_labels")
    else:
        labels = MagresViewLabels.get(atoms, store_array=True)

    labels = np.array(labels, dtype="U25")
    if atoms.has("labels"):
        atoms.set_array("labels", None)
    atoms.set_array("labels", labels)
    return atoms


def tag_functional_groups(
    average_group: str,
    atoms: Atoms,
    vdw_scale: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Atoms:
    """Tag groups of atoms (e.g. CH₃) so they can later be averaged together.

    Uses :func:`soprano.nmr.extract.find_XHn_groups` to identify
    XHₙ patterns and updates both the ``labels`` and ``tags`` arrays on the
    Atoms object in-place.

    Args:
        average_group: Comma-separated list of functional-group patterns,
            e.g. ``"CH3,CH2"``.
        atoms: Atoms object.  Must already carry a ``labels`` array.
        vdw_scale: Scaling factor applied to van-der-Waals radii when
            determining connectivity.  Defaults to ``1.0``.
        logger: Logger to use.  Falls back to the module logger when *None*.

    Returns:
        The modified Atoms object.
    """
    log = logger or _logger

    tags = atoms.get_tags() if atoms.has("tags") else np.arange(len(atoms))
    labels = atoms.get_array("labels").astype("U25")
    group_ids = np.full(len(atoms), -1, dtype=int)

    # Find each physical group from connectivity alone (default tags never
    # match), so that tensors are only ever averaged within one group.
    XHn_groups = find_XHn_groups(atoms, average_group, vdw_scale=vdw_scale)
    log.info(f"\nAveraging over functional groups: {average_group}")

    next_group_id = 0
    for ipat, pattern in enumerate(XHn_groups):
        if len(pattern) == 0:
            log.warning(
                f"No XHn groups found for pattern {average_group.split(',')[ipat]}"
            )
            continue
        log.debug(f"Found {len(pattern)} {average_group.split(',')[ipat]} groups")
        # Symmetry-equivalent groups share a tag, so they merge into one site
        # *after* each group has been averaged on its own.
        class_of: dict = {}
        for ig, group in enumerate(pattern):
            log.debug(f"    Group {ig} contains: {np.unique(labels[group])}")
            key = tuple(sorted(tags[group]))
            iclass = class_of.setdefault(key, len(class_of))
            group_ids[group] = next_group_id
            next_group_id += 1
            labels[group] = ",".join(np.unique(labels[group]))
            tags[group] = -(ipat + 1) * 1e5 - iclass

    atoms.set_array("labels", None)
    atoms.set_array("labels", labels, dtype="U25")
    atoms.set_tags(tags)
    atoms.set_array("xhn_group", None)
    atoms.set_array("xhn_group", group_ids)
    return atoms


def merge_tagged_sites(
    atoms_in: Atoms,
    merging_strategies: dict = {},
    symmetry_merging_strategies: dict = None,
) -> Atoms:
    """Merge atoms that share the same tag into a single representative site.

    Args:
        atoms_in: Atoms object with a ``tags`` array.  A copy is made
            internally so the original is not modified.
        merging_strategies: Passed verbatim to
            :func:`soprano.utils.merge_sites`.  Defines how to combine
            per-atom arrays when multiple atoms are collapsed into one.
        symmetry_merging_strategies: Optional dictionary of merging strategies for
            symmetry-equivalent sites (non-negative tags). If provided, these strategies
            are used for non-negative tags instead of ``merging_strategies``.

    Returns:
        A new Atoms object with duplicate-tag sites merged and sorted by tag.

    Sites come out ordered by sorted unique tag, which is what
    ``SiteMap.from_tags(atoms_in.get_tags())`` describes.  Build that map
    before calling if you need to know which atoms each site stands for; see
    :class:`soprano.sitemap.SiteMap`.
    """
    atoms = atoms_in.copy()
    if not atoms.has("tags"):
        return atoms

    unique_tags, unique_counts = np.unique(atoms.get_tags(), return_counts=True)
    for tag in unique_tags[unique_counts > 1]:
        # A negative tag marks functional groups.  Symmetry-equivalent groups
        # share it, but their tensors live in different orientations, so each
        # physical group is averaged on its own first (the fast-hop average)
        # and the per-group results are then merged like any symmetry copies.
        per_group = tag < 0 and atoms.has("xhn_group")
        if per_group:
            for gid in np.unique(atoms.get_array("xhn_group")[atoms.get_tags() == tag]):
                members = np.where(
                    (atoms.get_tags() == tag) & (atoms.get_array("xhn_group") == gid)
                )[0]
                if len(members) > 1:
                    atoms = merge_sites(
                        atoms, members, merging_strategies=merging_strategies,
                        keep_all=False,
                    )
        tag_idx = np.where(atoms.get_tags() == tag)[0]
        if len(tag_idx) < 2:
            continue
        # What is left to merge are symmetry copies (non-negative tags, or
        # already-averaged groups); a bare negative tag is one group to average.
        if (tag >= 0 or per_group) and symmetry_merging_strategies is not None:
            strategies = symmetry_merging_strategies
        else:
            strategies = merging_strategies
        atoms = merge_sites(
            atoms, tag_idx, merging_strategies=strategies, keep_all=False
        )

    if atoms.has("xhn_group"):
        del atoms.arrays["xhn_group"]
    return atoms[np.argsort(atoms.get_tags())]


def check_equivalent_sites_ms(
    atoms: Atoms,
    tags: np.ndarray,
    tolerance: float = 1e-3,
    tag: str = "ms",
) -> bool:
    """Return ``True`` if symmetry-equivalent sites have consistent MS isotropy.

    Compares the isotropic shielding of all atoms that share the same
    symmetry tag.  A warning should be issued by the caller when this
    returns ``False``.

    Args:
        atoms: Atoms object carrying the MS tensor array.
        tags: Integer tag array (one entry per atom).
        tolerance: Absolute tolerance for the shielding comparison.
        tag: Name of the MS tensor array stored on *atoms*.

    Returns:
        ``True`` if all equivalent sites agree within *tolerance*.
    """
    unique_sites, counts = np.unique(tags, return_counts=True)
    ms = MSIsotropy.get(atoms, tag=tag)
    for i in unique_sites[counts > 1]:
        idx = np.where(tags == i)[0]
        if not np.allclose(ms[idx], ms[idx[0]], atol=tolerance):
            return False
    return True


def get_ms_summary(
    atoms: Atoms,
    euler_convention: str,
    references: Optional[dict] = None,
    gradients: Optional[dict] = None,
    ms_tag: str = "ms",
) -> dict:
    """Build a dict of MS tensor properties for a single Atoms object.

    The dict is suitable for direct construction of a :class:`pandas.DataFrame`
    (each value is an array of length ``len(atoms)``).

    Args:
        atoms: Atoms object carrying the MS tensor array.
        euler_convention: Euler-angle convention (``"zyz"`` or ``"zxz"``).
        references: Dict mapping element symbol → reference shielding.
        gradients: Dict mapping element symbol → shielding gradient. Elements without
            one use the full NMR formula; ``None`` or ``{}`` uses it for all.
        ms_tag: Name of the MS tensor array.

    Returns:
        Dict with keys ``MS_shielding``, ``MS_shift``, ``MS_anisotropy``,
        ``MS_reduced_anisotropy``, ``MS_asymmetry``, ``MS_span``,
        ``MS_skew``, ``MS_alpha``, ``MS_beta``, ``MS_gamma``.
    """
    iso = MSIsotropy.get(atoms, tag=ms_tag)
    aniso = MSAnisotropy.get(atoms, tag=ms_tag)
    red_aniso = MSReducedAnisotropy.get(atoms, tag=ms_tag)
    asymm = MSAsymmetry.get(atoms, tag=ms_tag)
    span = MSSpan.get(atoms, tag=ms_tag)
    skew = MSSkew.get(atoms, tag=ms_tag)
    quat = MSQuaternion.get(atoms, tag=ms_tag)
    quat = average_quaternions_by_tags(quat, atoms.get_tags())
    alpha, beta, gamma = np.array(
        [q.euler_angles(mode=euler_convention) * 180 / np.pi for q in quat]
    ).T

    ms_summary = {
        "MS_shielding": iso,
        "MS_anisotropy": aniso,
        "MS_reduced_anisotropy": red_aniso,
        "MS_asymmetry": asymm,
        "MS_span": span,
        "MS_skew": skew,
        "MS_alpha": alpha,
        "MS_beta": beta,
        "MS_gamma": gamma,
    }

    if references:
        missing = sorted({el for el in atoms.get_chemical_symbols() if el not in references})
        if missing:
            raise MissingReferenceError(
                f"No reference shielding given for element(s) {', '.join(missing)}. "
                "Give a reference for every element in the structure, e.g. "
                "--references C:170,H:30."
            )
        # Elements without a gradient use the full NMR formula (see MSShift)
        ms_summary["MS_shift"] = MSShift.get(atoms, references=references, gradients=gradients)

    return ms_summary


def get_efg_summary(
    atoms: Atoms,
    isotopes: dict,
    euler_convention: str,
    tag: str = "efg",
) -> dict:
    """Build a dict of EFG tensor properties for a single Atoms object.

    The dict is suitable for direct construction of a :class:`pandas.DataFrame`
    (each value is an array of length ``len(atoms)``).

    Args:
        atoms: Atoms object carrying the EFG tensor array.
        isotopes: Dict mapping element symbol → isotope mass number.
        euler_convention: Euler-angle convention (``"zyz"`` or ``"zxz"``).
        tag: Name of the EFG tensor array.

    Returns:
        Dict with keys ``EFG_Vzz``, ``EFG_quadrupolar_constant``,
        ``EFG_asymmetry``, ``EFG_alpha``, ``EFG_beta``, ``EFG_gamma``,
        plus one ``EFG_NQR <transition>`` key per distinct NQR transition.
    """
    Vzz = EFGVzz.get(atoms, tag=tag)
    # Hartree/(e a0^2) in V/m^2: Ha [eV] / Bohr [Angstrom]^2 * 1e20 (1 A^2 = 1e-20 m^2)
    Vzz = Vzz * Ha / Bohr**2 * 1e20

    qP = EFGQuadrupolarConstant(isotopes=isotopes, tag=tag)
    qC = qP(atoms) / 1e6  # Hz → MHz

    eta = EFGAsymmetry.get(atoms, tag=tag)

    quat = EFGQuaternion.get(atoms, tag=tag)
    quat = average_quaternions_by_tags(quat, atoms.get_tags())
    alpha, beta, gamma = np.array(
        [q.euler_angles(mode=euler_convention) * 180 / np.pi for q in quat]
    ).T

    nqrs = EFGNQR.get(atoms, isotopes=isotopes, tag=tag)
    transition_keys = sorted({k for nqr in nqrs for k in nqr})
    nqr_dict = {}
    for k in transition_keys:
        values = np.array(
            [nqr[k] * 1e-6 if k in nqr else np.nan for nqr in nqrs]
        )
        nqr_dict[f"EFG_NQR {k}"] = values

    return {
        "EFG_Vzz": Vzz,
        "EFG_quadrupolar_constant": qC,
        "EFG_asymmetry": eta,
        "EFG_alpha": alpha,
        "EFG_beta": beta,
        "EFG_gamma": gamma,
        **nqr_dict,
    }


def build_nmr_df(
    atoms: Atoms,
    fname: str,
    isotopes: dict = {},
    references: dict = {},
    gradients: Optional[dict] = None,
    properties: List[str] = None,
    euler_convention: str = "zyz",
    property_tags: dict = {},
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Build a :class:`pandas.DataFrame` from the NMR properties of *atoms*.

    Args:
        atoms: Atoms object with labelled sites and NMR tensor arrays.
        fname: Source filename — stored as the ``file`` column for provenance.
        isotopes: Dict mapping element symbol → isotope mass number.
        references: Dict mapping element symbol → reference shielding.
        gradients: Dict mapping element symbol → shielding gradient. Elements without
            one use the full NMR formula; ``None`` or ``{}`` uses it for all.
        properties: List of properties to include, e.g. ``["ms", "efg"]``.
            Defaults to both when *None*.
        euler_convention: Euler-angle convention (``"zyz"`` or ``"zxz"``).
        property_tags: Dict mapping property name to its array tag on *atoms*,
            e.g. ``{"ms": "ms", "efg": "efg"}``.
        logger: Logger to use.  Falls back to the module logger when *None*.

    Returns:
        DataFrame with one row per (reduced) site and columns for all
        requested NMR properties.
    """
    if properties is None:
        properties = ["efg", "ms"]

    log = logger or _logger

    elements = atoms.get_chemical_symbols()
    isotopelist = _get_isotope_list(elements, isotopes=isotopes, use_q_isotopes=False)
    species = [f"{iso}{el}" for el, iso in zip(elements, isotopelist)]
    labels = np.asarray(atoms.get_array("labels"), dtype="U25")
    tags = atoms.get_tags()

    df = pd.DataFrame(
        {
            "indices": atoms.get_array("indices"),
            "original_index": np.arange(len(atoms)),
            "labels": labels,
            "species": species,
            "multiplicity": atoms.get_array("multiplicity"),
            "tags": tags,
        }
    )

    if atoms.has("magresview_labels"):
        df.insert(2, "MagresView_labels", atoms.get_array("magresview_labels"))

    df["file"] = fname

    if "ms" in properties:
        ms_tag = property_tags.get("ms", "ms")
        try:
            ms_summary = pd.DataFrame(
                get_ms_summary(atoms, euler_convention, references, gradients, ms_tag)
            )
            df = pd.concat([df, ms_summary], axis=1)
        except RuntimeError:
            log.warning(
                f"No MS data found in {fname} with tag '{ms_tag}'\n"
                "Set argument `-p efg` if the file(s) only contains EFG data "
            )
        except Exception:
            log.warning("Failed to load MS data from .magres")
            raise

    if "efg" in properties:
        efg_tag = property_tags.get("efg", "efg")
        try:
            efg_summary = pd.DataFrame(
                get_efg_summary(atoms, isotopes, euler_convention, efg_tag)
            )
            df = pd.concat([df, efg_summary], axis=1)
        except RuntimeError:
            log.warning(
                f"No EFG data found in {fname} with tag '{efg_tag}'\n"
                "Set argument `-p ms` if the file(s) only contains MS data "
            )
        except Exception:
            log.warning("Failed to load EFG data from .magres")
            raise

    total_explicit_sites = df["multiplicity"].sum()
    log.info(f"\nFound {int(total_explicit_sites)} total sites.")
    log.info(f"Reduced to {len(df)} sites.")

    return df


def _average_group_tensors(atoms: Atoms, keys) -> None:
    """Give every atom of each tagged group the group mean of the given arrays."""
    gids = atoms.get_array("xhn_group")
    for key in keys:
        if not atoms.has(key):
            continue
        arr = atoms.get_array(key).copy()
        for gid in np.unique(gids[gids >= 0]):
            members = np.where(gids == gid)[0]
            arr[members] = arr[members].mean(axis=0)
        atoms.set_array(key, None)
        atoms.set_array(key, arr)


def _merge_with_expanded_groups(
    atoms: Atoms,
    selected: np.ndarray,
    full_group_ids: np.ndarray,
    merging_strategies: dict,
    symmetry_merging_strategies: Optional[dict],
):
    """Merge symmetry copies, keeping every atom of one group per group class.

    Atoms in a tagged group (negative tag) are not merged into one site.  For
    each class of symmetry-equivalent groups, the first group present is kept
    whole, with ``multiplicity`` set to the number of groups in the class.
    All other atoms merge by tag as usual.

    Returns the merged atoms (group sites first, then the rest by tag) and,
    for each site, the source atoms its pair quantities are averaged over.
    """
    tags = atoms.get_tags()
    gids = atoms.get_array("xhn_group")
    in_group = gids >= 0

    kept, multiplicity, site_sets = [], [], []
    for tag in np.unique(tags[in_group]):
        cand = np.where(in_group & (tags == tag))[0]
        rep_gid = gids[cand[0]]
        members = cand[gids[cand] == rep_gid]
        n_groups = len(np.unique(gids[cand]))
        source_set = [int(i) for i in np.where(full_group_ids == rep_gid)[0]]
        for pos in members:
            kept.append(int(pos))
            multiplicity.append(n_groups)
            site_sets.append(source_set)

    group_part = atoms[kept]
    group_part.set_array("multiplicity", None)
    group_part.set_array("multiplicity", np.array(multiplicity, dtype=int))
    del group_part.arrays["xhn_group"]

    rest_pos = np.where(~in_group)[0]
    rest = merge_tagged_sites(
        atoms[rest_pos],
        merging_strategies=merging_strategies,
        symmetry_merging_strategies=symmetry_merging_strategies,
    )
    rest_tags = tags[rest_pos]
    for tag in np.unique(rest_tags):
        site_sets.append([int(selected[rest_pos[np.where(rest_tags == tag)[0][0]]])])

    result = group_part + rest if len(rest) else group_part
    return result, site_sets


def nmr_extract_atoms(
    atoms: Atoms,
    subset: str = "",
    reduce: bool = True,
    average_group: str = "",
    merging_strategies: dict = {
        "positions": lambda x: x[0],
        "labels": lambda x: x[0],
    },
    symprec: float = 1e-4,
    ms_tag: str = "ms",
    efg_tag: str = "efg",
    mean_merge: bool = False,
    logger: Optional[logging.Logger] = None,
    return_index_map: bool = False,
    return_site_sets: bool = False,
    expand_groups: bool = False,
) -> Optional[Atoms]:
    """Extract and reduce NMR sites from a single :class:`ase.Atoms` object.

    Performs the full per-structure pipeline:

    1. Initialise a ``multiplicity`` array.
    2. Optionally reduce to symmetry-unique sites (via
       :class:`soprano.properties.labeling.UniqueSites`).
    3. Warn if symmetry-equivalent sites have inconsistent MS data.
    4. Optionally tag and average functional groups (XHₙ).
    5. Apply an optional atom selection (subset string).
    6. Merge tagged sites.

    Args:
        atoms: Input Atoms object.
        subset: Selection string passed to
            :meth:`soprano.selection.AtomSelection.from_selection_string`.
        reduce: If ``True``, reduce to symmetry-unique sites.
        average_group: Comma-separated XHₙ patterns to average, e.g.
            ``"CH3,CH2"``.
        merging_strategies: Strategies for combining per-atom arrays when
            sites are merged.
        symprec: Symmetry tolerance for SPGLIB operations.
        ms_tag: Array tag for the MS tensors.
        efg_tag: Array tag for the EFG tensors.
        mean_merge: If ``True``, use ``merge_mean`` for NMR tensors (ms, efg) when
            reducing symmetry-equivalent sites. Default is ``False``, which uses
            ``merge_first`` for tensors to avoid corrupting Euler angles for
            non-translation symmetries (e.g. C₂ rotations, mirror planes).
        logger: Logger to use.  Falls back to the module logger when *None*.
        return_index_map: If ``True``, also return a NumPy array that maps
            each input-atom index to the corresponding index in the
            reduced/merged output.  The map is computed *before* any
            ``subset`` filtering.  When ``reduce`` is ``False`` the map
            is the identity.
        return_site_sets: If ``True``, also return, for each site of the
            result, the input-atom indices whose geometry it stands for when
            pair quantities such as dipolar couplings are computed: the
            atoms of one physical XHₙ group for an averaged group, or the
            single representative atom otherwise.  Averaging over symmetry
            copies of an ordinary site would be wrong, as nothing exchanges
            them.  See :func:`soprano.properties.nmr.averaged_dipolar_coupling_list`.
        expand_groups: If ``True``, an averaged group is not collapsed to one
            site.  One physical group stands for each class of
            symmetry-equivalent groups, and each of its atoms stays a site,
            with the group-mean shielding and EFG tensors and its original
            label.  Combined with ``return_site_sets`` and
            :func:`~soprano.properties.nmr.averaged_dipolar_coupling_list`,
            this keeps the coupling among the atoms of a group (the
            intra-methyl H-H term).  ``multiplicity`` is the number of
            equivalent groups.  Incompatible with ``return_index_map``.

    Returns:
        The processed Atoms object, or ``None`` if no atoms remain after
        the selection filter.  When ``return_index_map`` is ``True``,
        returns a tuple ``(atoms, index_map)`` instead.  ``index_map``
        is a 1-D integer array of length ``len(input_atoms)``.

    Wrap the returned map as ``SiteMap(index_map, len(result))`` to chain it
    with further derivations; see :class:`soprano.sitemap.SiteMap`.  Note the
    map does not account for ``subset``, so compose it with a second map built
    from that selection if you use both.
    """
    log = logger or _logger

    if expand_groups and return_index_map:
        raise ValueError("expand_groups cannot be combined with return_index_map")
    if expand_groups and average_group:
        # Averaging replaces tensors, which must not leak into the caller's atoms
        atoms = atoms.copy()

    multiplicity = np.ones(len(atoms), dtype=int)
    atoms.set_array("multiplicity", multiplicity)

    tags = np.arange(len(atoms))

    if reduce:
        log.info("\nTagging equivalent sites")
        tags = UniqueSites.get(atoms, symprec=symprec)
        unique_sites, unique_site_idx = np.unique(tags, return_index=True)
        log.debug(f"    This leaves {len(unique_sites)} unique sites")
        if atoms.has("labels"):
            labels = np.asarray(atoms.get_array("labels"), dtype="U25")
            log.debug(f"    The unique site labels are: {labels[unique_site_idx]}")

    if atoms.has(ms_tag) and not check_equivalent_sites_ms(atoms, tags, tag=ms_tag):
        log.warning(
            "    Some sites with the same symmetry tag/CIF label have different MS isotropy values."
        )
        log.warning(
            "    You can turn off symmetry reduction with the --no-reduce flag."
        )
        log.warning("    You can also turn on debug logging with the -vv flag.")
        log.warning(
            "    If you find that the (symmetry) reduction algorithm is working incorrectly,"
        )
        log.warning("    please report this to the developers.")

    atoms.set_tags(tags)

    expand = bool(expand_groups and average_group)
    if average_group:
        original_labels = (
            np.asarray(atoms.get_array("labels"), dtype="U25").copy()
            if expand and atoms.has("labels")
            else None
        )
        atoms = tag_functional_groups(
            average_group, atoms, vdw_scale=1.0, logger=log
        )
        if expand:
            _average_group_tensors(atoms, (ms_tag, efg_tag, "ms_isotropy", "ms_shielding"))
            if original_labels is not None:
                keep = atoms.get_array("xhn_group") >= 0
                labels = np.asarray(atoms.get_array("labels"), dtype="U25").copy()
                labels[keep] = original_labels[keep]
                atoms.set_array("labels", None)
                atoms.set_array("labels", labels, dtype="U25")

    # Build index map *before* subset filtering so that it refers to the
    # original (input) atom ordering.
    index_map = None
    if return_index_map:
        final_tags = atoms.get_tags()
        unique_tags = np.unique(final_tags)
        index_map = np.searchsorted(unique_tags, final_tags)

    full_tags = atoms.get_tags()
    full_group_ids = (
        atoms.get_array("xhn_group") if atoms.has("xhn_group") else None
    )

    all_selections = AtomSelection.all(atoms)
    if subset:
        log.info(f"\nSelecting atoms based on selection subset string: {subset}")
        try:
            sel = AtomSelection.from_selection_string(atoms, subset)
            all_selections *= sel
        except ValueError as e:
            log.error(f"Could not select atoms based on selection string: {e}")
            extras = ([index_map] if return_index_map else []) + (
                [[]] if return_site_sets else []
            )
            return (None, *extras) if extras else None
        log.debug(f"    Selected atoms: {all_selections.indices}")
        atoms = all_selections.subset(atoms)

    # Build symmetry-specific merging strategies.
    # By default we use merge_first for NMR tensors to avoid corrupting Euler
    # angles under non-translation symmetries (C2, mirrors, etc.).
    # Functional-group averaging (negative tags) keeps the default strategies.
    if mean_merge:
        symmetry_merging_strategies = None
    else:
        symmetry_merging_strategies = {
            **merging_strategies,
            "ms": merge_first,
            "ms_isotropy": merge_first,
            "ms_shielding": merge_first,
            "efg": merge_first,
        }

    selected = np.asarray(all_selections.indices, dtype=int)
    if expand:
        result, site_sets = _merge_with_expanded_groups(
            atoms, selected, full_group_ids, merging_strategies,
            symmetry_merging_strategies,
        )
    else:
        result = merge_tagged_sites(
            atoms,
            merging_strategies=merging_strategies,
            symmetry_merging_strategies=symmetry_merging_strategies,
        )
        sel_tags = full_tags[selected]
        site_sets = []
        for tag in np.unique(sel_tags):
            rep = int(selected[np.where(sel_tags == tag)[0][0]])
            if tag < 0 and full_group_ids is not None:
                site_sets.append(
                    [int(i) for i in np.where(full_group_ids == full_group_ids[rep])[0]]
                )
            else:
                site_sets.append([rep])
    extras = []
    if return_index_map:
        extras.append(index_map)
    if return_site_sets:
        extras.append(site_sets)
    if extras:
        return (result, *extras)
    return result


def nmr_extract_multi(
    files,
    merge: bool = False,
    logger: Optional[logging.Logger] = None,
    sortby: Optional[str] = None,
    sort_order: str = "ascending",
    isotopes: dict = {},
    references: dict = {},
    gradients: Optional[dict] = None,
    properties: List[str] = None,
    euler_convention: str = "zyz",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    query: str = "",
    return_sources: bool = False,
    **kwargs,
):
    """Extract NMR data from one or more magres/XYZ files.

    This is the main entry-point for programmatic access to the NMR
    extraction pipeline.  Each file is read, labelled, reduced, and
    converted to a :class:`pandas.DataFrame`.

    Args:
        files: Iterable of file paths to process.
        merge: If ``True``, concatenate all per-file DataFrames into one.
        logger: Logger to use.  Falls back to the module logger when *None*.
        sortby: Column name to sort each DataFrame by.
        sort_order: ``"ascending"`` or ``"descending"``.
        isotopes: Dict mapping element symbol → isotope mass number.
        references: Dict mapping element symbol → reference shielding.
        gradients: Dict mapping element symbol → shielding gradient. Elements without
            one use the full NMR formula; ``None`` or ``{}`` uses it for all.
        properties: List of properties to extract, e.g. ``["ms", "efg"]``.
            Defaults to both when *None*.
        euler_convention: Euler-angle convention (``"zyz"`` or ``"zxz"``).
        include: Column aliases/names to *keep* after extraction.
        exclude: Column names to *drop* after extraction.
        query: Pandas query string applied to each DataFrame.
        return_sources: If ``True``, also return the structures as they were
            read, before reduction or subsetting.  Callers that need the
            unreduced cell should use this rather than reading the files a
            second time.
        **kwargs: Forwarded to :func:`nmr_extract_atoms`.  Common keys:
            ``subset``, ``reduce``, ``average_group``, ``symprec``,
            ``ms_tag``, ``efg_tag``.

    Returns:
        Tuple ``(dfs, images)`` where *dfs* is a list of DataFrames and
        *images* is a list of the corresponding processed Atoms objects.
        With ``return_sources=True``, returns ``(dfs, images, sources)``.
    """
    if properties is None:
        properties = ["efg", "ms"]

    log = logger or _logger

    if isotopes:
        log.info(f"\nUsing custom isotopes for: {isotopes}")

    # CLI-specific header strings are defined there; keep them out of this module.
    _HEADER = "\n##########################################\n#  Extracting NMR info from magres file  #\n"
    _FOOTER = "\n# End of NMR info extraction            #\n##########################################\n"

    dfs = []
    images = []
    sources = []

    for fname in files:
        log.info(_HEADER)
        log.info(fname)
        log.info(f"\nExtracting properties: {properties}")

        try:
            atoms = io.read(fname)
            atoms = reload_as_molecular_crystal(atoms)
            atoms = label_atoms(atoms, logger=log)
        except OSError:
            log.error(f"Could not read file {fname}, skipping.")
            continue

        # Keep the structure as read: reduction below discards the symmetry
        # copies, and callers such as the 2D plotter need them back.
        source = atoms.copy()

        property_tags = {
            "ms": kwargs.get("ms_tag", "ms"),
            "efg": kwargs.get("efg_tag", "efg"),
        }
        required_tags = [property_tags[p] for p in properties if p in property_tags]
        if not any(atoms.has(t) for t in required_tags):
            log.error(
                f"File {fname} has no {' or '.join(required_tags)} data to extract. Skipping."
            )
            continue

        atoms = nmr_extract_atoms(atoms, logger=log, **kwargs)
        if atoms is None:
            continue

        df = build_nmr_df(
            atoms,
            fname,
            isotopes=isotopes,
            references=references,
            gradients=gradients,
            properties=properties,
            euler_convention=euler_convention,
            property_tags=property_tags,
            logger=log,
        )

        # Resolve "minimal" alias against single-property mode
        _include = list(include) if include is not None else None
        if len(properties) == 1 and _include is not None and "minimal" in _include:
            _include.pop(_include.index("minimal"))
            _include += list(properties)

        # apply filters
        cols_to_include_list = expand_aliases(_include or [], NMR_COLUMN_ALIASES)
        # if MS_shielding in the list _and_ references are provided, add the MS_shift column
        if "MS_shielding" in cols_to_include_list and references:
            log.debug(
                "Adding MS_shift column to dataframe as references are provided."
            )
            cols_to_include_list.append("MS_shift")

        df = apply_df_filtering(
            df,
            cols_to_include_list,
            exclude,
            query,
            essential_columns=NMR_COLUMN_ALIASES["essential"],
            logger=log,
        )

        if len(df) == 0:
            log.warning(
                f"No results found for {fname}.\n "
                "Try removing filters/checking the file contents."
            )
            continue

        atoms = atoms[np.isin(atoms.get_tags(), np.array(df["tags"].values))]

        dfs.append(df)
        images.append(atoms)
        sources.append(source)
        log.info(_FOOTER)

    if merge and dfs:
        dfs = [pd.concat(dfs, axis=0)]

    for i, df in enumerate(dfs):
        dfs[i] = sortdf(df, sortby, sort_order)

    for df in dfs:
        df.rename(columns=units_rename, inplace=True)

    return (dfs, images, sources) if return_sources else (dfs, images)
