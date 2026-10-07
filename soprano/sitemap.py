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

"""
sitemap.py

Contains the definition of a SiteMap, which records where the entries of a
derived structure came from in the structure it was derived from.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ase import Atoms

    from soprano.selection import AtomSelection

__all__ = ["SiteMap"]

#: Value in :attr:`SiteMap.to_sites` for an atom the derivation discarded.
DROPPED = -1


@dataclass(frozen=True, eq=False)
class SiteMap:
    """Correspondence between the atoms of a structure and the sites of a
    structure derived from it.

    Soprano routinely derives one ``Atoms`` object from another, either by
    selecting a subset or by merging equivalent positions together. The derived
    object has its own index space, and code downstream has to translate
    between the two. A SiteMap is that translation, and nothing else.

    The vocabulary follows ``CONTEXT.md``: the structure you started from has
    **atoms**, the derived structure has **sites**, and a site stands for one
    or more atoms. Selecting gives sites that each represent a single atom;
    merging gives sites that represent several.

    A SiteMap holds no reference to either structure, so it is cheap, picklable
    and cannot go stale. The cost is that pairing a map with the wrong
    structure is possible, which is what :meth:`validate` is for.

    Attributes:
        to_sites: Integer array of length ``n_atoms``. Entry ``i`` is the site
            that atom ``i`` maps to, or :data:`DROPPED` if the derivation
            discarded it. The array is read-only.
        n_sites: Number of sites in the derived structure.

    Examples:
        Relate a selection to the structure it came from::

            sel = AtomSelection.from_element(atoms, 'H')
            smap = SiteMap.from_selection(sel, atoms)
            smap.members(0)      # the atom behind site 0

        Chain two derivations into one map back to the original atoms::

            reduced_map = SiteMap.from_tags(atoms.get_tags())
            filtered_map = SiteMap.from_selection(sel, reduced)
            combined = reduced_map.compose(filtered_map)
    """

    to_sites: np.ndarray
    n_sites: int

    def __post_init__(self):
        arr = np.ascontiguousarray(self.to_sites, dtype=int)
        if arr.ndim != 1:
            raise ValueError(f"to_sites must be 1-D, got shape {arr.shape}")
        if self.n_sites < 0:
            raise ValueError(f"n_sites must be non-negative, got {self.n_sites}")
        if arr.size and (arr.min() < DROPPED or arr.max() >= self.n_sites):
            raise ValueError(
                f"to_sites entries must be {DROPPED} or in [0, {self.n_sites}), "
                f"got range [{arr.min()}, {arr.max()}]"
            )
        arr.setflags(write=False)
        object.__setattr__(self, "to_sites", arr)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, n_atoms: int) -> "SiteMap":
        """Map a structure onto itself, one site per atom.

        Args:
            n_atoms: Number of atoms.
        """
        return cls(np.arange(n_atoms, dtype=int), n_atoms)

    @classmethod
    def from_selection(
        cls, selection: "AtomSelection", source: Union["Atoms", int]
    ) -> "SiteMap":
        """Build the map produced by ``selection.subset(source)``.

        A selection already is the map, read the other way round: site ``k`` of
        the subset comes from atom ``selection.indices[k]``. Selection order is
        not guaranteed to be sorted, so reconstructing it by hand is a mistake.

        Args:
            selection: The selection that will be, or has been, applied.
            source: The structure the selection indexes, or just its length.
        """
        n_atoms = source if isinstance(source, (int, np.integer)) else len(source)
        indices = np.asarray(selection.indices, dtype=int)
        if indices.size and (indices.min() < 0 or indices.max() >= n_atoms):
            raise ValueError(
                f"selection indexes atoms outside [0, {n_atoms})"
            )
        to_sites = np.full(int(n_atoms), DROPPED, dtype=int)
        to_sites[indices] = np.arange(len(indices))
        return cls(to_sites, len(indices))

    @classmethod
    def from_tags(cls, tags) -> "SiteMap":
        """Build the map produced by merging atoms that share a tag.

        Sites come out ordered by sorted unique tag, which is the order
        :func:`soprano.nmr.extract.merge_tagged_sites` produces.

        Args:
            tags: Per-atom tag array. Atoms sharing a tag become one site.
        """
        tags = np.asarray(tags)
        unique = np.unique(tags)
        return cls(np.searchsorted(unique, tags).astype(int), len(unique))

    # ------------------------------------------------------------------
    # Use
    # ------------------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the source structure."""
        return len(self.to_sites)

    def __len__(self) -> int:
        return self.n_atoms

    def compose(self, other: "SiteMap") -> "SiteMap":
        """Chain this map with one that starts where this one ends.

        If ``self`` maps atoms to intermediate sites and ``other`` maps those
        intermediate sites to final sites, the result maps atoms straight to
        final sites. An atom dropped by either map is dropped in the result.

        Args:
            other: Map whose atom space is this map's site space.

        Raises:
            ValueError: If the two index spaces do not meet.
        """
        if self.n_sites != other.n_atoms:
            raise ValueError(
                f"cannot compose: this map produces {self.n_sites} sites but the "
                f"next expects {other.n_atoms} atoms"
            )
        to_sites = np.full(self.n_atoms, DROPPED, dtype=int)
        kept = self.to_sites != DROPPED
        to_sites[kept] = other.to_sites[self.to_sites[kept]]
        return SiteMap(to_sites, other.n_sites)

    def members(self, site: int) -> np.ndarray:
        """Every source atom that site ``site`` stands for.

        Args:
            site: Index into the derived structure.

        Raises:
            IndexError: If ``site`` is not a site of this map.
        """
        if not 0 <= site < self.n_sites:
            raise IndexError(
                f"site {site} out of range for a map with {self.n_sites} sites"
            )
        return np.flatnonzero(self.to_sites == site)

    def validate(self, atoms: "Atoms") -> None:
        """Check that ``atoms`` is the structure this map starts from.

        Only the length is checked, which catches the realistic mistake of
        holding a map from the wrong stage of a pipeline.

        Args:
            atoms: Candidate source structure.

        Raises:
            ValueError: If the length does not match.
        """
        if len(atoms) != self.n_atoms:
            raise ValueError(
                f"this map is for a structure of {self.n_atoms} atoms, "
                f"got {len(atoms)}"
            )

    def __repr__(self) -> str:
        dropped = int(np.count_nonzero(self.to_sites == DROPPED))
        return (
            f"SiteMap({self.n_atoms} atoms -> {self.n_sites} sites"
            + (f", {dropped} dropped)" if dropped else ")")
        )
