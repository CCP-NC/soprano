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
selection.py

Contains the definition of an AtomSelection class,
namely a group of selected atoms for a given structure,
and methods to build it.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import hashlib
import warnings
import operator
import numpy as np
import re
from collections import defaultdict, OrderedDict


from soprano.utils import minimum_supcell, supcell_gridgen, customize_warnings

customize_warnings()

# This decorator applies to all operators providing some basic checksP


def _operator_checks(opfunc):
    def decorated_opfunc(self, other):
        if not isinstance(other, AtomSelection):
            raise TypeError(
                "AtomSelection does not support operations with" " different types"
            )

        if self._auth is not None and other._auth is not None:
            # Check compatibility
            if self._auth != other._auth:
                raise ValueError("Selections come from different systems")

        return opfunc(self, other)

    return decorated_opfunc


class AtomSelection(object):

    """AtomSelection object.

    An AtomSelection represents a group of atoms from an ASE Atoms object.
    It keeps track of them and can be used to perform operations on them
    (for example geometrical transformation or extraction of specific
    properties).
    It does not keep track of the original Atoms object it's been created
    from, but can be "authenticated" to verify that it is indeed operating
    consistently on the same structure. It also provides a series of static
    methods to build selections with various criteria.

    """

    def __init__(self, atoms, sel_indices, authenticate=True):
        """Initialize the AtomSelection.

        | Args:
        |   atoms (ase.Atoms): the atoms object on which the selection is
        |                      applied
        |   sel_indices (list[int]): the list of indices of the atoms that
        |                            are to be selected
        |   authenticate (Optional[bool]): whether to use hashing to confirm
        |                                  the identity of the atoms object
        |                                  we're operating with

        """

        # A quick check: are the indices actually contained in the Atoms?
        if len(sel_indices) > 0:
            if min(sel_indices) < 0 or max(sel_indices) >= len(atoms):
                raise ValueError("Invalid indices for given Atoms object")

        self._indices = np.array(sel_indices)

        if authenticate:
            # Create an hash for certification
            self._auth = self._hash(atoms)
        else:
            self._auth = None

        self._arrays = {}

    @property
    def indices(self):
        return self._indices

    def _hash(self, atoms):
        """A function to create an identifying hash for a given Atoms system.
        This is used later to check that the system is indeed unchanged when
        the Selection is reused.

        While changes in positions or cell don't invalidate the Selection,
        changes in composition potentially do (indices of atoms can change).
        """

        h = hashlib.md5()
        h.update("".join(atoms.get_chemical_symbols()).encode())

        return h.hexdigest()

    def has(self, name):
        """Check if the selection has a given array

        | Args:
        |   name (str): name of the array to be checked for

        | Returns:
        |   has (bool): if the array is present or not
        """

        return name in self._arrays

    def set_array(self, name, array):
        """Save an array of given name containing arbitraty information
        tied to the selected atoms.
        This must match the length of the selection and will be passed on to
        any Atoms objects created with .subset.

        | Args:
        |   name (str): name of the array to be set or created
        |   array (np.ndarray): array of data to be saved

        """

        # First a check
        if len(array) != len(self):
            raise ValueError("Invalid array passed to set_array")

        self._arrays[name] = np.array(array)

    def get_array(self, name):
        """Retrieve a previously stored data array.

        | Args:
        |   name (str): name of the array to be set or created

        | Returns:
        |   array (np.ndarray): array of data to be saved

        """

        # If the name isn't right just let the KeyError happen
        return self._arrays[name]

    def validate(self, atoms):
        """Check that the given Atoms object validates with this selection."""
        if self._auth is None:
            warnings.warn("WARNING" " - this selection does not support validation")
            return True
        else:
            return self._hash(atoms) == self._auth

    def subset(self, atoms, use_cell_indices=False):
        """Generate an Atoms object containing only the selected atoms.

        | Args:
        |   atoms (ase.Atoms):       Atoms object from which to take the
        |                            selection
        |   use_cell_indices (bool): If True, use the cell_indices array to
        |                            pick the specified periodic copies of
        |                            the corresponding atoms (useful e.g. to
        |                            take the correct periodic copies for a
        |                            molecule). Default is False

        | Returns:
        |   subset (ase.Atoms):      Atoms object containing only the
        |                            specified selection
        """

        if not self.validate(atoms):
            raise ValueError("Given Atoms object does not match this selection")

        subset = atoms[self._indices]
        # Copy any extra arrays
        for k, arr in self._arrays.items():
            subset.set_array(k, arr.copy())

        if use_cell_indices and subset.has("cell_indices"):
            ijk = subset.get_array("cell_indices")
            subset.set_scaled_positions(subset.get_scaled_positions() + ijk)

        return subset

    def __getitem__(self, indices):
        """Slicing: take only part of a selection"""

        if type(indices) is int:
            # Special case, a single element!
            indices = slice(indices, indices + 1)

        try:
            newsel = self._indices[indices]
        except TypeError:
            newsel = [self._indices[i] for i in indices]

        sliced = copy.deepcopy(self)
        sliced._indices = newsel
        sliced._arrays = {k: a[indices] for k, a in self._arrays.items()}

        return sliced

    def __iter__(self):
        for i in range(len(self._indices)):
            yield self[i]

    # Overloading operators to allow sum, subtraction and product of selections
    @_operator_checks
    def __add__(self, other):
        """Sum: join selections"""

        # Join
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices).union(other.indices)))
        # For the arrays:
        # only join the ones present in BOTH selections
        common_k = set(self._arrays.keys()).intersection(set(other._arrays.keys()))
        ans._arrays = {}
        for k in common_k:
            ans._arrays[k] = np.concatenate((self._arrays[k], other._arrays[k]))

        return ans

    @_operator_checks
    def __sub__(self, other):

        # Difference
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices) - set(other.indices)))
        # For the arrays:
        # keep them but remove the removed indices
        arr_i = [np.where(self.indices == i)[0][0] for i in ans._indices]
        for k in ans._arrays:
            ans._arrays[k] = ans._arrays[k][arr_i]

        return ans

    @_operator_checks
    def __mul__(self, other):

        # Intersection
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices).intersection(other.indices)))
        # For the arrays:
        # keep the ones present in either selection,
        # but only the relevant indices of course,
        # and remove if conflicting!
        all_k = set(self._arrays.keys()).union(set(other._arrays.keys()))
        arr1_i = [np.where(self.indices == i)[0][0] for i in ans._indices]
        arr2_i = [np.where(other.indices == i)[0][0] for i in ans._indices]
        ans._arrays = {}
        for k in all_k:
            try:
                arr1 = self._arrays[k][arr1_i]
            except KeyError:
                arr1 = None
            try:
                arr2 = other._arrays[k][arr2_i]
            except KeyError:
                arr2 = None

            if arr1 is not None and arr2 is not None:
                # Do they conflict?
                if not np.all(arr1 == arr2):
                    print(
                        (
                            "WARNING - conflicting arrays of name {0} found"
                            " will be removed during intersection"
                            " operation"
                        ).format(k)
                    )
                    continue

            ans._arrays[k] = arr1 if arr1 is not None else arr2

        return ans

    def __len__(self):
        return len(self._indices)

    def __contains__(self, item):
        return item in self._indices

    @staticmethod
    def all(atoms):
        """Generate a selection for the given Atoms object of all atoms.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection

        | Returns:
        |   selection (AtomSelection)

        """

        return AtomSelection(atoms, range(len(atoms)))

    @staticmethod
    def from_element(atoms, element):
        """Generate a selection for the given Atoms object of all atoms of a
        specific element.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   element (str): symbol of the element to select

        | Returns:
        |   selection (AtomSelection)

        """

        sel_i = np.where(np.array(atoms.get_chemical_symbols()) == element)[0]

        return AtomSelection(atoms, sel_i)
    @staticmethod
    def from_selection_string(atoms, selection_string):
        """Generate a selection for the given Atoms object based on a standardised 
        string format.
        (Useful to parse command-line-arguments.)

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   selection_string (str): string specifying a subset of atoms. See example below.

        | Returns:
        |   selection (AtomSelection)


        Examples of selection_string: 
        'Si' - select all Si atoms
        'Si.1' - select the first Si atom
        'Si.1-3' - select the first three Si atoms
        'Si.1-3,5' - select the first three and fifth Si atoms
        'C.1,H.2' - select the first carbon and second hydrogen atoms
        'C1' - select the atom with 'C1' label, regardles of where it appears. 
        'C1,C3a' - select the atoms with 'C1' and 'C3a' labels, regardles of where they appear. 
        """
        def has_numbers(selection_string):
            return bool(re.search(r'\d', selection_string))
        def has_hyphen(selection_string):
            return bool(re.search(r'-', selection_string))
        def has_period(selection_string):
            return bool(re.search(r'\.', selection_string))

        selection = defaultdict(lambda: [])
        for split in selection_string.split(","):
            # split into element and sites
            sites = re.split('([a-zA-Z]+)', split)[1:]
            el = sites.pop(0)
            # make sure no numbers left in el
            if has_numbers(el):
                raise ValueError("Problem parsing selection string: " + selection_string
                                 + " - wasn't expect more numbers in " + el)
            # get the indices of each element in the atoms object
            element_indices = AtomSelection.from_element(atoms, el).indices
            # make sure the chosen element is present!
            if not el in atoms.symbols:
                raise ValueError(
                    "Element {0} not present in the atoms object".format(el)
                )
            # make sure the spitting worked as expected
            sites = sites[0]
            
            # if empty string -> select all element indices
            if sites == '':
                selection[el] = element_indices
                continue

            # if starts with period -> regular index
            if sites[0] == '.':
                # split on '.'
                sites = sites[1:].split(".")
                el_indices = []
                for site in sites:
                    # if it has a hyphen:
                    if has_hyphen(site):
                        site = site.split("-")
                        el_indices+=range(int(site[0]), int(site[1]) + 1)
                    else:
                        el_indices.append(int(site))
                # if zero in el_indices -> throw error
                if 0 in el_indices:
                    raise ValueError(
                        "WARNING - zero in selection string, please use 1-indexing"
                        " during selection operation. Always double-check if selection matches your expectations!"
                    )
                # switch to python indexing:
                el_indices = np.array(el_indices) - 1
                selection[el].extend(element_indices[el_indices])
        
            else:
                # must be a cif-style label!
                # sites is of the form 'C1'
                if has_hyphen(sites):
                    raise ValueError(
                        "Error - hyphen in selection string while using cif labels."
                        "Use explicit comma-separated string intead. e.g. '-s C1,C2'"
                    )
                if has_period(sites):
                    raise ValueError(
                        "Error - period in selection string while using cif labels"
                        "Use explicit comma-separated string intead. e.g. '-s C1,C2'"
                    )
                # use 'split' as the label to look for:
                indices = np.where(atoms.get_array('labels') == split)[0]
                if len(indices) == 0:
                    # raise error if no atoms with this label found
                    raise ValueError(
                        "No atoms with label {0} found in the atoms object".format(split)
                    )
                    
                selection[el].extend(indices)
        
        # flatten and remove any duplicate indices
        # (this way preserves the order. From python 3.7 onwards, we can use standard dictionaries)
        sel_i = list(OrderedDict.fromkeys([idx for el in selection for idx in selection[el]]))
        # Return the selection
        return AtomSelection(atoms, sel_i)

    @staticmethod
    def from_box(atoms, abc0, abc1, periodic=False, scaled=False):
        """Generate a selection for the given Atoms object of all atoms within
        a given box volume.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   abc0 ([float, float, float]): bottom corner of box
        |   abc1 ([float, float, float]): top corner of box
        |   periodic (Optional[bool]): if True, include periodic copies of the
        |                              atoms
        |   scaled (Optional[bool]): if True, consider scaled (fractional)
        |                            coordinates instead of absolute ones

        | Returns:
        |   selection (AtomSelection)

        """

        if scaled:
            pos = atoms.get_scaled_positions()
        else:
            pos = atoms.get_positions()
        # Do we need periodic copies?
        if periodic and any(atoms.get_pbc()):
            # Get the range
            max_r = np.linalg.norm(np.array(abc1) - abc0)
            scell_shape = minimum_supcell(
                max_r, latt_cart=atoms.get_cell(), pbc=atoms.get_pbc()
            )
            grid_frac, grid = supcell_gridgen(atoms.get_cell(), scell_shape)
            if scaled:
                pos = pos[:, None, :] + grid_frac[None, :, :]
            else:
                pos = pos[:, None, :] + grid[None, :, :]

        where_i = np.where(np.all(pos > abc0, axis=-1) & np.all(pos < abc1, axis=-1))[
            :2
        ]

        sel_i = where_i[0]

        sel = AtomSelection(atoms, sel_i)
        if periodic:
            sel.set_array("cell_indices", grid_frac[where_i[1]])

        return sel

    @staticmethod
    def from_sphere(atoms, center, r, periodic=False, scaled=False):
        """Generate a selection for the given Atoms object of all atoms within
        a given spherical volume.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   center ([float, float, float]): center of the sphere
        |   r (float): radius of the sphere
        |   periodic (Optional[bool]): if True, include periodic copies of the
        |                              atoms
        |   scaled (Optional[bool]): if True, consider scaled (fractional)
        |                            coordinates instead of absolute ones

        | Returns:
        |   selection (AtomSelection)

        """

        if scaled:
            pos = atoms.get_scaled_positions()
        else:
            pos = atoms.get_positions()
        # Do we need periodic copies?
        if periodic and any(atoms.get_pbc()):
            # Get the range
            r_bounds = minimum_supcell(
                r, latt_cart=atoms.get_cell(), pbc=atoms.get_pbc()
            )
            grid_frac, grid = supcell_gridgen(atoms.get_cell(), r_bounds)
            if scaled:
                pos = pos[:, None, :] + grid_frac[None, :, :]
            else:
                pos = pos[:, None, :] + grid[None, :, :]

        where_i = np.where(np.linalg.norm(pos - center, axis=-1) <= r)

        sel_i = where_i[0]

        sel = AtomSelection(atoms, sel_i)
        if periodic:
            sel.set_array("cell_indices", grid_frac[where_i[1]])

        return sel

    @staticmethod
    def from_bonds(atoms, center, n, op="le"):
        """Generate a selection for the given Atoms object of other atoms
        based on their reciprocal bonding distance. Default is selection of
        all atoms that are within a certain bonding distance (less-or-equal
        than n). However different operators can be specified for different
        selection criteria. Atoms that do not belong to the same tree of the
        bonding graph are never selected.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   center (int): index of the atom to compute the bonding distance
        |                 from
        |   n (int): bonding distance to compare
        |   op (Optional[str]): operator to use for comparison with the given
        |                       bonding distance. By default it's le, meaning
        |                       "less or equal" than n, which means all atoms
        |                       will be selected that are at most n bonds away
        |                       from the center.
        |                       Other options are the functions present in the
        |                       `operator` module and are:
        |                             - lt : less than
        |                             - le : less or equal
        |                             - eq : exactly equal
        |                             - ge : greater or equal
        |                             - gt : greater than
        """

        # Start by computing the bonding graph
        from soprano.properties.linkage import BondGraph
        from soprano.utils import get_bonding_distance

        bgraph = BondGraph.get(atoms)
        op = getattr(operator, op)
        sel_i = []

        for i in np.arange(len(atoms)):
            d = get_bonding_distance(bgraph, center, i)
            if d > -1 and op(d, n):
                sel_i.append(i)

        return AtomSelection(atoms, sel_i)

    @staticmethod
    def from_array(atoms, name, value, op="eq"):
        """Generate a selection for the given Atoms object of other atoms
        based on a comparison with some array value. Default is selection of
        all atoms that have the same exact value. However different operators
        can be specified for different selection criteria.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   name (str): name of the array to select with
        |   value (any type): value to compare the contents of the array with
        |   op (Optional[str]): operator to use for comparison with the given
        |                       value. By default it's eq, meaning
        |                       "equal" to value, which means all atoms
        |                       will be selected for whose the array of given
        |                       name has the given value.
        |                       Other options are the functions present in the
        |                       `operator` module and are:
        |                             - lt : less than
        |                             - le : less or equal
        |                             - eq : exactly equal
        |                             - ge : greater or equal
        |                             - gt : greater than
        """

        arr = atoms.get_array(name)
        op = getattr(operator, op)
        sel_i = np.where(op(arr, value))[0]

        return AtomSelection(atoms, sel_i)


    @staticmethod
    def unique(atoms, symprec=1e-4):
        """Generate a selection for the given Atoms object containing
        only the symmetry-unique atoms.

        We use the spacegroup as found by spglib to determine the symmetry
        operations and then use these to tag the equivalent atoms.

        | Args:
        |   atoms (ase.Atoms): Atoms object on which to perform selection
        |   sympres (float): tolerance for symmetry equivalence
        | Returns:
        |   selection (AtomSelection)

        """
        from soprano.properties.labeling import UniqueSites
        from soprano.utils import has_cif_labels

        sitetags = UniqueSites.get(atoms)
        max_tag = max(sitetags)
        sel_i = [np.argmax(np.array(sitetags)==i) for i in range(max_tag+1)]
        # Now we make sure that, for structures with cif labels
        # the symmetry-unique sites that remain are those we 
        # would expect based on the existing CIF labels.
        if has_cif_labels(atoms):
            ciflabels = atoms.get_array('labels')
            # get indices of unique cif labels using OrderedDict
            unique_cif_labels = list(OrderedDict.fromkeys(ciflabels))
            # take first match of each unique cif label
            sel_i_cif = [np.argmax(np.array(ciflabels)==i) for i in unique_cif_labels]
            # test that they all match otherwise raise warning
            if len(sel_i_cif) != len(unique_cif_labels):
                all_matched = False
            else:
                all_matched = all(np.array(sel_i) == np.array(sel_i_cif))
            if not all_matched:
                warnings.warn("The symmetry-reduced sites don't match the CIF labels!"
                "Manually check that the symmetry reduction is working as expected."
                "Proceeding with the CIF label reduction rather than the symmetry reduction.")
                sel_i = sel_i_cif

        sel = AtomSelection(atoms, sel_i)
        
        return sel