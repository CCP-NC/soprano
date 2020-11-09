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

from soprano.utils import minimum_supcell, supcell_gridgen, customize_warnings

customize_warnings()

# This decorator applies to all operators providing some basic checksP


def _operator_checks(opfunc):

    def decorated_opfunc(self, other):
        if not isinstance(other, AtomSelection):
            raise TypeError('AtomSelection does not support operations with'
                            ' different types')

        if self._auth is not None and other._auth is not None:
            # Check compatibility
            if self._auth != other._auth:
                raise ValueError('Selections come from different systems')

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
            if (min(sel_indices) < 0 or
                    max(sel_indices) >= len(atoms)):
                raise ValueError('Invalid indices for given Atoms object')

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
        h.update(''.join(atoms.get_chemical_symbols()).encode())

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
            warnings.warn('WARNING'
                          ' - this selection does not support validation')
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
        |                            molecule)

        | Returns:
        |   subset (ase.Atoms):      Atoms object containing only the
        |                            specified selection
        """

        if not self.validate(atoms):
            raise ValueError(
                'Given Atoms object does not match this selection')

        subset = atoms[self._indices]
        # Copy any extra arrays
        for k, arr in self._arrays.items():
            subset.set_array(k, arr.copy())

        if use_cell_indices and subset.has('cell_indices'):
            ijk = subset.get_array('cell_indices')
            subset.set_scaled_positions(subset.get_scaled_positions() + ijk)

        return subset

    def __getitem__(self, indices):
        """Slicing: take only part of a selection"""

        if type(indices) is int:
            # Special case, a single element!
            indices = slice(indices, indices+1)

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
        common_k = set(self._arrays.keys()
                       ).intersection(set(other._arrays.keys()))
        ans._arrays = {}
        for k in common_k:
            ans._arrays[k] = np.concatenate((self._arrays[k],
                                             other._arrays[k]))

        return ans

    @_operator_checks
    def __sub__(self, other):

        # Difference
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices)-set(other.indices)))
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
        ans._indices = np.array(list(set(self.indices)
                                     .intersection(other.indices)))
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
                    print(('WARNING - conflicting arrays of name {0} found'
                           ' will be removed during intersection'
                           ' operation').format(k))
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
            max_r = np.linalg.norm(np.array(abc1)-abc0)
            scell_shape = minimum_supcell(max_r, latt_cart=atoms.get_cell(),
                                          pbc=atoms.get_pbc())
            grid_frac, grid = supcell_gridgen(atoms.get_cell(), scell_shape)
            if scaled:
                pos = (pos[:, None, :]+grid_frac[None, :, :])
            else:
                pos = (pos[:, None, :]+grid[None, :, :])

        where_i = np.where(np.all(pos > abc0, axis=-1) &
                           np.all(pos < abc1, axis=-1))[:2]

        sel_i = where_i[0]

        sel = AtomSelection(atoms, sel_i)
        if periodic:
            sel.set_array('cell_indices', grid_frac[where_i[1]])

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
            r_bounds = minimum_supcell(r, latt_cart=atoms.get_cell(),
                                       pbc=atoms.get_pbc())
            grid_frac, grid = supcell_gridgen(atoms.get_cell(), r_bounds)
            if scaled:
                pos = (pos[:, None, :]+grid_frac[None, :, :])
            else:
                pos = (pos[:, None, :]+grid[None, :, :])

        where_i = np.where(np.linalg.norm(pos-center, axis=-1) <= r)

        sel_i = where_i[0]

        sel = AtomSelection(atoms, sel_i)
        if periodic:
            sel.set_array('cell_indices', grid_frac[where_i[1]])

        return sel

    @staticmethod
    def from_bonds(atoms, center, n, op='le'):
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
        from soprano.properties.linkage import BondGraph, Bonds
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
    def from_array(atoms, name, value, op='eq'):
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
