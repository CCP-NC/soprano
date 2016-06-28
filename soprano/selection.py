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
import numpy as np

from soprano.utils import minimum_supcell, supcell_gridgen


# This decorator applies to all operators providing some basic checks
def _operator_checks(opfunc):

    def decorated_opfunc(self, other):
        if type(other) is not AtomSelection:
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
                max(sel_indices) >= atoms.get_number_of_atoms()):
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
        h.update(''.join(atoms.get_chemical_symbols()))

        return h.hexdigest()

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

    def subset(self, atoms):
        """Generate an Atoms object containing only the selected atoms."""

        if not self.validate(atoms):
            raise ValueError('Given Atoms object does not match this selection')

        subset = atoms.copy()
        not_sel = list(set(range(atoms.get_number_of_atoms())) -
                       set(self._indices))
        del subset[not_sel]

        # Now the arrays
        for k in self._arrays:
            # The array needs to be sorted with the indices
            # in order to match the order of the atoms in the Atoms object
            _, arr = zip(*sorted(zip(self._indices, self._arrays[k]),
                                 key = lambda t: t[0]))
            subset.new_array(k, np.array(arr))

        return subset

    # Overloading operators to allow sum, subtraction and product of selections
    @_operator_checks
    def __add__(self, other):
        """Sum: join selections"""

        # Join
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices).union(other.indices)))

        return ans

    @_operator_checks
    def __sub__(self, other):

        # Difference
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices)-set(other.indices)))

        return ans

    @_operator_checks
    def __mul__(self, other):

        # Intersection
        ans = copy.deepcopy(self)
        ans._indices = np.array(list(set(self.indices)
                                     .intersection(other.indices)))

        return ans

    def __len__(self):
        return len(self._indices)

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
                pos = (pos[:,None,:]+grid_frac[None,:,:])
            else:
                pos = (pos[:,None,:]+grid[None,:,:])

        sel_i = np.where(np.logical_and(np.all(pos > abc0, axis=-1),
                                        np.all(pos < abc1, axis=-1)))[0]

        return AtomSelection(atoms, sel_i)

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
                pos = (pos[:,None,:]+grid_frac[None,:,:])
            else:
                pos = (pos[:,None,:]+grid[None,:,:])

        sel_i = np.where(np.linalg.norm(pos-center, axis=-1) <= r)[0]

        return AtomSelection(atoms, sel_i)


        





