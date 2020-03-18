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
Contains the NMRTensor class, simplifying the process of diagonalisation of an
NMR tensor as well as its representation in multiple conventions
"""

import numpy as np
from soprano.nmr.utils import (_haeb_sort, _anisotropy, _asymmetry, _span,
                               _skew, _evecs_2_quat, _dip_constant)
from soprano.data.nmr import _get_isotope_data


class NMRTensor(object):
    """NMRTensor

    Class containing an NMR tensor, useful to access all its most important
    properties and representations.
    """

    def __init__(self, data):
        """
        Initialise the NMRTensor

        Create an NMRTensor object from a 3x3 matrix.

        Arguments:
            data (np.ndarray):  3x3 matrix containing the tensor
        """

        self._data = data
        self._symm = (data+data.T)/2.0

        # Diagonalise tensor
        evals, evecs = np.linalg.eigh(self._symm)

        self._haeb_evals = _haeb_sort([evals])
        self._anisotropy = _anisotropy(self._haeb_evals)[0]
        self._redaniso = _anisotropy(self._haeb_evals, True)[0]
        self._asymmetry = _asymmetry(self._haeb_evals)[0]
        self._span = _span(evals[None,:])[0]
        self._skew = _skew(evals[None,:])[0]

        self._trace = np.trace(data)

        # Spherical tensor components
        self._sph0 = np.eye(3)*self.trace/3
        self._sph1 = (data-data.T)/2.0
        self._sph2 = self._symm - self._sph0

        self._evals = evals
        self._evecs = evecs

        self._quat = _evecs_2_quat([evecs])[0]

    @property
    def data(self):
        return self._data

    @property
    def eigenvalues(self):
        return self._evals

    @property
    def haeb_eigenvalues(self):
        return self._haeb_evals[0]

    @property
    def eigenvectors(self):
        return self._evecs

    @property
    def trace(self):
        return self._trace

    @property
    def isotropy(self):
        return self._trace/3.0

    @property
    def anisotropy(self):
        return self._anisotropy

    @property
    def reduced_anisotropy(self):
        return self._redaniso

    @property
    def asymmetry(self):
        return self._asymmetry

    @property
    def span(self):
        return self._span

    @property
    def skew(self):
        return self._skew

    @property
    def quaternion(self):
        return self._quat

    @property
    def spherical_repr(self):
        return [self._sph0, self._sph1, self._sph2]

    @staticmethod
    def make_dipolar(a, i, j, cell=[0, 0, 0], isotopes={}, isotope_i=None,
                     isotope_j=None):
        """Create a dipolar NMRTensor

        Create a dipolar NMR tensor from an atoms object and the indices
        of two atoms. Values are in Hz.

        | Args:
        |   a (ase.Atoms):      Atoms object of the structure to compute the 
        |                       tensor for
        |   i (int):            index of first atom
        |   j (int):            index of second atom
        |   cell (np.array):    vector of the cell of the second atom, for 
        |                       couplings between atoms in different cells.
        |                       By default is [0,0,0].
        |   isotopes (dict): dictionary of specific isotopes to use, by element
        |                    symbol. If the isotope doesn't exist an error will
        |                    be raised.
        |   isotope_i (int): isotope of atom i. To be used if
        |                    different atoms of the same element are supposed
        |                    to be of different isotopes. If None it will fall
        |                    back on the previous definitions. Otherwise it
        |                    overrides everything else.
        |   isotope_j (int): isotope of atom j. See above.

        | Returns:
        |   diptens (NMRTensor):    an NMRTensor object with the dipolar 
        |                           coupling matrix as data.

        """

        pos = a.get_positions()
        elems = np.array(a.get_chemical_symbols())
        r = pos[j]-pos[i] + np.dot(a.get_cell(), cell)
        gammas = _get_isotope_data(elems[[i, j]], 'gamma', isotopes=isotopes,
                                   isotope_list=[isotope_i, isotope_j])

        d = _dip_constant(np.linalg.norm(r)*1e-10, *gammas)
        D = d*(3*r[:, None]*r[None, :]-np.eye(3))/2.0

        return NMRTensor(D)
