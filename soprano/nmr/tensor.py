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
from soprano.nmr.utils import (_evals_sort, _haeb_sort, _anisotropy,
                               _asymmetry, _span, _skew, _evecs_2_quat,
                               _dip_constant)
from soprano.data.nmr import _get_isotope_data
from ase.quaternions import Quaternion


class NMRTensor(object):
    """NMRTensor

    Class containing an NMR tensor, useful to access all its most important
    properties and representations.
    """

    ORDER_INCREASING = 'i'
    ORDER_DECREASING = 'd'
    ORDER_HAEBERLEN = 'h'
    ORDER_NQR = 'n'

    def __init__(self, data, order=ORDER_INCREASING):
        """
        Initialise the NMRTensor

        Create an NMRTensor object from a 3x3 matrix.

        Arguments:
            data (np.ndarray or tuple):  3x3 matrix containing the tensor, or
                                         pair [evals, evecs] for the symmetric
                                         part alone.
            order (str):        Order to use for eigenvalues/eigenvectors. Can
                                be 'i' (ORDER_INCREASING), 'd' 
                                (ORDER_DECREASING), 'h' (ORDER_HAEBERLEN) or
                                'n' (ORDER_NQR). Default is 'i'.
        """

        self._order = order

        if len(data) == 3:
            self._data = np.array(data)
            if self._data.shape != (3, 3):
                raise ValueError('Invalid matrix data passed to NMRTensor')
            self._symm = (self._data+self._data.T)/2.0
            evals, evecs = np.linalg.eigh(self._symm)
        elif len(data) == 2:
            evals, evecs = data
            evecs = np.array(evecs)
            sort_i = np.argsort(evals)
            if len(evals) != 3 or evecs.shape != (3, 3):
                raise ValueError('Invalid eigenvalues/vectors passed to '
                                 'NMRTensor')
            evals = evals[sort_i]
            evecs = evecs[:, sort_i]
            self._symm = np.linalg.multi_dot([evecs, np.diag(evals), evecs.T])
            self._data = self._symm

        self._incr_evals = evals
        self._haeb_evals = _haeb_sort([evals])[0]
        self._anisotropy = None
        self._redaniso = None
        self._asymmetry = None
        self._span = None
        self._skew = None

        self._trace = None

        # Spherical tensor components
        self._sph = None

        # Sort eigenvalues and eigenvectors as specified
        if order != self.ORDER_INCREASING:
            self._evals, sort_i = _evals_sort([evals], order, True)
            self._evals = self._evals[0]
            self._evecs = evecs[:, sort_i[0]]
        else:
            # No point in fixing what ain't broken
            self._evals = evals
            self._evecs = evecs

        # Last eigenvector must be the cross product of the first two
        # (apparently this is much faster than np.cross. Beats me why)
        self._evecs[0, 2] = (self._evecs[1, 0]*self._evecs[2, 1] -
                             self._evecs[2, 0]*self._evecs[1, 1])
        self._evecs[1, 2] = (self._evecs[2, 0]*self._evecs[0, 1] -
                             self._evecs[0, 0]*self._evecs[2, 1])
        self._evecs[2, 2] = (self._evecs[0, 0]*self._evecs[1, 1] -
                             self._evecs[1, 0]*self._evecs[0, 1])

        self._quat = None

    @property
    def data(self):
        return self._data.copy()

    @property
    def eigenvalues(self):
        return self._evals.copy()

    @property
    def eigenvectors(self):
        return self._evecs.copy()

    @property
    def trace(self):
        if self._trace is None:
            self._trace = np.trace(self._data)
        return self._trace

    @property
    def isotropy(self):
        return self.trace/3.0

    @property
    def anisotropy(self):
        if self._anisotropy is None:
            self._anisotropy = _anisotropy(self._haeb_evals[None, :])[0]
        return self._anisotropy

    @property
    def reduced_anisotropy(self):
        if self._redaniso is None:
            self._redaniso = _anisotropy(self._haeb_evals[None, :], True)[0]
        return self._redaniso

    @property
    def asymmetry(self):
        if self._asymmetry is None:
            self._asymmetry = _asymmetry(self._haeb_evals[None, :])[0]
        return self._asymmetry

    @property
    def span(self):
        if self._span is None:
            self._span = _span(self._incr_evals[None, :])[0]
        return self._span

    @property
    def skew(self):
        if self._skew is None:
            self._skew = _skew(self._incr_evals[None, :])[0]
        return self._skew

    @property
    def quaternion(self):
        if self._quat is None:
            self._quat = _evecs_2_quat([self._evecs])[0]
        return Quaternion(self._quat.q)

    @property
    def spherical_repr(self):
        if self._sph is None:
            self._sph = np.zeros((3, 3, 3))
            self._sph[0] = np.eye(3)*self.trace/3
            self._sph[1] = (self._data-self._data.T)/2.0
            self._sph[2] = self._symm - self._sph0

        return self._sph.copy()

    def euler_angles(self, convention='zyz'):
        """Return Euler angles of the Principal Axis System

        Return Euler angles of the PAS for this tensor in the
        required convention (currently supported: zyz, zxz).

        Keyword Arguments:
            convention {str} -- Euler angles convention to use
            (default: {'zyz'})
        """

        convention = convention.lower()
        return self.quaternion.euler_angles(convention)

    @staticmethod
    def make_dipolar(a, i, j, cell=[0, 0, 0], isotopes={}, isotope_i=None,
                     isotope_j=None, rotation_axis=None):
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
        |   rotation_axis (np.array):   an axis around which the selected pair
        |                               is rotating. If present, the tensor
        |                               will be averaged for infinitely fast
        |                               rotation around it.

        | Returns:
        |   diptens (NMRTensor):    an NMRTensor object with the dipolar 
        |                           coupling matrix as data.

        """

        pos = a.get_positions()
        elems = np.array(a.get_chemical_symbols())
        r = pos[j]-pos[i] + np.dot(a.get_cell(), cell)
        rnorm = np.linalg.norm(r)
        gammas = _get_isotope_data(elems[[i, j]], 'gamma', isotopes=isotopes,
                                   isotope_list=[isotope_i, isotope_j])

        d = _dip_constant(rnorm*1e-10, *gammas)

        evals = np.zeros(3)
        evecs = np.zeros((3, 3))

        if rotation_axis is None:
            axis = r/rnorm
        else:
            axis = np.array(rotation_axis)/np.linalg.norm(rotation_axis)
            vp2 = np.dot(r/rnorm, axis)**2
            d = 0.5*d*(3*vp2-1)

        evals[2] = d
        evals[:2] = -d/2.0

        evecs[:, 2] = axis

        x, y = axis[:2]
        evecs[1, 0] = x
        evecs[0, 0] = -y
        evecs[:, 0] /= (x**2+y**2)**0.5

        evecs[0, 1]
        evecs[0, 1] = -evecs[2, 2]*evecs[1, 0]
        evecs[1, 1] = evecs[2, 2]*evecs[0, 0]
        evecs[2, 1] = evecs[0, 2]*evecs[1, 0]-evecs[1, 2]*evecs[0, 0]

        if d < 0:
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

        D = [evals, evecs]

        return NMRTensor(D)
