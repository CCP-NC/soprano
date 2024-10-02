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

import warnings
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from ase.quaternions import Quaternion
from scipy.spatial.transform import Rotation

from soprano.data.nmr import EFG_TO_CHI, _get_isotope_data, nmr_gamma, nmr_quadrupole, nmr_spin
from soprano.nmr.utils import (
    _anisotropy,
    _asymmetry,
    _dip_constant,
    _equivalent_euler,
    _equivalent_relative_euler,
    _evals_sort,
    _evecs_2_quat,
    _haeb_sort,
    _handle_euler_edge_cases,
    _matrix_to_euler,
    _skew,
    _span,
    _split_species,
    _test_euler_rotation,
    _tryallanglestest,
)

DEGENERACY_TOLERANCE = 1e-6

class NMRTensor:
    """NMRTensor

    Class containing an NMR tensor, useful to access all its most important
    properties and representations.
    """

    ORDER_INCREASING = "i"
    ORDER_DECREASING = "d"
    ORDER_HAEBERLEN = "h"
    ORDER_NQR = "n"

    def __init__(self,
                 data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                 order: str=ORDER_INCREASING):
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
        self._order = None
        self._data = None
        self._symm = None
        self._evals = None
        self._evecs = None
        self._process_data(data)
        # The following will also sort the eigenvalues and eigenvectors
        self.order = order

        # Initialize other attributes
        self._anisotropy = None
        self._redaniso = None
        self._asymmetry = None
        self._span = None
        self._skew = None
        self._trace = None
        self._degeneracy = None
        # Spherical tensor components
        self._sph = None
        self._quat = None

        self._incr_evals = self._evals
        self._haeb_evals = _haeb_sort([self._evals])[0]

    def _process_data(self, data):
        if len(data) == 3:
            self._data = np.array(data)
            if self._data.shape != (3, 3):
                raise ValueError("Invalid matrix data passed to NMRTensor")
            self._symm = (self._data + self._data.T) / 2.0
            evals, evecs = np.linalg.eigh(self._symm)
        elif len(data) == 2:
            evals, evecs = data
            evecs = np.array(evecs)
            if len(evals) != 3 or evecs.shape != (3, 3):
                raise ValueError("Invalid eigenvalues/vectors passed to NMRTensor")
            self._symm = np.linalg.multi_dot([evecs, np.diag(evals), evecs.T])
            self._data = self._symm
        else:
            raise ValueError("Data must be a 3x3 matrix or a pair of [evals, evecs]")
        self._evals = evals
        self._evecs = evecs

    def _order_tensor(self, order):
        # Sort eigenvalues and eigenvectors as specified
        if self._order is None or self._order != order:
            self._evals, sort_i = _evals_sort([self._evals], order, True)
            self._evals = self._evals[0]
            self._evecs = self._evecs[:, sort_i[0]]
        # Last eigenvector must be the cross product of the first two
        self._evecs[:, 2] = np.cross(self._evecs[:, 0], self._evecs[:, 1])

        # For any property that depends on the eigenvalue order, reset it
        self._anisotropy = None
        self._redaniso = None
        self._asymmetry = None
        self._quat = None

    @property
    def order(self):
        return self._order
    # method to update the order of the tensor
    @order.setter
    def order(self, value):
        self._order_tensor(value)
        self._order = value

    @property
    def data(self):
        return self._data.copy()

    @property
    def eigenvalues(self):
        return self._evals.copy()

    @property
    def PAS(self):
        '''
        Returns the principal axis system (PAS) of the tensor.
        '''
        return np.diag(self._evals)

    @property
    def degeneracy(self):
        '''
        Returns the degeneracy of the tensor.
        For example, a tensor with eigenvalues [1, 1, 1] has a degeneracy of 3.
        A tensor with eigenvalues [1, 1, 2] has a degeneracy of 2.
        A tensor with eigenvalues [1, 2, 3] has a degeneracy of 1.
        '''
        if self._degeneracy is None:
            self._degeneracy = np.sum(np.abs(self._evals - self._evals[0]) < DEGENERACY_TOLERANCE)
        return self._degeneracy

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
        return self.trace / 3.0

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
        """
        Spherical representation of the tensor

        Returns a 3x3x3 array containing the isotropic, antisymmetric and
        symmetric parts of the tensor. The isotropic part is the average of
        the trace of the tensor, the antisymmetric part is the difference
        between the tensor and its transpose divided by 2, and the symmetric
        part is the sum of the tensor and its transpose divided by 2 minus
        the isotropic part. This construction is such that the sum of the
        components is equal to the original tensor.

        .. math::
            \\sigma = \\sigma_{iso} + \\sigma_{A} + \\sigma_{S}

        where

        .. math::
            \\sigma_{iso} = \\frac{1}{3} \\text{Tr}(\\sigma) \\mathbf{I}
                
            \\sigma_{A} = \\frac{1}{2} (\\sigma - \\sigma^T)

            \\sigma_{S} = \\frac{1}{2} (\\sigma + \\sigma^T) - \\sigma_{iso}

        Returns:
            np.ndarray -- 3x3x3 array containing the isotropic, antisymmetric
            and symmetric parts of the tensor
        
        """
        if self._sph is None:
            self._sph = np.zeros((3, 3, 3))
            # Isotropic part
            self._sph[0] = np.eye(3) * self.trace / 3
            # Anti-symmetric part
            self._sph[1] = (self._data - self._data.T) / 2.0
            # Symmetric part - isotropic part
            self._sph[2] = self._symm - self._sph[0]
            # _sph[0] + _sph[1] + _sph[2] == _data

        return self._sph.copy()

    def euler_angles(self, convention: str = "zyz", passive: bool = False, degrees=False) -> np.ndarray:
        """Euler angles of the Principal Axis System

        Return Euler angles of the PAS for this tensor in the
        required convention (currently supported: zyz, zxz).

        Args:
            convention {str} -- Euler angles convention to use
            (default: {'zyz'})
            passive {bool} -- Whether to return the passive Euler angles
            (default: {False})

        Returns:
            np.ndarray -- Euler angles in radians. Size of the array is (3,)
        """


        angles = _matrix_to_euler(self.eigenvectors, convention, passive)

        angles = _handle_euler_edge_cases(
                    angles,
                    self.eigenvalues,
                    self._symm,
                    convention = convention,
                    passive = passive
                    )
        # finally, test that the angles give a consistent rotation
        consistent_rotation = _test_euler_rotation(
                                    angles,
                                    self.eigenvalues,
                                    self.eigenvectors,
                                    convention,
                                    passive)
        if not consistent_rotation:
            warnings.warn("The Euler angles do not give a consistent rotation. "
                            "This is likely due to a degeneracy in the tensor. "
                            "Care must be taken when comparing the Euler angles of degenerate tensors.")

            # Re-running the Euler angle calculation with -self.eigenvectors
            self._evecs = -self.eigenvectors
            angles = _matrix_to_euler(self.eigenvectors, convention, passive)
            angles = _handle_euler_edge_cases(
                        angles,
                        self.eigenvalues,
                        self._symm,
                        convention = convention,
                        passive = passive
                        )

            # check again
            consistent_rotation = _test_euler_rotation(
                                    angles,
                                    self.eigenvalues,
                                    self.eigenvectors,
                                    convention,
                                    passive)
            # TODO raise error if still not consistent
            # if not consistent_rotation:
            #     raise ValueError("The Euler angles do not give a consistent rotation. "
            #                     "This is likely due to a degeneracy in the tensor. "
            #                     "Care must be taken when comparing the Euler angles of degenerate tensors.")

        if degrees:
            angles = np.degrees(angles)

        return angles

    def equivalent_euler_angles(self, convention="zyz", passive=False):
        '''
        Returns the equivalent Euler angles of the tensor.

        Args:
            convention {str} -- Euler angles convention to use
            (default: {'zyz'})
            passive {bool} -- Whether to return the passive Euler angles
            (default: {False})
        Returns:
            np.array -- Euler angles in radians. 
                        Size of the array is (4, 3) as there are 4 equivalent sets of Euler angles.


        '''
        euler_angles = self.euler_angles(convention, passive)

        return _equivalent_euler(euler_angles, passive=passive)

    def rotation_to(self, other):
        '''
        Returns the rotation matrix that rotates the tensor to the other tensor.

        '''
        R1 = self.eigenvectors
        R2 = other.eigenvectors
        R = np.linalg.inv(R1) @ R2
        # Need to guarantee that R expresses a *proper* rotation
        # (i.e. det(R) = 1)
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        return R

    def euler_to(self, other, convention="zyz", passive=False, eps = 1e-6):
        '''
        Returns the Euler angles that rotate the tensor to the other tensor.
        '''
        convention = convention.lower()
        # first make sure they're not the same!
        if np.allclose(self.eigenvalues, other.eigenvalues):
            # check eigenvectors are the same up to a sign
            if np.allclose(self.eigenvectors, other.eigenvectors) or np.allclose(self.eigenvectors, -other.eigenvectors):
                warnings.warn("The tensors are identical. Returning zero Euler angles.")
                return np.zeros(3)
        if self.degeneracy == 2 or other.degeneracy == 2:
            Aevals = self.eigenvalues
            Bevals = other.eigenvalues
            # B (other) in the reference frame of A (self) - from paper
            # B_at_A = np.linalg.inv(self._symm) @ other._symm

            # alternative from the code
            R_A = Rotation.from_matrix(self.eigenvectors).as_matrix() # normalises the rotation matrix
            R_B = Rotation.from_matrix(other.eigenvectors).as_matrix() # normalises the rotation matrix
            Rrel1 = np.linalg.inv(R_B) @ R_A
            Rrel2 = np.linalg.inv(R_A) @ R_B

            B_at_A = Rrel1.dot(np.diag(Aevals).dot(np.linalg.inv(Rrel1)))

            if self.degeneracy == 2 and other.degeneracy == 2:
                # Both are axially symmetric - need to be careful

                # If both are axially symmetric, then do the following:
                # quick check if the angles are all zero:
                if np.abs(B_at_A[1,2] + B_at_A[0,2] + B_at_A[0,1]) < eps:
                    warnings.warn("The tensors are perfectly aligned. Returning zero Euler angles.")
                    return np.zeros(3)

                if convention == 'zyz':
                    # If both are axially symmetric, then
                    alpha = 0
                    gamma = 0
                    beta = np.arcsin(np.sqrt(
                        (B_at_A[2, 2]  - Aevals[2]) /
                        (Aevals[0] - Aevals[2])
                        ))
                    beta = np.abs(beta) # we can choose the sign of beta arbitrarily
                    return np.array([alpha, beta, gamma])

                if convention == 'zxz':
                    # in this convention, it depends on the unique axis
                    # of the other tensor
                    # but the possible angles are:
                    a = np.pi / 2
                    b = np.arcsin(-1 * np.sqrt(
                                (B_at_A[2, 2]  - Aevals[2]) /
                                (Aevals[0] - Aevals[2])
                                ))
                    b = np.abs(b) # we can choose the sign arbitrarily
                    c = 0


                    if np.abs(Bevals[0] - Bevals[1]) < eps:
                        # Unique axis is z
                        return np.array([a, b, c]) # 90, arcsin(...), 0
                    elif np.abs(Bevals[1] - Bevals[2]) < eps:
                        # Unique axis is x
                        return np.array([c, a, b]) # 0, 90, arcsin(...)
                    else:
                        raise ValueError('Unexpected eigenvalue ordering for axially symmetric tensor'
                                            'in zxz convention. Eigenvalues are: ', Bevals)
            elif self.degeneracy == 2 and other.degeneracy == 1:
                # If self is axially symmetric, but other isn't
                # just get the angles from Rrel1
                angles = _matrix_to_euler(Rrel1, convention, False) # always active here!
                angles = _handle_euler_edge_cases(
                    angles,
                    Aevals,
                    self._symm,
                    convention = convention,
                    passive = passive
                )
                if passive:
                    angles = angles[::-1]
                return angles
            elif self.degeneracy == 1 and other.degeneracy == 2:
                # If other is axially symmetric, but self isn't
                # TODO: check if we need to re-order Rrel2 to match ordering convention chosen for other tensor
                angles = _matrix_to_euler(Rrel2, convention, True) # always passive here!
                angles = _tryallanglestest(angles, np.diag(Aevals), np.diag(Bevals), Rrel1, convention)
                angles = -1*_handle_euler_edge_cases(
                    -1*angles[::-1],
                    Bevals,
                    other._symm,
                    convention = convention,
                    passive = passive
                )
                if not passive:
                    angles= angles[::-1]

                return angles





            if convention not in ["zyz", "zxz"]:
                # if neither zyz nor zxz, warn
                warnings.warn('Euler angles for axially symmetric tensors are only corrected for zyz and zxz conventions.'
                                ' Returning the uncorrected Euler angles.')

        if other.degeneracy == 2 and self.degeneracy != 2:
            # then let's swap them around, then transpose the result
            # TODO: test this!
            R = other.rotation_to(self).T
        else:
            R = self.rotation_to(other)
        return _matrix_to_euler(R.T, convention, passive)

    def equivalent_euler_to(self, other, convention="zyz", passive=False):
        '''
        Returns the equivalent Euler angles that rotate the tensor to the other tensor.
        '''
        euler_angles = self.euler_to(other, convention, passive)

        return _equivalent_relative_euler(euler_angles, passive=passive)

    @staticmethod
    def make_dipolar(
        a,
        i,
        j,
        cell=[0, 0, 0],
        isotopes={},
        isotope_i=None,
        isotope_j=None,
        rotation_axis=None,
    ):
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
        r = pos[j] - pos[i] + np.dot(a.get_cell(), cell)
        rnorm = np.linalg.norm(r)
        gammas = _get_isotope_data(
            elems[[i, j]],
            "gamma",
            isotopes=isotopes,
            isotope_list=[isotope_i, isotope_j],
        )

        d = _dip_constant(rnorm * 1e-10, *gammas)

        evals = np.zeros(3)
        evecs = np.zeros((3, 3))

        if rotation_axis is None:
            axis = r / rnorm
        else:
            axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
            vp2 = np.dot(r / rnorm, axis) ** 2
            d = 0.5 * d * (3 * vp2 - 1)

        evals[2] = 2 * d
        evals[:2] = -d

        evecs[:, 2] = axis

        x, y = axis[:2]
        evecs[1, 0] = x
        evecs[0, 0] = -y
        evecs[:, 0] /= (x ** 2 + y ** 2) ** 0.5

        evecs[0, 1]
        evecs[0, 1] = -evecs[2, 2] * evecs[1, 0]
        evecs[1, 1] = evecs[2, 2] * evecs[0, 0]
        evecs[2, 1] = evecs[0, 2] * evecs[1, 0] - evecs[1, 2] * evecs[0, 0]

        if d < 0:
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

        D = [evals, evecs]

        return NMRTensor(D)

    def __repr__(self) -> str:
        """
        Return a string representation of the tensor
        """
        return f"NMRTensor(data={self.data})"

    def __str__(self) -> str:
        """
        Return a string representation of the tensor. Nicely print out the 3x3 matrix (data), the conventions, and the derived properties.
        """
        return f"NMRTensor with data: \n{self.data}\n\n" + \
                f"Isotropy: {self.isotropy:.5f}\n" + \
                f"Anisotropy: {self.anisotropy:.5f}\n" + \
                f"Asymmetry: {self.asymmetry:.5f}\n" + \
                f"Span: {self.span:.5f}\n" + \
                f"Skew: {self.skew:.5f}\n" + \
                f"Eigenvalues: {self.eigenvalues}\n" + \
                f"Eigenvectors: \n{self.eigenvectors}\n" + \
                f"Euler angles (deg): {self.euler_angles(degrees=True)}\n"


class MagneticShielding(NMRTensor):
    """MagneticShielding

    Class containing a magnetic shielding tensor, a subclass of NMRTensor

    It provides easy access to common representations of the tensor in common conventions
    such as: IPAC, Herzfeld-Berger, Haeberlen, Maryland, and Mehring.
    For more information on these conventions, see the documentation for the corresponding
    NamedTuple and also :cite:p:`Harris2008`

    """

    # Define some NamedTuples for the different representations
    # of the tensor. See here for a description of the different
    # representations:
    # http://anorganik.uni-tuebingen.de/klaus/nmr/index.php?p=conventions/csa/csa
    class IUPACNotaion(NamedTuple):
        """
        IUPAC convention :cite:p:`Mason1993` (equivalent to the Mehring convention :cite:p:`Mehring1983`). 
        Follows the high frequency-positive order.
        Therefore:
        :math:`\\sigma_{11}` corresponds to the direction of least shielding, with the highest frequency,
        :math:`\\sigma_{33}` corresponds to the direction of highest shielding, with the lowest frequency.
        
        .. math::
            \\sigma_{11} \\leq \\sigma_{22} \\leq \\sigma_{33}

        The isotropic value, :math:`\\sigma_{iso}`, is the average values of the principal components,
        and corresponds to the center of gravity of the line shape.

        Note that the IUPAC convention is equivalent to the Mehring convention.

        

        """
        sigma_iso: float
        sigma_11: float
        sigma_22: float
        sigma_33: float

        def __str__(self):
            return (f"IUPACNotaion/MehringNotation:\n"
                    f"  sigma_11:  {self.sigma_11:.5f}\n"
                    f"  sigma_22:  {self.sigma_22:.5f}\n"
                    f"  sigma_33:  {self.sigma_33:.5f}\n"
                    f"  sigma_iso: {self.sigma_iso:.5f}")

    # Herzfeld-Berger Notation
    class HerzfeldBergerNotation(NamedTuple):
        """
        Herzfeld-Berger convention :cite:p:`Herzfeld1980` uses the following parameters:

        * :math:`\\sigma_{iso}` : isotropic magnetic shielding
        * :math:`\\Omega = \\sigma_{33} - \\sigma_{11}` (where these are the max and min principal components respectively): the span 
        * :math:`\\kappa = 3(\\sigma_{iso} - \\sigma_{22}) / \\Omega` (where :math:`\\sigma_{22}` is the median principal component): the skew

        """
        sigma_iso: float
        omega: float
        kappa: float

        def __str__(self):
            return (f"HerzfeldBergerNotation/MarylandNotation:\n"
                    f"  sigma_iso (isotropy): {self.sigma_iso:.5f}\n"
                    f"  omega (span):     {self.omega:.5f}\n"
                    f"  kappa (skew):     {self.kappa:.5f}")
    class MarylandNotation(HerzfeldBergerNotation):
        """The same as the Herzfeld-Berger notation."""

    class HaeberlenNotation(NamedTuple):
        """
        Haeberlen convention :cite:p:`Haeberlen1976` follows this ordering:
        
        .. math::
            |\\sigma_{zz} - \\sigma_{iso} | \\geq |\\sigma_{xx} - \\sigma_{iso} | \\geq |\\sigma_{yy} - \\sigma_{iso} |


        and uses the following parameters to describe the shielding tensor:

        * :math:`\\sigma_{iso}` : isotropic magnetic shielding
        * :math:`\\sigma = \\sigma_{zz} - \\sigma_{iso}` : the reduced anisotropy
        * :math:`\\Delta = \\sigma_{zz} - (\\sigma_{xx} + \\sigma_{yy}) / 2 = 3\\sigma / 2`  : the anisotropy
        * :math:`\\eta = (\\sigma_{yy} - \\sigma_{xx}) / \\sigma` : the asymmetry ( :math:`0 \\leq \\eta \\leq +1` )

        """

        sigma_iso: float
        sigma: float
        delta: float
        eta: float

        def __str__(self):
            return (f"HaeberlenNotation:\n"
                    f"  sigma_iso (isotropy): {self.sigma_iso:.5f}\n"
                    f"  sigma (reduced anisotropy): {self.sigma:.5f}\n"
                    f"  delta (anisotropy): {self.delta:.5f}\n"
                    f"  eta (asymmetry): {self.eta:.5f}")

    class MehringNotation(IUPACNotaion):
        """
        Mehring convention :cite:p:`Mehring1983` is equivalent to the IUPAC convention.
        """


    def __init__(self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        species: str,
        order:str = NMRTensor.ORDER_HAEBERLEN,
        reference:Optional[float]=None,
        gradient: float=-1.0,
        tag = None):
        """
        Initialise the MSTensor

        Create an MSTensor object from a 3x3 matrix.

        Arguments:
            species (str):      Element or isotope symbol of the atom the tensor refers to. e.g 'H', 'C', '13C'
            data (np.ndarray or tuple):  3x3 matrix containing the tensor, or
                                         pair [evals, evecs] for the symmetric
                                         part alone.
            order (str):        Order to use for eigenvalues/eigenvectors. Can
                                be 'i' (ORDER_INCREASING), 'd'
                                (ORDER_DECREASING), 'h' (ORDER_HAEBERLEN) or
                                'n' (ORDER_NQR). Default is 'h' for MS tensors.
            references (dict):  Dictionary of references for the magnetic shielding 
                                to chemical shift conversion. Keys are element symbols,
                                values are the reference chemical shift in ppm.
            gradients (dict):   Dictionary of gradients for the magnetic shielding
                                tensor. Keys are element symbols, values are the
                                gradient. Any unspecified gradients will be set to -1.
            tag (str):          Optional tag to identify the tensor. In a magres file, this would be the
                                'ms_sometag' though for most magres file ms is not decomposed into contributions and the tag
                                is simply 'ms'. By default, this is None.


        """
        super().__init__(data, order=order)
        self.species = species
        self.reference = reference
        self.gradient = gradient
        self.mstag = tag

    @property
    def element(self):
        '''
        Returns the element of the tensor. Species could have the isotope info, but
        here we just want the element.
        e.g. 13C -> C
        1H -> H
        '''
        return self.species.strip('1234567890')

    @property
    def shift(self):
        '''
        Returns the isotropic chemical shift of the tensor (ppm).
        If the reference is not set, will raise a ValueError. You can set the reference
        using the reference attribute.
        '''
        if self.reference is None:
            raise ValueError('Reference chemical shift not set for this tensor.')

        return self.reference + (self.gradient * self.isotropy) / (1 + self.reference * 1e-6)

    @property
    def shift_tensor(self):
        '''
        Returns the chemical shift tensor in ppm.
        '''
        if self.reference is None:
            raise ValueError('Reference chemical shift not set for this tensor.')
        shift_data = self.reference + (self.gradient * self.data) / (1 + self.reference * 1e-6)
        return MagneticShielding(shift_data, self.species, order=self.order, reference=self.reference, gradient=self.gradient)

    @property
    def haeberlen_values(self):
        """The magnetic shielding tensor in Haeberlen Notation."""
        sigma_iso = self.isotropy
        sigma = self.reduced_anisotropy
        delta = self.anisotropy
        eta = self.asymmetry
        return self.HaeberlenNotation(sigma_iso, sigma, delta, eta)

    @property
    def mehring_values(self):
        """The magnetic shielding tensor in Mehring Notation."""
        sigma_iso = self.isotropy
        # sort the eigenvalues in increasing order
        sigma_11, sigma_22, sigma_33 = sorted(self.eigenvalues)
        return self.MehringNotation(sigma_iso, sigma_11, sigma_22, sigma_33)

    @property
    def iupac_values(self):
        """The magnetic shielding tensor in IUPAC Notation."""
        sigma_iso = self.isotropy
        # sort the eigenvalues in increasing order
        sigma_11, sigma_22, sigma_33 = sorted(self.eigenvalues)
        return self.IUPACNotaion(sigma_iso, sigma_11, sigma_22, sigma_33)

    @property
    def maryland_values(self):
        """The magnetic shielding tensor in Maryland Notation."""
        sigma_iso = self.isotropy
        omega = self.span
        kappa = self.skew
        return self.MarylandNotation(sigma_iso, omega, kappa)
    @property
    def herzfeldberger_values(self):
        """The magnetic shielding tensor in Herzfeld-Berger Notation."""
        sigma_iso = self.isotropy
        omega = self.span
        kappa = self.skew
        return self.HerzfeldBergerNotation(sigma_iso, omega, kappa)

    # Set/update the reference chemical shift
    def set_reference(self, reference: float):
        '''
        Set the reference chemical shift for this tensor.
        '''
        self.reference = reference

    # Set/update the gradient
    def set_gradient(self, gradient: float):
        '''
        Set the gradient for this tensor.
        '''
        self.gradient = gradient

    def __str__(self):
        """
        Neatly formatted string representation of the tensor and Haeberlen description.
        """
        s =  f"Magnetic Shielding Tensor for {self.species}:\n"
        # 3x3 tensor representation neatly formatted with 5 decimal places
        s += str(np.array2string(self.data, precision=5, separator=",", suppress_small=True)) + "\n"
        s += str(self.haeberlen_values)
        return s


class ElectricFieldGradient(NMRTensor):
    """ElectricFieldGradient

    Class containing an electric field gradient tensor, a subclass of NMRTensor.

    The EFG tensor is a symmetric 3x3 matrix, with a trace of zero and has therefore
    only 5 independent degrees of freedom. These are often captured using the following: 

    * largest absolute eigenvalue (:math:`V_{zz}`)
    * the asymmetry parameter ( :math:`\\eta` )
    *  α, β, γ Euler angles describing the orientation of the tensor.

    The principal components of the tensor are the eigenvalues of the tensor, and they
    are usually sorted according to the NQR convention ( :math:`|V_{zz}| \\geq |V_{yy}| \\geq |V_{xx}|` ). Note however, that 
    Simpson uses the convention :math:`|V_{zz}| \\geq |V_{xx}| \\geq |V_{yy}|` . This is can be compensated for by using the
    order parameter when creating the tensor.

    Note that some conventions use the reduced anisotropy ( :math:`\\zeta` ) instead of the asymmetry parameter.

    """

    def __init__(self,
                data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                species: str,
                order: str=NMRTensor.ORDER_NQR,
                quadrupole_moment:Optional[float] = None,
                gamma:Optional[float] = None):
        """
        Initialise the EFGTensor

        Create an EFGTensor object from a 3x3 matrix.

        Args:
            data (np.ndarray or tuple):  3x3 matrix containing the tensor, or
                                         pair [evals, evecs] for the symmetric
                                         part alone.
            species (str):      Isotope symbol of the atom the tensor refers to. e.g '2H', '13C'
            order (str):        Order to use for eigenvalues/eigenvectors. Can
                                be 'i' (ORDER_INCREASING), 'd'
                                (ORDER_DECREASING), 'h' (ORDER_HAEBERLEN) or
                                'n' (ORDER_NQR). Default is 'h' for EFG tensors.
            quadrupole_moment (float): Quadrupole moment of the nucleus in millibarn.
            gamma (float):      Nuclear gyromagnetic ratio in rad/(s*T). If not provided, will be
                                looked up using the species.
        """
        super().__init__(data, order=order)
        self.species = species
        isotope_number, element = _split_species(species)
        self.element = element

        # Nuclear spin in Bohr magnetons
        self.spin = nmr_spin(element, iso=isotope_number)

        # Quadrupole moment in barns
        self.quadrupole_moment = quadrupole_moment or nmr_quadrupole(element, iso=isotope_number)

        # gamma in rad/(s*T)
        self.gamma = gamma or nmr_gamma(element, iso=isotope_number)



    @property
    def eta(self):
        '''
        Returns the asymmetry parameter of the tensor.
        '''
        return self.asymmetry

    @property
    def zeta(self):
        '''
        Returns the reduced anisotropy of the tensor.
        '''
        return self.reduced_anisotropy

    @property
    def Vzz(self):
        '''
        Returns the largest absolute eigenvalue of the tensor (
        the principal component of the EFG tensor).
        This should be in atomic units (a.u.) if read in from e.g. a magres file.
        '''
        # The eigenvalues are already sorted in abs increasing order (NQR convention)
        if self.order == self.ORDER_NQR:
            return self.eigenvalues[2]
        else:
            # warn user that the eigenvalues are not sorted in NQR order
            warnings.warn("The eigenvalues are not sorted in NQR order. "
                            "Returning the largest abs. eigenvalue following NQR convention.")
            # sort them in NQR order
            return _evals_sort(self.eigenvalues, convention=self.ORDER_NQR)[2]

    @property
    def Cq(self) -> float:
        '''
        Calculates the quadrupolar constant in Hz for this EFG tensor.
        The constant will be zero for non-quadrupole active nuclei.
        The quadrupole moment used is that for the nucleus of the isotope
        specified in the species attribute.

        This property is defined as

        .. math::

            C_Q = \\frac{e^2qQ}{h}

        in Hz. It is important to keep in mind that therefore this represents a
        *frequency*; the corresponding 'omega' (pulsation) would be the same value
        multiplied by 2*pi. This is, for example, exactly the value required as
        input in Simpson's SPINSYS section.

        Returns:
            float: Quadrupolar constant value in Hz.
        '''

        return EFG_TO_CHI * self.quadrupole_moment * self.Vzz

    @property
    def Pq(self) -> float:
        '''
        Calculates the quadrupolar product in Hz for this EFG tensor.

        .. math::

            P_Q = C_Q (1+\\frac{\\eta_Q^2}{3})^{1/2}

        Returns:
            float: Quadrupolar product value in Hz.
        '''
        return self.Cq * (1 + self.eta ** 2 / 3)**0.5

    @property
    def nuq(self) -> float:
        '''
        Calculates the quadrupolar frequency in Hz for this EFG tensor.
        This is also known as the quadrupolar splitting parameter.

        .. math::

            \\nu_Q = \\frac{3C_{Q}}{2I(2I-1)}

        where I is the nuclear spin of the nucleus.

        See this for conventions:
        http://anorganik.uni-tuebingen.de/klaus/nmr/index.php?p=conventions/efg/quadtools

        Returns:
            float: Quadrupolar frequency value in Hz.
        '''
        return 3 * self.Cq / (2 * self.spin * (2 * self.spin - 1))

    def get_larmor_frequency(self, Bext):
        '''
        Returns the Larmor frequency of the nucleus in an external magnetic field in Hz.

        .. math::
            
                \\nu_L = \\gamma B_{ext} / (2\\pi)

        where :math:`\\gamma` is the gyromagnetic ratio of the nucleus in rad/(s*T) and :math:`B_{ext}` is the external magnetic field in T.

        Args:
            Bext (float): External magnetic field in T.

        Returns:
            float: Larmor frequency in Hz.
        '''
        return self.gamma * Bext / (2 * np.pi)


    def get_quadrupolar_perturbation(self, Bext):
        '''
        Returns the perturbation of the quadrupolar Hamiltonian due to an external magnetic field.
        The perturbation is given by:

        .. math::

            a = \\frac{\\nu_Q^2}{\\nu_L} (I(I+1) - 3/2)

        where :math:`\\nu_Q` is the quadrupolar frequency and :math:`\\nu_L` is the Larmor frequency.

        Args:
            Bext (float): External magnetic field in T.

        Returns:
            float: Perturbation in Hz.
        '''
        nu_larmor = self.get_larmor_frequency(Bext)
        spin = self.spin
        nuq = self.nuq
        a = (nuq**2 / nu_larmor) * (spin*(spin+1) - 3/2)
        return a

    def get_MAS_full_max_linewidth(self, Bext):
        '''
        Returns the MAS full maximum line width (Hz) of the central transition.

        Defined as:

        .. math::
            
                \\Delta\\nu = a\\frac{(6+\\eta)^2}{504}

        where :math:`a` is the quadrupolar perturbation and :math:`\\eta` is the asymmetry parameter.

        Args:
            Bext (float): External magnetic field in T.

        Returns:
            float: MAS full maximum line width in Hz.
        '''
        a = self.get_quadrupolar_perturbation(Bext)
        return a * (6 + self.eta)**2 / 504

    # def get_MAS_second_order_shift(self, Bext):
    #     '''
    #     Returns the MAS second order shift (ppm) of the central transition.

    #     TODO: eq. taken from here: http://anorganik.uni-tuebingen.de/klaus/nmr/index.php?p=conventions/efg/quadtools
    #     double-check.

    #     Defined as:

    #     .. math::

    #             \\nu_{c.g.} = \\nu_{0} - \\frac{1}{30} a (1 + \\eta^{\\frac{2}{3}})

    #     Args:
    #         Bext (float): External magnetic field in T.

    #     Returns:
    #         float: MAS second order shift in ppm.
    #     '''
    #     a = self.get_quadrupolar_perturbation(Bext)
    #     nu_larmor = self.get_larmor_frequency(Bext)
    #     return nu_larmor - (a / 30) * (1 + self.eta**(2/3))


