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
from collections import defaultdict
from enum import Enum
from typing import Any, Literal, NamedTuple, Optional, Tuple, TypeVar, Union

from numpy.typing import NDArray
import numpy as np
from ase.quaternions import Quaternion
from numpy.lib.mixins import NDArrayOperatorsMixin

from soprano.data.nmr import EFG_TO_CHI, _get_isotope_data, nmr_gamma, nmr_quadrupole, nmr_spin
from soprano.nmr.utils import (
    _anisotropy,
    _asymmetry,
    _dip_constant,
    _equivalent_euler,
    _equivalent_relative_euler,
    _evals_sort,
    _evecs_2_quat,
    _frange,
    _haeb_sort,
    _alpha_zero_euler_angles,
    _alpha_zero_gauge,
    _degeneracy_pattern,
    _handle_euler_edge_cases,
    _matrix_to_euler,
    _normalise_euler_angles,
    _skew,
    _span,
    _split_species,
    _test_euler_rotation,
)

DEGENERACY_TOLERANCE = 1e-6


def _proper(E):
    """Eigenvector matrix as a proper rotation (the sign of an axis is arbitrary)."""
    E = np.array(E, dtype=float)
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1
    return E


def _both_axial_euler_angles(theta, axis_a, axis_b, convention, passive):
    """Relative Euler angles of two axial tensors whose unique axes make the angle ``theta``.

    ``axis_a`` and ``axis_b`` are the PAS axes (0 = x, 2 = z) that carry the unique
    eigenvalue of the first and second tensor. Rotating either tensor about its unique axis
    changes nothing, so any rotation ``R`` with ``|R[axis_a, axis_b]| = cos(theta)`` describes
    the pair. Following TensorView (Svenningsson & Mueller 2023, Section 6, Table 1):

    * ZYZ: ``(0, beta, 0)``, ``R = Ry(beta)``; ``beta = theta`` if the unique axes have the
      same label and ``pi/2 - theta`` otherwise (the paper's ``asin(...)`` formula).
    * ZXZ, first tensor unique along z: ``(pi/2, theta, 0)`` if the second is unique along z,
      ``(0, pi/2, pi/2 - theta)`` if it is unique along x (Table 1).
    * ZXZ, first tensor unique along x (not covered by the paper): ``(0, 0, theta)`` or
      ``(pi/2, pi/2 - theta, 0)``.

    ``passive=True`` returns the passive angles of the same rotation (``R_passive(a, b, g) =
    R_active(-g, -b, -a)``, TensorView Eq. 9) with a non-negative beta. The caller
    normalises the range.
    """
    half_pi = np.pi / 2
    if convention == "zyz":
        beta = theta if axis_a == axis_b else half_pi - theta
        active = [0.0, beta, 0.0]
    elif axis_a == 2:
        active = [half_pi, theta, 0.0] if axis_b == 2 else [0.0, half_pi, half_pi - theta]
    else:
        active = [0.0, 0.0, theta] if axis_b == 0 else [half_pi, half_pi - theta, 0.0]
    if not passive:
        return np.array(active)
    a, b, g = -active[2], -active[1], -active[0]
    if b < 0:  # R(a, b, g) = R(a + pi, -b, g + pi) (TensorView Eq. 8)
        a, b, g = a + np.pi, -b, g + np.pi
    return np.array([a, b, g]) % (2 * np.pi)


class TensorConvention(str, Enum):
    Haeberlen = "h"
    Increasing = "i"
    Decreasing = "d"
    NQR = "n"

    @classmethod
    def from_input(cls, input_str: str) -> 'TensorConvention':
        normalized = input_str.lower().strip()
        conversion_map = {
            'h': cls.Haeberlen,
            'haeberlen': cls.Haeberlen,
            'i': cls.Increasing,
            'increasing': cls.Increasing,
            'd': cls.Decreasing,
            'decreasing': cls.Decreasing,
            'n': cls.NQR,
            'nqr': cls.NQR
        }

        try:
            return conversion_map[normalized]
        except KeyError:
            raise ValueError(f"Invalid convention: {input_str}")

T = TypeVar("T", bound="NMRTensor")

class NMRTensor(NDArrayOperatorsMixin):
    """NMRTensor

    Class containing an NMR tensor, useful to access all its most important
    properties and representations.
    """

    ORDER_INCREASING = TensorConvention.Increasing
    ORDER_DECREASING = TensorConvention.Decreasing
    ORDER_HAEBERLEN = TensorConvention.Haeberlen
    ORDER_NQR = TensorConvention.NQR

    def __init__(self,
                 data: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
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

        self._incr_evals = _evals_sort([self._evals], 'i')[0]
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
            # Pairwise comparison so that patterns like [a, b, b] or [a, a, b]
            # return 2 regardless of which value is at index 0.
            diffs = np.abs(self._evals[:, None] - self._evals[None, :])
            self._degeneracy = int(np.max(np.sum(diffs < DEGENERACY_TOLERANCE, axis=1)))
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

    def euler_angles(
            self,
            convention: Literal["zyz", "zxz"] = "zyz",
            passive: bool = False,
            degrees: bool = False
        ) -> NDArray[np.float64]:
        """
        Return Euler angles of the PAS for this tensor in the
        required convention (currently supported: zyz, zxz).

        With ``E`` the eigenvector matrix (the PAS axes as columns, in this tensor's
        eigenvalue order) the tensor is ``E diag(eigenvalues) E^T``, and the angles are
        defined as in TensorView for MATLAB :cite:p:`Svenningsson2023` (Eqs. 3, 9, 65, 75),
        with ``Rz``, ``Ry``, ``Rx`` ordinary right-handed rotation matrices:

        * *active* (``passive=False``): ``E = Rz(alpha) Ry(beta) Rz(gamma)`` for ``zyz``
          and ``E = Rz(alpha) Rx(beta) Rz(gamma)`` for ``zxz``;
        * *passive* (``passive=True``): the angles of the inverse rotation,
          ``E = Rz(-gamma) Ry(-beta) Rz(-alpha)`` (``Rx`` for ``zxz``). This is the
          convention SIMPSON uses.

        A tensor is unchanged by a half turn about any PAS axis, so four angle sets describe
        it (see :meth:`equivalent_euler_angles`). The paper allows ``0 <= beta <= pi``; this
        method returns the same member as TensorView's code (``MFtoEuler.m``): beta in
        [0, pi/2), and then gamma in [0, pi) for active angles or alpha in [0, pi) for passive
        ones.

        Degenerate tensors have a free angle, which is chosen as in TensorView. With the
        unique axis along z (the Haeberlen order), the active gamma is zero and the passive
        alpha is zero. With the unique axis along x (e.g. increasing order), the frame with
        active alpha = 0 is chosen and the range is then normalised as above, so the active
        alpha is 0 or pi and the passive gamma is 0 or pi. A spherical tensor has all angles
        zero.

        Args:
            convention {Literal["zyz", "zxz"]} -- Euler angles convention to use
            (default: {'zyz'})
            passive {bool} -- Whether to return the passive Euler angles
            (default: {False})
            degrees {bool} -- Return degrees rather than radians (default: {False})

        Returns:
            np.ndarray -- Euler angles in radians. Size of the array is (3,)

        .. footbibliography::
        """


        def pas_angles():
            if convention.lower() in ("zyz", "zxz") and _degeneracy_pattern(self.eigenvalues)[1] == 0:
                # Unique axis along x: the PAS can be turned freely about x. Choose the frame
                # with alpha = 0 (TensorView), then read the angles off that frame.
                # TensorView then normalises the range as usual, which turns alpha into pi
                # whenever beta >= pi/2.
                frame = _alpha_zero_gauge(_proper(self.eigenvectors), convention)
                if not passive:
                    return _normalise_euler_angles(_alpha_zero_euler_angles(frame, convention))
                return _matrix_to_euler(frame, convention, passive)
            angles = _matrix_to_euler(self.eigenvectors, convention, passive)
            return _handle_euler_edge_cases(
                angles,
                self.eigenvalues,
                convention = convention,
                passive = passive,
                angles_are_passive = passive
                )

        angles = pas_angles()
        # finally, test that the angles give a consistent rotation
        consistent_rotation = _test_euler_rotation(
                                    angles,
                                    self.eigenvalues,
                                    self.eigenvectors,
                                    convention,
                                    passive)
        if not consistent_rotation:
            if self.degeneracy == 1:
                # Only warn when the tensor is non-degenerate: inconsistency is unexpected there.
                warnings.warn("The Euler angles do not give a consistent rotation. "
                                "This is likely due to a degeneracy in the tensor. "
                                "Care must be taken when comparing the Euler angles of degenerate tensors.")

            # Re-running the Euler angle calculation with -self.eigenvectors
            self._evecs = -self.eigenvectors
            angles = pas_angles()

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

    def mrsimulator_euler_angles(self, degrees: bool = False) -> NDArray[np.float64]:
        """
        Euler angles of the PAS in the convention used by mrsimulator's code.

        mrsimulator builds the eigenvector matrix (the PAS axes as columns) as
        ``Rz(gamma) Ry(beta) Rz(alpha)`` with ordinary active rotation matrices; this is
        what ``mrsimulator.utils.cartesian_tensor`` does (``Rotation.from_euler("zyz", ...)``).
        In TensorView's terms (see :meth:`euler_angles`) these are the *active* ZYZ angles
        with ``alpha`` and ``gamma`` interchanged. They are not the passive angles from
        :meth:`euler_angles`, which satisfy ``Rz(-gamma) Ry(-beta) Rz(-alpha)``, and not the
        active angles either: Appendix B of the mrsimulator paper writes the active order, which
        the code does not follow.

        Both describe the same tensor frame; this method converts the (canonicalised)
        passive angles exactly, as ``(-alpha - pi, beta, pi - gamma)``. For a tensor
        with its unique axis along z the first angle is free and is set to zero.

        Args:
            degrees {bool} -- Return degrees rather than radians (default: {False})

        Returns:
            np.ndarray -- Euler angles, shape (3,), in the range [0, 2*pi)
        """
        if self.degeneracy == 3:
            return np.zeros(3)

        alpha, beta, gamma = self.euler_angles(convention="zyz", passive=True)
        evals = self.eigenvalues
        unique_axis_is_z = abs(evals[0] - evals[1]) < DEGENERACY_TOLERANCE
        alpha_mrs = 0.0 if unique_axis_is_z else -alpha - np.pi
        angles = np.array([alpha_mrs, beta, np.pi - gamma]) % (2 * np.pi)
        # a value that is 2*pi within rounding is zero
        angles[angles > 2 * np.pi - 1e-12] = 0.0

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
        Returns the relative rotation ``R = E_selfᵀ E_other`` (TensorView Eq. 31 with
        A = self, B = other): the rotation that carries this tensor's PAS onto ``other``'s,
        so ``E_other = E_self R``. Here ``E`` is the matrix of PAS axes (eigenvectors as columns)
        and the result is made a proper rotation. ``euler_to`` returns the Euler angles of
        this rotation.
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
        Returns the Euler angles of the rotation carrying this tensor's PAS onto ``other``'s.

        The rotation is ``R = E_selfᵀ E_other`` (see :meth:`rotation_to`), i.e. the orientation
        of ``other`` in the PAS of this tensor (TensorView Eqs. 29-32 with A = self and
        B = other). The *active* angles (``passive=False``) satisfy
        ``E_other = E_self R(alpha, beta, gamma)``; ``passive=True`` gives the angles of the
        inverse parametrisation of the same R (``R = Rz(-gamma) Ry(-beta) Rz(-alpha)``), exactly
        as for :meth:`euler_angles`. A single representative of the 16 equivalent sets is
        returned (:meth:`equivalent_euler_to` returns all), normalised to the same range as
        :meth:`euler_angles`.

        In TensorView's software, ``RelativeAngles(Tensor1, Tensor2)`` gives the orientation
        of Tensor 1 in the PAS of Tensor 2 (it decomposes ``U2^-1 U1``), so
        ``A.euler_to(B)`` is TensorView with Tensor 1 = ``B`` and Tensor 2 = ``A``.

        A rotation about the unique axis of an axial tensor changes nothing. If exactly one
        tensor is axial, it is written in the PAS of the other and given the single-tensor
        gauge of :meth:`euler_angles`, as TensorView does. For a unique axis along z this sets
        the first active angle to zero if ``self`` is axial and the last if ``other`` is (the
        other way round for passive angles). If both are axial, the angles are ``(0, beta, 0)``
        for ZYZ (beta is the angle between the two unique axes, or its complement if exactly
        one of them is along x) and TensorView's Table 1 for ZXZ, each then normalised to the
        range above (so Table 1's ``(0, 90 deg, g)`` comes back as ``(180 deg, 90 deg,
        180 deg - g)``, as TensorView displays it). Identical or perfectly aligned tensors
        return zeros with a warning.
        '''
        convention = convention.lower()
        # first make sure they're not the same!
        if np.allclose(self.eigenvalues, other.eigenvalues):
            # check eigenvectors are the same up to a sign
            if np.allclose(self.eigenvectors, other.eigenvectors) or np.allclose(self.eigenvectors, -other.eigenvectors):
                warnings.warn("The tensors are identical. Returning zero Euler angles.")
                return np.zeros(3)
        if self.degeneracy == 2 or other.degeneracy == 2:
            if self.degeneracy == 2 and other.degeneracy == 2 and convention in ("zyz", "zxz"):
                # Both are axially symmetric: only the angle theta between the two unique
                # axes matters, and TensorView gives a representative with the free angles
                # zero (ZYZ) or from its Table 1 (ZXZ).
                axis_a = _degeneracy_pattern(self.eigenvalues, eps)[1]
                axis_b = _degeneracy_pattern(other.eigenvalues, eps)[1]
                if 1 in (axis_a, axis_b):
                    # same as euler_angles: not reachable with the supported orders
                    raise ValueError("Unexpected degeneracy when computing Euler angles.\n"
                                     "Eigenvalues are ordered: ", self.eigenvalues, other.eigenvalues)
                if axis_a in (0, 2) and axis_b in (0, 2):
                    R = _proper(self.eigenvectors).T @ _proper(other.eigenvectors)
                    cos_theta = min(abs(R[axis_a, axis_b]), 1.0)
                    theta = np.arctan2(np.sqrt(1.0 - cos_theta**2), cos_theta)
                    # "Perfectly aligned" means the unique axes carry the same label (both z
                    # or both x) and coincide. With different labels the PAS frames are a
                    # quarter turn apart even when the axes are parallel, so there is a
                    # non-trivial rotation to report (and it is not a warning case).
                    if axis_a == axis_b and theta < eps:
                        warnings.warn("The tensors are perfectly aligned. Returning zero Euler angles.")
                        return np.zeros(3)
                    # TensorView's RelativeAngles.m normalises these like any other angles,
                    # e.g. Table 1's (0, 90 deg, g) is reported as (180 deg, 90 deg, 180 deg - g)
                    return _normalise_euler_angles(
                        _both_axial_euler_angles(theta, axis_a, axis_b, convention, passive), passive=passive)
            elif {self.degeneracy, other.degeneracy} == {1, 2} and convention in ("zyz", "zxz"):
                # Exactly one of the tensors is axially symmetric. As in TensorView's
                # RelativeAngles.m, write the axial tensor in the PAS of the triaxial one and
                # take its single-tensor angles, so that it gets the same gauge as in
                # euler_angles (unique axis along z or along x). If other is axial these are
                # the angles of B in the PAS of A directly. If self is axial, the active
                # angles of R are the passive angles of R^T (A in the PAS of B), and the
                # other way round.
                E_self, E_other = _proper(self.eigenvectors), _proper(other.eigenvectors)
                if other.degeneracy == 2:
                    axial, rel, sense = other, E_self.T @ E_other, passive
                else:
                    axial, rel, sense = self, E_other.T @ E_self, not passive
                unique_axis = _degeneracy_pattern(axial.eigenvalues, eps)[1]
                if unique_axis == 2 and np.hypot(rel[0, 2], rel[1, 2]) < eps:
                    # unique axis (anti)parallel to the triaxial tensor's z: only a rotation
                    # about it is left, which the axial tensor is blind to
                    return np.zeros(3)
                rel_tensor = NMRTensor((axial.eigenvalues, rel), order=axial.order)
                angles = rel_tensor.euler_angles(convention, passive=sense)
                return _normalise_euler_angles(angles, passive=passive)

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
        # Decompose the relative rotation ``rel = R_selfᵀ · R_other`` directly.
        # The *active* angles (passive=False) then satisfy
        # ``R_other = R_self · R(α,β,γ)`` — the intrinsic rotation carrying this
        # tensor's PAS onto ``other``'s PAS — consistent with the physical
        # convention and with ``euler_angles``. (Previously this decomposed
        # ``R.T``, which inverted the active/passive sense for euler_to.)
        return _matrix_to_euler(R, convention, passive)

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

    def make_isotropic_like(self):
        """
        Create an isotropic NMRTensor with the same isotropy value as this tensor.
        The initialisation parameters are kept the same.

        Returns:
            NMRTensor: An isotropic NMRTensor object.
        """
        data = np.eye(3) * self.isotropy
        return self._create_like(data)

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

    def __array__(self, dtype=None):
        """
        Return a numpy array representation of the tensor.
        Required for NDArrayOperatorsMixin.
        """
        return np.asarray(self._data, dtype=dtype)
    def __eq__(self, other: 'NMRTensor') -> bool:
        """
        Check if two NMRTensor objects are equal.
        
        NMRTensors are considered equal if they have:
        - The same order convention
        - Equivalent eigenvalues (within numerical precision)
        - Equivalent eigenvectors (within numerical precision)
        
        Parameters:
        other (NMRTensor): Another NMRTensor object to compare against
        
        Returns:
        bool: True if tensors are equivalent, False otherwise
        """
        # Quick type and identity checks
        if not isinstance(other, NMRTensor):
            return False

        # Check order convention
        if self.order != other.order:
            return False

        # Check eigenvalues
        # Use numpy's allclose to handle floating-point comparison
        if not np.allclose(self.eigenvalues, other.eigenvalues, rtol=1e-05, atol=1e-08):
            return False

        # Check eigenvectors
        # TODO verify if this is the best way to compare eigenvectors
        # Eigenvectors can have opposite signs and still represent the same tensor
        # So we check if the absolute values of eigenvectors are close
        for our_evec, other_evec in zip(self.eigenvectors.T, other.eigenvectors.T):
            if not (np.allclose(np.abs(our_evec), np.abs(other_evec), rtol=1e-05, atol=1e-08)):
                return False

        return True

    @property
    def _initialisation_params(self):
        """
        Return the parameters used for the initialisation of the tensor.
        This is useful for debugging and understanding how the tensor was created.
        """
        return {
            "order": self.order
        }

    def _create_like(self, data):
        """
        Create a new tensor of the same type with the provided data
        but preserving other properties of this tensor.
        
        Args:
            data (np.ndarray): New tensor data
            
        Returns:
            NMRTensor: A new tensor with the same properties
        """
        params = self._initialisation_params
        return self.__class__(data = data, **params)
    
    def copy(self):
        """
        Create a copy of the NMRTensor object.
        
        Returns:
            NMRTensor: A new NMRTensor object with the same data and properties.
        """
        return self._create_like(self.data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy's universal functions for tensor operations.
        This enables proper handling of arithmetic operations.
        """
        # Only handle the __call__ method of ufuncs
        if method != '__call__':
            return NotImplemented

        # Check if the operation is supported
        supported_ops = [
            np.add, np.subtract, np.multiply,
            np.true_divide, np.matmul,
            np.negative, np.positive,
            np.equal, np.not_equal]

        if ufunc not in supported_ops:
            return NotImplemented

        # Handle unary operations
        if len(inputs) == 1:
            if ufunc is np.negative or ufunc is np.positive:
                # Let subclasses handle their own instantiation if possible
                data = -self.data if ufunc is np.negative else self.data
                return self._create_like(data)
            return NotImplemented

        # Handle binary operations
        if len(inputs) == 2:
            # Find which input is self
            self_idx = None
            for i, x in enumerate(inputs):
                if isinstance(x, NMRTensor) and x is self:
                    self_idx = i
                    break

            if self_idx is None:
                return NotImplemented

            # Get the other operand
            other_idx = 1 - self_idx
            other = inputs[other_idx]

            if ufunc is np.equal:
                return self.__eq__(other)
            elif ufunc is np.not_equal:
                return not self.__eq__(other)

            # Convert inputs to numpy arrays
            tensor_data = self.data

            # Process the other operand based on its type
            if isinstance(other, NMRTensor):
                # Check parameter consistency if both are same subclass
                if isinstance(other, self.__class__):
                    self._check_binary_op_params(other)
                # Check if tensors have the same order, warn if not
                if other.order != self.order:
                    warnings.warn(f"Operating on tensors with different ordering conventions: "
                                f"{self.order} and {other.order}. Using {self.order}.")
                other_data = other.data
            elif isinstance(other, (int, float)):
                other_data = other
            elif isinstance(other, np.ndarray):
                # Special handling for matmul with vectors
                if ufunc is np.matmul and len(other.shape) == 1 and other.shape[0] == 3 or other.shape == (3, 3):
                    other_data = other
                else:
                    raise ValueError(f"Cannot perform operation with array of shape {other.shape}")
            else:
                return NotImplemented

            # Apply the operation with correct argument order
            args = (tensor_data, other_data) if self_idx == 0 else (other_data, tensor_data)
            result_data = ufunc(*args, **kwargs)

            # Return appropriate result based on the operation
            if ufunc in [np.add, np.subtract, np.multiply, np.true_divide]:
                return self._create_like(result_data)
            elif ufunc is np.matmul:
                if result_data.shape == (3, 3):
                    return self._create_like(result_data)
                else:
                    return result_data

        # For unsupported operations
        return NotImplemented

    def _check_binary_op_params(self, other):
        """
        Check that parameters of two tensors are compatible for binary operations.
        Warns about any inconsistencies.

        Args:
            other (NMRTensor): Another tensor to compare parameters with
        """
        # Convenience method to check parameters for two tensors
        self._check_compatible([self, other])


    @classmethod
    def _check_compatible(cls, tensors: list[T]) -> None:
        """
        Tensor compatibility checks.

        Args:
            tensors (list[T]): list of NMRTensor objects

        Raises:
            ValueError: If the tensors are not compatible
        """
        first_tensor = tensors[0]
        expected_params = first_tensor._initialisation_params

        param_keys = expected_params.keys()

        for t in tensors[1:]:
            tensor_params = t._initialisation_params
            for key in param_keys:
                if tensor_params.get(key) != expected_params.get(key):
                    raise ValueError(f"All tensors must have the same {key}. "
                                    f"Found: {[t._initialisation_params.get(key) for t in tensors]}")


    @classmethod
    def mean(cls: type[T], tensor_list: list[Any], axis: Optional[int]=None, weights: Optional[np.ndarray]=None) -> Union[T, list[T]]:
        """
        Calculate the mean of a list of NMRTensor objects along a specified axis.

        Args:
            tensor_list (list[Any]): list or nested list of NMRTensor objects to average.
                For a 2D array with shape [N, M], this represents N rows of M tensors.
            axis (int, optional): Axis along which the mean is computed:
                - None: Average all tensors into a single tensor
                - 0: Average over rows (result has length M)
                - 1: Average over columns (result has length N)
                - etc. for higher dimensions
            weights (np.ndarray, optional): Array of weights to use for weighted average.
                Must have length matching the dimension being averaged:
                - If axis=None: weights should have same length as flattened tensor_list
                - If axis=0: weights should have length N (one weight per row)
                - If axis=1: weights should have length M (one weight per column)

        Returns:
            Union[T, list[T]]: A new tensor or list of tensors with averaged values

        Raises:
            ValueError: If tensor_list is empty, has incompatible dimensions,
                       or if weights have incorrect shape
        """
        if not tensor_list:
            raise ValueError("Cannot calculate mean of an empty list")

        # Helper function for averaging a flat list of tensors
        def average_tensor_list(tensors, weights=None):
            if not tensors:
                raise ValueError("Cannot calculate mean of an empty list")

            try:
                cls._check_compatible(tensors)
            except ValueError as e:
                raise ValueError("It seems you are trying to average incompatible tensors. \n"
                                 f"{e}\n"
                                 " Did you mean to average over a different axis perhaps?\n"
                                 " If so, please use the axis argument (e.g. axis=0).\n")
            tensor_data = np.array([t.data for t in tensors])

            if weights is not None:
                if len(weights) != len(tensors):
                    raise ValueError(f"Weights length ({len(weights)}) doesn't match tensor count ({len(tensors)})")
                mean_data = np.average(tensor_data, axis=0, weights=weights)
            else:
                mean_data = np.mean(tensor_data, axis=0)

            return tensors[0]._create_like(mean_data)

        # Handle 1D list of tensors with axis=None
        if all(isinstance(t, NMRTensor) for t in tensor_list) and axis is None:
            return average_tensor_list(tensor_list, weights)

        # Convert to numpy object array for easier handling of dimensions
        tensor_array = np.array(tensor_list, dtype=object)
        array_shape = tensor_array.shape[:-2]  # Exclude the last two dimensions (3x3)

        # For axis=None, flatten the array and average all tensors
        if axis is None:
            flat_tensors = tensor_array.flatten().tolist()
            if weights is not None and len(weights) != len(flat_tensors):
                raise ValueError(f"For axis=None, weights length ({len(weights)}) must match "
                                 f"total tensor count ({len(flat_tensors)})")
            return average_tensor_list(flat_tensors, weights)

        # Validate axis
        if axis >= len(array_shape):
            raise ValueError(f"Axis {axis} out of bounds for array of dimension {len(array_shape)}")

        # Validate weights for the specified axis
        if weights is not None:
            if len(weights) != array_shape[axis]:
                raise ValueError(f"For axis={axis}, weights length ({len(weights)}) must match "
                                 f"dimension size ({array_shape[axis]})")

        # Use numpy's apply_along_axis to handle the averaging
        result_shape = list(array_shape)
        result_shape.pop(axis)  # Remove the dimension being averaged
        result = np.empty(result_shape, dtype=object)

        # Helper function to apply averaging along the specified axis
        def average_along_axis():
            # For each slice perpendicular to the specified axis
            for idx in np.ndindex(result_shape):
                idx_list = list(idx)
                if axis > 0:
                    idx_list.insert(axis, slice(None))
                else:
                    idx_list = [slice(None)] + idx_list

                # Extract tensors along the axis
                tensors_to_average = tensor_array[tuple(idx_list)].tolist()

                # Apply weights if provided
                if weights is not None:
                    avg_tensor = average_tensor_list(tensors_to_average, weights)
                else:
                    avg_tensor = average_tensor_list(tensors_to_average)

                result[idx] = avg_tensor
            return result

        # Handle 1D and 2D cases explicitly for clarity
        if len(array_shape) == 1:
            # Simple 1D case - single average
            return average_tensor_list(tensor_list, weights)
        elif len(array_shape) == 2:
            # Common 2D case - handle more explicitly
            if axis == 0:  # Average over rows
                result = np.empty(array_shape[1], dtype=object)
                for col in range(array_shape[1]):
                    # Extract all tensor objects from tensor_list in the column
                    col_tensors = [row[col] for row in tensor_list]
                    result[col] = average_tensor_list(col_tensors, weights)
                return result.tolist()
            elif axis == 1:  # Average over columns
                result = np.empty(array_shape[0], dtype=object)
                for row in range(array_shape[0]):
                    row_tensors = tensor_list[row]
                    result[row] = average_tensor_list(row_tensors, weights)
                return result.tolist()

        # Generic case for higher dimensions
        return average_along_axis().tolist()


    def __hash__(self) -> int:
        """
        Generate a hash for the NMRTensor object.
        
        This allows NMRTensor objects to be used in sets and as dictionary keys.
        
        Returns:
        int: A hash value for the NMRTensor object
        """

        def array_to_hashable(arr):
            if arr is None:
                return None
            # Round to avoid floating-point precision issues
            return tuple(np.round(arr.flatten(), decimals=8))

        # Hash components
        return hash((
            self.order,
            array_to_hashable(self.eigenvalues),
            array_to_hashable(self.eigenvectors),
        ))


def has_reference(func):
    """
    Decorator to check if a reference chemical shift has been set.
    Raises a ValueError if the reference is not set.
    """
    def wrapper(self, *args, **kwargs):
        if self.reference is None:
            raise ValueError(
                "Reference chemical shift not set for this tensor. "
                "Please set the reference using the `set_reference` method."
            )
        return func(self, *args, **kwargs)
    return wrapper

class MagneticShielding(NMRTensor):
    """MagneticShielding

    Class containing a magnetic shielding tensor, a subclass of NMRTensor

    It provides easy access to common representations of the tensor in common conventions
    such as: IUPAC, Herzfeld-Berger, Haeberlen, Maryland, and Mehring.
    For more information on these conventions, see the documentation for the corresponding
    NamedTuple and also :cite:p:`Harris2008`

    """
    default_order = TensorConvention.Haeberlen

    def __init__(self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        species: str,
        order:str = default_order,
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
            reference (float):  Reference chemical shift for the tensor. If set,
                                the isotropic chemical shift can be calculated.
                                Defaults to None.
            gradient (float):   Gradient for the magnetic shielding to shift conversion.
                                Defaults to -1.0.
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
    @has_reference
    def shift_anisotropy(self) -> float:
        shift_haeb_evals = self.shift_eigenvalues_haeberlen[None, :]
        return _anisotropy(shift_haeb_evals)[0]
    
    @property
    @has_reference
    def shift_reduced_anisotropy(self) -> float:
        shift_haeb_evals = self.shift_eigenvalues_haeberlen[None, :]
        return _anisotropy(shift_haeb_evals, reduced=True)[0]
    
    @property
    def shift_asymmetry(self) -> float:
        """
        The asymmetry is not dependent on the referencing
        so we just return the shielding asymmetry
        """
        return self.asymmetry

    @property
    @has_reference
    def shift_skew(self) -> float:
        """
        The absolute value of the skew parameter is 
        not dependent on the referencing but the sign is.
        """
        return _skew(self.shift_eigenvalues[None, :])[0]
    
    @property
    def shift_span(self) -> float:
        """
        The span is not dependent on the referencing
        so we just return the shielding span
        """
        return self.span

    # Define some NamedTuples for the different representations
    # of the tensor. See here for a description of the different
    # representations:
    # http://anorganik.uni-tuebingen.de/klaus/nmr/index.php?p=conventions/csa/csa
    class IUPACShielding(NamedTuple):
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
            return (f"IUPACShielding/MehringNotation:\n"
                    f"  sigma_11:  {self.sigma_11:.5f}\n"
                    f"  sigma_22:  {self.sigma_22:.5f}\n"
                    f"  sigma_33:  {self.sigma_33:.5f}\n"
                    f"  sigma_iso: {self.sigma_iso:.5f}")

    class IUPACShift(NamedTuple):
        """
        The same as the IUPACShielding, but for chemical shifts. i.e. sigma -> delta.
        Therefore:

        .. math::
            \\delta_{33} \\leq \\delta_{22} \\leq \\delta_{11}

        """
        delta_iso: float
        delta_11: float
        delta_22: float
        delta_33: float

        def __str__(self):
            return (f"IUPACShift:\n"
                    f"  delta_11:  {self.delta_11:.5f}\n"
                    f"  delta_22:  {self.delta_22:.5f}\n"
                    f"  delta_33:  {self.delta_33:.5f}\n"
                    f"  delta_iso: {self.delta_iso:.5f}")

    # Herzfeld-Berger Notation
    class HerzfeldBergerShielding(NamedTuple):
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
            return (f"HerzfeldBergerShielding/MarylandShielding:\n"
                    f"  sigma_iso (isotropy): {self.sigma_iso:.5f}\n"
                    f"  omega (span):     {self.omega:.5f}\n"
                    f"  kappa (skew):     {self.kappa:.5f}")

    class HerzfeldBergerShift(NamedTuple):
        """
        The same as the Herzfeld-BergerShielding, but for chemical shifts. i.e. sigma -> delta.
        Therefore there is a corresponding swap in the sign of the skew (but not the span) value:

        .. math::
            \\Omega = \\delta_{11} - \\delta_{33}
            \\kappa = 3(\\delta_{22} - \\delta_{iso}) / \\Omega
        """
        delta_iso: float
        omega: float
        kappa: float

        def __str__(self):
            return (f"HerzfeldBergerShift/MarylandShift:\n"
                    f"  delta_iso (isotropic shift): {self.delta_iso:.5f}\n"
                    f"  omega (span):     {self.omega:.5f}\n"
                    f"  kappa (skew):     {self.kappa:.5f}")





    class HaeberlenShielding(NamedTuple):
        """
        Haeberlen convention :cite:p:`Haeberlen1976` follows this ordering:
        
        .. math::
            |\\sigma_{zz} - \\sigma_{iso} | \\geq |\\sigma_{xx} - \\sigma_{iso} | \\geq |\\sigma_{yy} - \\sigma_{iso} |


        and uses the following parameters to describe the shielding tensor:

        * :math:`\\sigma_{iso}` : isotropic magnetic shielding
        * :math:`\\zeta = \\sigma_{zz} - \\sigma_{iso}` : the reduced anisotropy
        * :math:`\\Delta = \\sigma_{zz} - (\\sigma_{xx} + \\sigma_{yy}) / 2 = 3\\zeta / 2`  : the anisotropy
        * :math:`\\eta = (\\sigma_{yy} - \\sigma_{xx}) / \\zeta` : the asymmetry ( :math:`0 \\leq \\eta \\leq +1` )

        """

        sigma_iso: float
        zeta: float
        delta: float
        eta: float

        def __str__(self):
            return (f"HaeberlenShielding:\n"
                    f"  sigma_iso (isotropy): {self.sigma_iso:.5f}\n"
                    f"  zeta (reduced anisotropy): {self.zeta:.5f}\n"
                    f"  Delta (anisotropy): {self.delta:.5f}\n"
                    f"  eta (asymmetry): {self.eta:.5f}")

    class HaeberlenShift(NamedTuple):
        """
        The same as the HaeberlenShielding, but for chemical shifts. i.e. sigma -> delta.
        Therefore:

        .. math::
            |\\delta_{zz} - \\delta_{iso} | \\geq |\\delta_{xx} - \\delta_{iso} | \\geq |\\delta_{yy} - \\delta_{iso} |

        Note that there is a corresponding swap in the sign of the anisotropy and reduced anisotropy values.
        """
        delta_iso: float
        zeta: float
        delta: float
        eta: float

        def __str__(self):
            return (f"HaeberlenShift:\n"
                    f"  delta_iso (isotropic shift): {self.delta_iso:.5f}\n"
                    f"  zeta (reduced anisotropy): {self.zeta:.5f}\n"
                    f"  Delta (anisotropy): {self.delta:.5f}\n"
                    f"  eta (asymmetry): {self.eta:.5f}")

    class MehringShielding(IUPACShielding):
        """
        Mehring convention :cite:p:`Mehring1983` is equivalent to the IUPAC convention.
        """
    class MehringShift(IUPACShift):
        """
        The same as the MehringShielding, but for chemical shifts. i.e. sigma -> delta.
        """
    class MarylandShielding(HerzfeldBergerShielding):
        """The same as the HerzfeldBergerShielding notation."""

    class MarylandShift(HerzfeldBergerShift):
        """The same as the HerzfeldBergerShift."""

    @property
    def element(self):
        '''
        Returns the element of the tensor. Species could have the isotope info, but
        here we just want the element.
        e.g. 13C -> C
        1H -> H
        '''
        return self.species.strip('1234567890')

    @staticmethod
    def _get_referenced_eigenvalues(eigenvalues: Union[list, np.ndarray], reference: float, gradient: float):
        """
        Get the referenced eigenvalues for the magnetic shielding tensor.

        Args:
            eigenvalues (Union[list, np.ndarray]): Eigenvalues of the magnetic shielding tensor (ppm)
            reference (float): Reference shielding (ppm)
            gradient (float): Gradient for the magnetic shielding to shift conversion
        Returns:
            np.ndarray: Chemical shift eigenvalues (ppm)
        """
        # Convert to numpy array if not already
        eigenvalues = np.array(eigenvalues)
        # Reference components
        shift_eigenvalues = (reference + gradient * eigenvalues) / (1 - reference * 1e-6)
        return shift_eigenvalues

    @property
    @has_reference
    def shift_eigenvalues(self) -> np.ndarray:
        '''
        Returns the eigenvalues of the shift tensor in ppm in descending order. i.e. 
        
        :math:`\\delta_{11} \\geq \\delta_{22} \\geq \\delta_{33}`

        If the reference is not set, will raise a ValueError. You can set the reference
        using the reference attribute.
        '''
        return self._get_referenced_eigenvalues(self._incr_evals, self.reference, self.gradient) # type: ignore
    
    # For convenience, we can also easily get the Haeberlen-ordered shift eigenvalues
    @property
    @has_reference
    def shift_eigenvalues_haeberlen(self) -> np.ndarray:
        '''
        Returns the eigenvalues of the shift tensor in ppm in Haeberlen order. i.e. 
        
        :math:`|\\delta_{zz} - \\delta_{iso}|  \\geq  |\\delta_{xx} - \\delta_{iso}| \\geq |\\delta_{yy} - \\delta_{iso}|`

        If the reference is not set, will raise a ValueError. You can set the reference
        using the reference attribute.
        '''
        return self._get_referenced_eigenvalues(self._haeb_evals, self.reference, self.gradient) # type: ignore
    
    @property
    @has_reference
    def shift(self) -> float:
        '''
        Returns the isotropic chemical shift of the tensor (ppm).
        If the reference is not set, will raise a ValueError. You can set the reference
        using the reference attribute.
        '''
        return np.mean(self.shift_eigenvalues)

    @property
    def haeberlen_shielding(self) -> HaeberlenShielding:
        """The magnetic shielding tensor in Haeberlen Notation."""
        return self.HaeberlenShielding(
            self.isotropy,
            self.reduced_anisotropy,
            self.anisotropy,
            self.asymmetry)

    @property
    @has_reference
    def haeberlen_shift(self) -> HaeberlenShift:
        """The chemical shift tensor in Haeberlen Notation."""
        return self.HaeberlenShift(
            self.shift,
            self.shift_reduced_anisotropy,
            self.shift_anisotropy,
            self.shift_asymmetry)

    @property
    def mehring_shielding(self) -> MehringShielding:
        """The magnetic shielding tensor in Mehring Notation."""
        evals = self.eigenvalues
        # sort the eigenvalues in increasing order
        sigma_11, sigma_22, sigma_33 = sorted(evals)
        return self.MehringShielding(
            self.isotropy,
            sigma_11,
            sigma_22,
            sigma_33)



    @property
    @has_reference
    def mehring_shift(self) -> MehringShift:
        """The chemical shift tensor in Mehring Notation."""
        delta_11, delta_22, delta_33 = self.shift_eigenvalues
        shift = (delta_11 + delta_22 + delta_33) / 3
        return self.MehringShift(
            shift,
            delta_11,
            delta_22,
            delta_33
        )
    
    @property
    def iupac_shielding(self) -> IUPACShielding:
        """The magnetic shielding tensor in IUPAC Notation."""
        sigma_iso = self.isotropy
        # sort the eigenvalues in increasing order
        sigma_11, sigma_22, sigma_33 = sorted(self.eigenvalues)
        return self.IUPACShielding(sigma_iso, sigma_11, sigma_22, sigma_33)
    
    @property
    @has_reference
    def iupac_shift(self) -> IUPACShift:
        """The chemical shift tensor in IUPAC Notation."""
        delta_iso = self.shift
        delta_11, delta_22, delta_33 = self.shift_eigenvalues
        return self.IUPACShift(delta_iso, delta_11, delta_22, delta_33)

    @property
    def maryland_shielding(self) -> MarylandShielding:
        """The magnetic shielding tensor in Maryland Notation."""
        iso = self.isotropy
        omega = self.span
        kappa = self.skew
        return self.MarylandShielding(iso, omega, kappa)

    @property
    @has_reference
    def maryland_shift(self) -> MarylandShift:
        """The chemical shift tensor in Maryland Notation."""
        iso = self.shift
        omega = self.shift_span
        kappa = self.shift_skew
        return self.MarylandShift(iso, omega, kappa)

    @property
    def herzfeldberger_shielding(self) -> HerzfeldBergerShielding:
        """The shielding tensor in Herzfeld-Berger Notation."""
        iso = self.isotropy
        omega = self.span
        kappa = self.skew
        return self.HerzfeldBergerShielding(iso, omega, kappa)
    @property
    @has_reference
    def herzfeldberger_shift(self) -> HerzfeldBergerShift:
        """The chemical shift tensor in Herzfeld-Berger Notation."""
        iso = self.shift
        omega = self.shift_span
        kappa = self.shift_skew
        return self.HerzfeldBergerShift(iso, omega, kappa)

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

    @property
    def _initialisation_params(self):
        """
        Return the parameters used for initialisation of the tensor.
        """
        return {
            'species': self.species,
            'order': self.order,
            'reference': self.reference,
            'gradient': self.gradient,
            'tag': self.mstag
        }


    def __str__(self):
        """
        Neatly formatted string representation of the tensor and Haeberlen description.
        """
        s =  f"Magnetic Shielding Tensor for {self.species}:\n"
        # 3x3 tensor representation neatly formatted with 5 decimal places
        s += str(np.array2string(self.data, precision=5, separator=",", suppress_small=True)) + "\n"
        s += str(self.haeberlen_shielding)
        return s

    def __eq__(self, other: 'MagneticShielding') -> bool:
        # First, check if it's a MagneticShielding object
        if not isinstance(other, MagneticShielding):
            return False

        # Use parent class equality for tensor comparison
        if not super().__eq__(other):
            return False

        # Compare additional MagneticShielding-specific attributes
        # Species comparison is optional and can be modified as needed
        return self.species == other.species

    def __hash__(self) -> int:
        # Combine parent class hash with species
        return hash((
            super().__hash__(),
            self.species
        ))



def check_quadrupole_active(func):
    """
    Decorator to check if the nucleus is quadrupole active.
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_quadrupole_active:
            raise ValueError(f"Nuclear spin is {self.spin}. This is not quadrupolar active.")
        return func(self, *args, **kwargs)
    return wrapper

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
    default_order = TensorConvention.NQR

    def __init__(self,
                data: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
                species: str,
                order: str=default_order,
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

        # Quadrupole moment in millibarn
        self.quadrupole_moment = quadrupole_moment or nmr_quadrupole(element, iso=isotope_number)

        # gamma in rad/(s*T)
        self.gamma = gamma or nmr_gamma(element, iso=isotope_number)


    @property
    def is_quadrupole_active(self) -> bool:
        '''
        Returns True if the nucleus is quadrupole active, False otherwise.
        '''
        return self.spin > 0.5
    @property
    def eta(self) -> float:
        '''
        Returns the asymmetry parameter of the tensor.
        '''
        return self.asymmetry

    @property
    def zeta(self) -> float:
        '''
        Returns the reduced anisotropy of the tensor.
        '''
        return self.reduced_anisotropy

    @property
    def Vzz(self) -> float:
        '''
        Returns the largest absolute eigenvalue of the tensor (
        the principal component of the EFG tensor).
        This should be in atomic units (a.u.) if read in from e.g. a magres file.
        '''
        # The eigenvalues are already sorted in abs increasing order 
        # (NQR convention or Haberlen conventions (since isotropic value is zero))
        if self.order == self.ORDER_NQR or self.order == self.ORDER_HAEBERLEN:
            return self.eigenvalues[2]
        else:
            # warn user that the eigenvalues are not sorted in NQR order
            warnings.warn("The eigenvalues are not sorted in NQR order. "
                            "Returning Vzz as the largest abs. eigenvalue following NQR convention.")
            # sort them in NQR order
            return _evals_sort([self.eigenvalues], convention=self.ORDER_NQR)[0][2]

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
    @check_quadrupole_active
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

    def get_larmor_frequency(self, Bext: float) -> float:
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


    @check_quadrupole_active
    def get_quadrupolar_perturbation(self, Bext: float) -> float:
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


    @check_quadrupole_active
    def get_MAS_full_max_linewidth(self, Bext: float) -> float:
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

    @property
    def NQR(self):
        '''
        EFGNQR

        Produces an array containing NQR transition frequencies (in Hz) for every atom
        in a system. For non-quadrupole active nuclei, the we return an empty dictionary.
        Unless specified otherwise, the spin and quadrupole moment of the most common
        NMR-active isotope is used.

        For reference: the value returned by this property is defined as

        .. math::

            A = \\frac{V_{zz} Q}{4I(2I - 1)}
            fq = 3A(2m+1)\\sqrt{1 + \\eta^2/3}

        in Hz.
        It is important to keep in mind that therefore this represents a
        *frequency*; the corresponding 'omega' (pulsation) would be the same value
        multiplied by 2*pi. 
        '''
        result = defaultdict(None)

        if self.spin <= 0.5:
            # NQR requires spin > 1/2; spin-0 has no transitions, and
            # spin-1/2 has no quadrupole moment (formula denominator is zero).
            return result

        A = EFG_TO_CHI * self.Vzz * self.quadrupole_moment / (4 * self.spin * (2 * self.spin - 1))
        ms = [m for m in _frange(-self.spin, self.spin + 1, 1) if m >= 0.0][:-1]
        for m in ms:
            key = f'm={m}->{m+1}'
            fq = 3 * A * (2 * m + 1) * np.sqrt(1 + self.eta ** 2 / 3)
            result[key] = fq

        return result


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

    @property
    def _initialisation_params(self):
        """
        Return the parameters used for initialisation of the tensor.
        """
        return {
            'species': self.species,
            'order': self.order,
            'quadrupole_moment': self.quadrupole_moment,
            'gamma': self.gamma
        }

    def __str__(self):
        """
        Neatly formatted string representation of the tensor and essential parameters.
        """
        s =  f"Electric Field Gradient Tensor for {self.species}:\n"
        # 3x3 tensor representation neatly formatted with 5 decimal places
        s += str(np.array2string(self.data, precision=5, separator=",", suppress_small=True)) + "\n"
        s += f"Quadrupole moment: {self.quadrupole_moment} millibarn\n"
        s += f"Nuclear spin: {self.spin}\n"
        if self.Cq == 0:
            s += "This site is not quadrupole active.\n"
        else:
            s += f"Quadrupolar constant: {self.Cq} Hz\n"
            s += f"Quadrupolar product: {self.Pq} Hz\n"
            s += f"Quadrupolar frequency: {self.nuq} Hz\n"
        return s

    def __eq__(self, other: 'ElectricFieldGradient') -> bool:
        # First, check if it's a ElectricFieldGradient object
        if not isinstance(other, ElectricFieldGradient):
            return False

        # Use parent class equality for tensor comparison
        if not super().__eq__(other):
            return False

        # Compare additional MagneticShielding-specific attributes
        # Species comparison is optional and can be modified as needed
        return self.species == other.species

    def __hash__(self) -> int:
        # Combine parent class hash with species
        return hash((
            super().__hash__(),
            self.species
        ))


def contains_nmr_tensors(values):
    """Check if values contain NMRTensor objects at any nesting level"""
    if isinstance(values, (list, np.ndarray)):
        if len(values) > 0:
            if isinstance(values[0], NMRTensor):
                return True
            elif isinstance(values[0], (list, np.ndarray)):
                return contains_nmr_tensors(values[0])
    return False

