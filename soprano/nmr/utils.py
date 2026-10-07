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

"""Utility functions for NMR-related properties  
"""

import re
import warnings
from typing import Union, Optional

import numpy as np
import scipy.constants as cnst
from ase.quaternions import Quaternion
from scipy.spatial.transform import Rotation

# Left here for backwards compatibility
from soprano.data.nmr import _get_isotope_data, _get_nmr_data, _el_iso
from soprano.utils import FloatOrArray

def _split_species(species: str) -> tuple[int, str]:
        """
        Validate the species string and extract the isotope number and element.

        Args:
            species (str): The species string to validate. e.g. '13C' or '1H'.

        Returns:
            tuple: A tuple containing the isotope number (int), and element symbol (str, such as 'C' or 'H').
        """
        match = re.match(r"^(\d+)([A-Za-z]+)$", species)
        if not match:
            raise ValueError(f"Species must be an isotope symbol such as '13C' or '1H'. Got '{species}' instead.")
        isotope_number, element = match.groups()
        return int(isotope_number), element


def _evals_sort(evals, convention="i", return_indices=False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sort a list of eigenvalue triplets by various conventions

    The sorting conventions are:
    - i: increasing order
    - d: decreasing order
    - h: Haeberlen order
    - n: NQR order (absolute values)
    
    Args:
        evals (np.ndarray): Array of eigenvalue triplets to sort.
        convention (str): The sorting convention to use ('i', 'd', 'h', 'n').
        return_indices (bool): Whether to return the sorting indices.

        Returns:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]: Sorted eigenvalues, and optionally the sorting indices.
    """
    evals = np.array(evals)
    # Handle special case of all eigenvalues being the same (within tolerance)
    if np.all(np.abs(evals - evals[:, 0:1]) < 1e-16):
        # Return indices matching the input shape (0,1,2 for each eigenvalue)
        if not return_indices:
            return evals
        else:
            indices = np.tile(np.arange(evals.shape[1]), (evals.shape[0], 1))
            return evals, indices
        
    iso = np.average(evals, axis=1)

    if convention in ("i", "d"):
        to_sort = evals
    elif convention == "h":
        to_sort = np.abs(evals - iso[:, None])
    elif convention == "n":
        # Although iso should be zero for NQR (EFG tensors are traceless),
        # we should really just sort by the absolute values of the eigenvalues

        # We can warn the user if the isotropic value is not zero
        if np.any(np.abs(iso) > 1e-6):
            warnings.warn(f"Isotropic value(s) are not zero ({iso}) but NQR order is requested.\n"
                "If you're dealing with an EFG tensor, "
                "then check it carefully since these should be traceless.\n"
                "Sorting by absolute values.\n"
                )
        to_sort = np.abs(evals)
    else:
        raise ValueError("Invalid convention. Must be one of 'i', 'd', 'h', or 'n'. Received: ", convention)

    sort_i = np.argsort(to_sort, axis=1)
    if convention == "d":
        sort_i = sort_i[:, ::-1]
    elif convention == "h":
        sort_i[:, 0], sort_i[:, 1] = sort_i[:, 1], sort_i[:, 0].copy()
    sorted_evals = evals[np.arange(evals.shape[0])[:, None], sort_i]
    if not return_indices:
        return sorted_evals
    else:
        return sorted_evals, sort_i


def _haeb_sort(evals, return_indices=False):
    return _evals_sort(evals, "h", return_indices)


def _anisotropy(haeb_evals, reduced=False) -> FloatOrArray:
    """Calculate anisotropy given eigenvalues sorted with Haeberlen
    convention"""

    f = 2.0 / 3.0 if reduced else 1.0

    return (haeb_evals[:, 2] - (haeb_evals[:, 0] + haeb_evals[:, 1]) / 2.0) * f


def _asymmetry(haeb_evals) -> FloatOrArray:
    """Calculate asymmetry"""

    aniso = _anisotropy(haeb_evals, reduced=True)
    # Fix the anisotropy zero values
    aniso = np.where(aniso == 0, np.inf, aniso)

    return (haeb_evals[:, 1] - haeb_evals[:, 0]) / aniso


def _span(evals) -> FloatOrArray:
    """Calculate span
    
    .. math::
        \\Omega = \\sigma_{33} - \\sigma_{11}

    where :math:`\\sigma_{33}` is the largest, and :math:`\\sigma_{11}` is the
    smallest eigenvalue.
    
    """

    return np.amax(evals, axis=-1) - np.amin(evals, axis=-1)


def _skew(evals) -> FloatOrArray:
    """Calculate skew

    .. math::
        \\kappa = 3 (\\sigma_{iso} - \\sigma_{22}) / \\Omega

    where :math:`\\Omega` is the span of the tensor.

    Note that for chemical shift tensors (:math:`\\delta`), the sign is reversed.
    """

    span = _span(evals)
    span = np.where(span == 0, np.inf, span)
    return 3 * (np.average(evals, axis=1) - np.median(evals, axis=1)) / span


def _evecs_2_quat(evecs):
    """Convert a set of eigenvectors to a Quaternion expressing the
    rotation of the tensor's PAS with respect to the Cartesian axes"""

    # First, guarantee that the eigenvectors express *proper* rotations
    evecs = np.array(evecs) * np.linalg.det(evecs)[:, None, None]

    # Then get the quaternions
    return [Quaternion.from_matrix(evs.T) for evs in evecs]


def _dip_constant(Rij, gi, gj) -> FloatOrArray:
    """Dipolar constants for pairs ij, with distances Rij and gyromagnetic
    ratios gi and gj"""

    return -(cnst.mu_0 * cnst.hbar * gi * gj / (8 * np.pi ** 2 * Rij ** 3))


def _dip_tensor(d, r, rotation_axis=None):
    """Full dipolar tensor given a constant and a connecting vector"""

    r = np.array(r)
    r /= np.linalg.norm(r)

    if rotation_axis is None:
        D = d * (3 * r[:, None] * r[None, :] - np.eye(3))
    else:
        a = np.array(rotation_axis)
        a /= np.linalg.norm(a)
        vp2 = np.dot(r, a) ** 2
        D = 0.5 * d * (3 * vp2 - 1) * (3 * a[:, None] * a[None, :] - np.eye(3))

    return D


def _J_constant(Kij, gi, gj) -> FloatOrArray:
    """J coupling constants for pairs ij, with reduced constant Kij and
    gyromagnetic ratios gi and gj.

    Returns J in Hz:  J = h · γ_i · γ_j · K_ij / (4π²).

    The trailing ``1e19`` converts the reduced coupling constant K_ij from its
    tabulated/stored unit of ``10^19 T² J⁻¹`` (= ``10^19 kg⁻¹ m⁻² s²``) to SI
    (T² J⁻¹).  With γ in rad s⁻¹ T⁻¹ and h in J s the product is dimensionless
    of s⁻¹, i.e. Hz.
    """

    return cnst.h * gi * gj * Kij / (4 * np.pi ** 2) * 1e19

def _matrix_to_euler(R: Union[list[list[float]], np.ndarray],
                     convention: str = "zyz",
                     passive: bool = False
                     ) -> np.ndarray:
    """Convert an NMR tensor eigenframe matrix to NMR-canonical Euler angles (in radians).

    **NMR-specific function.** This function is designed for use with NMR tensor
    eigenframes, where eigenvector signs are physically arbitrary. It applies
    :func:`_normalise_euler_angles` after extracting angles from scipy, which
    restricts β to [0, π/2] by exploiting the sign freedom of eigenvectors.
    This transformation does *not* preserve a general rotation matrix: for
    arbitrary rotations with β > π/2 the returned angles reconstruct a different
    matrix. Do not use this function for general rotation matrices.

    SciPy's upper-case convention strings denote intrinsic rotations; all
    input convention strings are converted to upper case so the behaviour is
    always intrinsic.

    Args:
        R: Rotation matrix (3×3), typically the eigenvector matrix of an NMR
           tensor. Array-like inputs are accepted.
           If ``det(R) < 0``, the last column is negated before proceeding.
           For eigenvector matrices this is valid because eigenvector signs are
           arbitrary; for general improper rotations it is not the nearest
           proper rotation in a metric sense.
        convention: Euler-angle convention, e.g. ``"zyz"`` or ``"zxz"``.
                    Converted to upper case internally.  Defaults to ``"zyz"``.
        passive: If ``True``, return the *passive* (alias) angles, i.e. those
                 describing the rotation of the coordinate frame rather than
                 the object.  Defaults to ``False``.

    Returns:
        np.ndarray: NMR-canonical Euler angles in radians, shape ``(3,)``.
    """
    # Rotation.from_matrix silently approximates non-special-orthogonal matrices,
    # but the approximation for improper (det < 0) matrices discards the reflection
    # and returns an incorrect result. Correct explicitly before passing to SciPy.
    # For NMR eigenvector matrices this is always valid since eigenvector signs
    # are arbitrary.
    if np.linalg.det(np.asarray(R)) < 0:
        R = np.array(R, dtype=float)
        R[:, -1] *= -1

    Rot = Rotation.from_matrix(R)

    if passive:
        Rot = Rot.inv()

    euler_angles = Rot.as_euler(convention.upper(), degrees=False)
    return _normalise_euler_angles(euler_angles, passive=passive)

def _test_euler_rotation(
        euler_angles: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvecs: np.ndarray,
        convention: str = "zyz",
        passive: bool = False,
        eps: float=1e-6)->bool:
    """
    Test that the Euler angles correctly rotate the tensor.

    We compare the tensor rotated by the Euler angles to that you get by
    rotating the tensor with the rotation matrix corresponding to the Euler
    angles. i.e. 

    PAS = np.diag(eigenvalues)
    R = Rotation.from_euler(convention, euler_angles).as_matrix()
    A_rot = np.dot(R, np.dot(PAS, R.T))

    B_rot = np.dot(eigenvecs, np.dot(PAS, eigenvecs.T))

    Args:
        euler_angles (np.ndarray): Euler angles in radians
        eigenvalues (np.ndarray): Eigenvalues of the tensor
        eigenvecs (np.ndarray): Eigenvectors of the tensor
        convention (str, optional): Euler angle convention. Defaults to "zyz".
        passive (bool, optional): Whether the angles are passive rotations. Defaults to False.
        eps (float, optional): Tolerance for degeneracy. Defaults to 1e-6.

    Returns:
        bool: True if the Euler angles correctly rotate the tensor. False otherwise.
    """

    PAS = np.diag(eigenvalues)
    Rot = Rotation.from_euler(convention.upper(), euler_angles)
    if passive:
        Rot = Rot.inv()
    R = Rot.as_matrix()
    A_rot = np.dot(R, np.dot(PAS, np.linalg.inv(R)))
    # B_rot should just be the symmetric tensor
    B_rot = np.linalg.multi_dot([eigenvecs, PAS, np.linalg.inv(eigenvecs)])
    return np.allclose(A_rot, B_rot, atol=eps)





def _degeneracy_pattern(eigenvalues, eps: float = 1e-6):
    """Classify the eigenvalue pattern of a tensor in its stored (sorted) order.

    Returns ``(degeneracy, unique_axis)``: ``degeneracy`` is 1 (triaxial), 2 (axial) or
    3 (spherical); ``unique_axis`` is the index (0, 1, 2 for x, y, z) of the eigenvalue
    that differs from the other two when ``degeneracy == 2``, else ``None``.
    Every pair is compared, so ``[a, b, b]`` is recognised (a unique axis along x), which
    comparing everything to ``eigenvalues[0]`` would miss.
    """
    e = np.asarray(eigenvalues, dtype=float)
    close = {
        2: abs(e[0] - e[1]) < eps,  # x ~ y: unique axis along z
        0: abs(e[1] - e[2]) < eps,  # y ~ z: unique axis along x
        1: abs(e[0] - e[2]) < eps,  # x ~ z: unique axis along y
    }
    n_close = sum(close.values())
    if n_close == 0:
        return 1, None
    if n_close == 1:
        return 2, next(axis for axis, is_close in close.items() if is_close)
    return 3, None


def _alpha_zero_gauge(E: np.ndarray, convention: str = "zyz") -> np.ndarray:
    """Pick the PAS frame of a tensor with a unique x axis that makes the active alpha zero.

    A tensor with eigenvalues ``[a, b, b]`` is symmetric under any rotation about its PAS x
    axis, so ``E -> E Rx(phi)`` describes the same tensor. TensorView fixes this freedom by
    asking for ``alpha = 0`` in the active ZYZ/ZXZ angles (Eq. 70 of Svenningsson & Mueller,
    Solid State Nucl. Magn. Reson. 123 (2023) 101849). Here the same ``phi`` is found directly
    from the matrix: ``alpha`` is the azimuth of the third column of ``E``, so ``phi`` rotates
    that column into the xz plane (ZYZ) or the yz plane (ZXZ), on the side with the sign that
    gives ``alpha = 0`` rather than ``pi``. There are no divisions by ``sin(gamma) cos(gamma)``
    or square roots, so there are no singular orientations.
    """
    E = np.array(E, dtype=float)
    v1, v2 = E[:, 1], E[:, 2]  # third column of E Rx(phi) is cos(phi) v2 - sin(phi) v1
    # (u, w): component that must vanish (ZYZ: y, ZXZ: x) and the one whose sign is fixed
    u, w = (1, 0) if convention.lower() == "zyz" else (0, 1)
    if np.hypot(v1[u], v2[u]) > 1e-12:
        phi = np.arctan2(v2[u], v1[u])
    else:  # the component vanishes for every phi: just orient the other one
        phi = np.arctan2(-v1[w], v2[w])
    col = np.cos(phi) * v2 - np.sin(phi) * v1
    # ZYZ: alpha = atan2(R12, R02) = 0 needs R02 >= 0; ZXZ: alpha = atan2(R02, -R12) = 0 needs R12 <= 0
    wrong_side = col[w] < -1e-12 if convention.lower() == "zyz" else col[w] > 1e-12
    if wrong_side:
        phi += np.pi
    c, s = np.cos(phi), np.sin(phi)
    return E @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _alpha_zero_euler_angles(R: np.ndarray, convention: str = "zyz", eps: float = 1e-9) -> np.ndarray:
    """Active ZYZ/ZXZ angles of ``R`` with ``alpha == 0`` (``R`` from :func:`_alpha_zero_gauge`).

    No range normalisation is applied here: beta is in [0, pi] and gamma in [0, 2 pi).
    ``NMRTensor.euler_angles`` normalises afterwards, as TensorView does, which turns
    ``alpha = 0`` into ``pi`` when beta >= pi/2. At gimbal lock (beta = 0 or pi) only a combination of
    alpha and gamma is defined, and it is all put into gamma.
    """
    zyz = convention.lower() == "zyz"
    sin_beta = np.hypot(R[0, 2], R[1, 2]) if zyz else np.hypot(R[2, 0], R[2, 1])
    beta = np.arctan2(sin_beta, R[2, 2])
    if sin_beta < eps:
        if R[2, 2] > 0:
            gamma = np.arctan2(R[1, 0], R[0, 0])
        else:
            gamma = np.arctan2(R[1, 0], -R[0, 0]) if zyz else np.arctan2(-R[1, 0], R[0, 0])
    else:
        gamma = np.arctan2(R[2, 1], -R[2, 0]) if zyz else np.arctan2(R[2, 0], R[2, 1])
    return np.array([0.0, beta, gamma % (2 * np.pi)])


def _handle_euler_edge_cases(
        euler_angles: np.ndarray,
        eigenvalues: np.ndarray,
        convention: str = "zyz",
        passive: bool = False,
        eps: float=1e-6,
        angles_are_passive: bool = False):
    """Handle edge cases in the Euler angle conventions for degenerate tensors.

    Handled here: spherical tensors (all angles zero) and tensors with the unique axis
    along z (the free angle is set to zero). Tensors with the unique axis along x need the
    PAS frame to be chosen *before* the angles are extracted
    (:func:`_alpha_zero_gauge`, done by ``NMRTensor.euler_angles``); if they arrive here
    their angles are just normalised.

    Args:
        euler_angles (np.ndarray): Euler angles in radians
        eigenvalues (np.ndarray): Eigenvalues of the tensor
        convention (str, optional): Euler angle convention. Defaults to "zyz".
        passive (bool, optional): Whether the angles are passive. Defaults to False.
        eps (float, optional): Tolerance for degeneracy. Defaults to 1e-6.
        angles_are_passive (bool, optional): Whether ``euler_angles`` are passive angles on
            entry. Only affects which angle is the free one for a tensor with its unique axis
            along z: gamma if the angles are active, alpha if they are passive. Defaults to False.
    """
    # only handle zyz or zxz for now
    if convention.lower() not in ["zyz", "zxz"]:
        warnings.warn(
            f"Edge cases not handled for {convention} convention. "
            "Returning unmodified Euler angles."
        )
        return euler_angles

    degeneracy, unique_axis = _degeneracy_pattern(eigenvalues, eps)
    if degeneracy == 1:
        # No degeneracy, just check that we're in the right range
        return _normalise_euler_angles(euler_angles, passive=passive)
    if degeneracy == 3:
        # All eigenvalues are the same: we can set all angles to zero
        return np.zeros(3)
    if unique_axis == 2:
        # Unique axis along z: the last rotation (active) or first (passive) is about the
        # unique axis and is free, so set it to zero.
        euler_angles[0 if angles_are_passive else 2] = 0
        return euler_angles
    if unique_axis == 0:
        return _normalise_euler_angles(euler_angles, passive=passive)
    # We shouldn't have the unique axis along y for reasonably sorted eigenvalues
    raise ValueError("Unexpected degeneracy when computing Euler angles.\n"
                     "Eigenvalues are ordered: ", eigenvalues)


def _normalise_euler_angles(
        euler_angles: np.ndarray,
        passive: bool = False,
        eps: float = 1e-6,
        )-> np.ndarray:
    """
    Normalise Euler angles to the orientation-equivalent set with beta in [0, pi/2).

    The four equivalent sets of a tensor (a half turn about each PAS axis) are those of
    TensorView for MATLAB :cite:p:`Svenningsson2023`, Eqs. 16-23. The paper allows
    ``0 <= beta <= pi``; the member chosen here is the one TensorView's code reports
    (``MFtoEuler.m`` and ``RelativeAngles.m``): beta in [0, pi/2), then gamma in [0, pi) for
    active angles or alpha in [0, pi) for passive ones. The result always describes the same
    tensor.

    Args:
        euler_angles (np.ndarray): Euler angles in radians
        passive (bool, optional): Whether the angles are passive rotations. Defaults to False.
        eps (float, optional): Tolerance for degeneracy. Defaults to 1e-6.



    .. footbibliography::
    
    """
    alpha, beta, gamma = euler_angles

    # wrap any negative angles
    alpha = alpha % (2 * np.pi)
    beta  = beta  % (2 * np.pi)
    gamma = gamma % (2 * np.pi)


    if passive:
        # Note this assumes we've already transposed the R matrix
        # beta > pi is reachable when called from _handle_euler_edge_cases
        # with manually constructed angles (e.g. [-c, -b, -a] where b may be
        # negative, landing in (pi, 2pi] after the modulo above).
        if beta > np.pi:
            beta = 2 * np.pi - beta
            gamma = gamma - np.pi
            gamma  = gamma % (2 * np.pi)

        if beta >= np.pi/2 - eps:
            alpha = np.pi - alpha
            alpha  = alpha % (2 * np.pi)
            beta = np.pi - beta
            beta = beta % (2 * np.pi)
            gamma = np.pi + gamma
            gamma = gamma % (2 * np.pi)

        if alpha >= np.pi - eps:
            alpha = alpha - np.pi
    else:
        # beta > pi is reachable when called from _handle_euler_edge_cases
        # with manually constructed angles.
        if beta > np.pi:
            beta = 2 * np.pi - beta
            alpha = alpha - np.pi
            alpha  = alpha % (2 * np.pi)

        if beta >= np.pi/2 - eps:
            alpha = alpha + np.pi
            alpha  = alpha % (2 * np.pi)
            beta = np.pi - beta
            beta = beta % (2 * np.pi)
            gamma = np.pi - gamma
            gamma = gamma % (2 * np.pi)

        if gamma >= np.pi - eps:
            gamma = gamma - np.pi

    return np.array([alpha, beta, gamma])


def _equivalent_euler(euler_angles: np.ndarray, passive: bool = False) -> np.ndarray:
    """
    Find the equivalent Euler angles for a given set of Euler angles.

    This set should be correct for NMR tensors, according to 
    TensorView for MATLAB: :cite:p:`Svenningsson2023`
    """

    equiv_angles = np.zeros((4, 3))

    alpha, beta, gamma = euler_angles

    # set the first row of the array to the original Euler angles
    equiv_angles[0] = [alpha, beta, gamma]
    # the order of these doesn't matter, but has been chosen to match
    # that in the TensorView for MATLAB code
    # (which is different to that in the corresponding paper)
    if passive:
        equiv_angles[1,:] = [np.pi + alpha, beta, gamma]
        equiv_angles[2,:] = [np.pi - alpha, np.pi - beta, np.pi + gamma]
        equiv_angles[3,:] = [2*np.pi - alpha, np.pi - beta,np.pi + gamma]
    else:
        equiv_angles[1,:] = [alpha, beta, np.pi + gamma]
        equiv_angles[2,:] = [np.pi + alpha, np.pi - beta, np.pi - gamma]
        equiv_angles[3,:] = [np.pi + alpha, np.pi - beta, 2*np.pi - gamma]

    # now a few checks
    # wrap any negative angles
    equiv_angles[equiv_angles < 0] = equiv_angles[equiv_angles < 0] % (2*np.pi)
    # wrap any values > 2pi
    equiv_angles[equiv_angles >= 2*np.pi] = equiv_angles[equiv_angles >= 2*np.pi] % (2*np.pi)

    return equiv_angles

def _equivalent_relative_euler(euler_angles: np.ndarray, passive: bool = False) -> np.ndarray:
    """
    Returns a list of 16 equivalent relative Euler angles for a given set of Euler angles that corresponds to the relative orientation of two NMR tensors.
    See TensorView for MATLAB: :cite:p:`Svenningsson2023`
    """
    equiv_angles = np.zeros((16, 3))
    alpha, beta, gamma = euler_angles

    if passive:
        # --- Passive ZYZ or ZXZ --- #
        equiv_angles[0,:] = [alpha, beta, gamma]
        equiv_angles[1,:] = [2*np.pi - alpha, np.pi - beta, np.pi + gamma]
        equiv_angles[2,:] = [np.pi - alpha, np.pi - beta, np.pi + gamma]
        equiv_angles[3,:] = [np.pi + alpha, beta, gamma]
        equiv_angles[4,:] = [np.pi + alpha, np.pi - beta, 2*np.pi - gamma]
        equiv_angles[5,:] = [np.pi - alpha, beta, np.pi - gamma]
        equiv_angles[6,:] = [2*np.pi - alpha, beta, np.pi - gamma]
        equiv_angles[7,:] = [alpha, np.pi - beta, 2*np.pi - gamma]
        equiv_angles[8,:] = [np.pi + alpha, np.pi - beta, np.pi - gamma]
        equiv_angles[9,:] = [np.pi - alpha, beta, 2*np.pi - gamma]
        equiv_angles[10,:] = [2*np.pi - alpha, beta,2*np.pi - gamma]
        equiv_angles[11,:] = [alpha, np.pi - beta, np.pi - gamma]
        equiv_angles[12,:] = [alpha, beta, np.pi + gamma]
        equiv_angles[13,:] = [2*np.pi - alpha, np.pi - beta, gamma]
        equiv_angles[14,:] = [np.pi - alpha, np.pi - beta, gamma]
        equiv_angles[15,:] = [np.pi + alpha, beta, np.pi + gamma]
    else:
        # --- Active ZYZ or ZXZ--- #
        equiv_angles[0,:] = [alpha, beta, gamma]
        equiv_angles[1,:] = [alpha + np.pi, np.pi - beta, 2*np.pi - gamma]
        equiv_angles[2,:] = [alpha + np.pi, np.pi - beta, np.pi - gamma]
        equiv_angles[3,:] = [alpha, beta, np.pi + gamma]
        equiv_angles[4,:] = [2*np.pi - alpha, np.pi - beta, np.pi + gamma]
        equiv_angles[5,:] = [np.pi - alpha, beta, np.pi - gamma]
        equiv_angles[6,:] = [np.pi - alpha, beta, 2*np.pi - gamma]
        equiv_angles[7,:] = [2*np.pi - alpha, np.pi - beta, gamma]
        equiv_angles[8,:] = [np.pi - alpha, np.pi - beta, np.pi + gamma]
        equiv_angles[9,:] = [2*np.pi - alpha, beta, np.pi - gamma]
        equiv_angles[10,:] = [2*np.pi - alpha, beta,2*np.pi - gamma]
        equiv_angles[11,:] = [np.pi - alpha, np.pi - beta, gamma]
        equiv_angles[12,:] = [alpha + np.pi, beta, gamma]
        equiv_angles[13,:] = [alpha, np.pi - beta, 2*np.pi - gamma]
        equiv_angles[14,:] = [alpha, np.pi - beta, np.pi - gamma]
        equiv_angles[15,:] = [alpha + np.pi, beta, np.pi + gamma]



    # wrap any negative angles
    equiv_angles[equiv_angles < 0] = equiv_angles[equiv_angles < 0] % (2*np.pi)
    # wrap any values >= 2pi
    equiv_angles[equiv_angles >= 2*np.pi] = equiv_angles[equiv_angles >= 2*np.pi] % (2*np.pi)


    return equiv_angles


def compute_b0_from_spectrometer_freq(spectrometer_freq_mhz: float) -> float:
    """Compute B0 field (Tesla) from the ¹H spectrometer frequency in MHz.

    Parameters
    ----------
    spectrometer_freq_mhz : float
        Proton (¹H) Larmor frequency in MHz as displayed on the spectrometer
        (e.g. 600 for a "600 MHz" instrument).

    Returns
    -------
    float
        Magnetic field strength in Tesla.
    """
    from soprano.data.nmr import nmr_gamma

    gamma_H = abs(nmr_gamma("H"))  # rad s⁻¹ T⁻¹
    return spectrometer_freq_mhz * 1e6 * 2 * np.pi / gamma_H


def compute_larmor_frequency(element: str, b0_tesla: float, isotope: Optional[int] = None) -> float:
    """Compute the Larmor frequency (in MHz) for a nucleus at a given field.

    Parameters
    ----------
    element : str
        Element symbol (e.g. ``'H'``, ``'C'``).
    b0_tesla : float
        Magnetic field strength in Tesla.
    isotope : int, optional
        Mass number.  Defaults to the most abundant NMR-active isotope.

    Returns
    -------
    float
        Larmor frequency in MHz.

    Raises
    ------
    ValueError
        If the element has no known gyromagnetic ratio data.
    """
    from soprano.data.nmr import nmr_gamma

    try:
        gamma = nmr_gamma(element, iso=isotope)  # rad s⁻¹ T⁻¹
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Cannot compute Larmor frequency for element '{element}': "
            f"no gyromagnetic ratio data available."
        ) from exc
    return abs(gamma * b0_tesla) / (2 * np.pi * 1e6)


def _frange(a, b, x):
    while a < b:
        yield a
        a += x

def _ensure_tensor_format(tensors):
    """
    Ensure tensors are in 3x3 format, converting from flattened 1x9 if necessary.
    
    Args:
        tensors (np.ndarray): Tensor array, either (N, 3, 3) or (N, 9)
        
    Returns:
        np.ndarray: Tensors in (N, 3, 3) format
    """
    tensors = np.array(tensors)
    
    if tensors.ndim == 2 and tensors.shape[1] == 9:
        return tensors.reshape(-1, 3, 3)
    elif tensors.ndim == 3 and tensors.shape[1:] == (3, 3):
        return tensors
    else:
        raise ValueError(f"Invalid tensor format: expectd (N, 9) or (N, 3, 3), got {tensors.shape}")

def _get_tensor_array(s, tensor_tag):
    """
    Get tensor array from structure, ensuring 3x3 format.
    
    Args:
        s (Atoms): The structure
        tensor_tag (str): Name of the tensor array
        
    Returns:
        np.ndarray: Tensor array in (N, 3, 3) format
        
    Raises:
        RuntimeError: If tensor array not found
    """
    if not s.has(tensor_tag):
        raise RuntimeError(f"Structure does not have tensor array '{tensor_tag}'")
    
    tensors = s.get_array(tensor_tag)
    return _ensure_tensor_format(tensors)


    
    

def _gradients_to_list(gradients: Union[float, dict[str, float], list[float]], elements: list[str]) -> list[float]:
    """
    Convert gradients input to a list with one value per element in elements.

    Parameters:
    gradients (Union[float, dict[str, float], list[float]]): The gradients input, which can be a float, a dictionary, or a list.
    elements (list[str]): The list of elements.

    Returns:
    Optional[list[Optional[float]]]: A list of gradients corresponding to the elements, or None if gradients is None.

    Raises:
    ValueError: If the gradients input is not a float, dictionary, or list, or if the length of the gradients list does not match the length of elements.
    """
    if gradients is None:
        return None
    DEFAULT_GRADIENT = -1.0
    if isinstance(gradients, float):
        return [gradients] * len(elements)
    elif isinstance(gradients, dict):
        return [gradients.get(el, DEFAULT_GRADIENT) for el in elements]
    elif isinstance(gradients, list):
        if len(gradients) == len(elements):
            return gradients
        else:
            raise ValueError(
                "Mismatch between number of sites and number of gradients given."
            )
    else:
        raise ValueError(
            "gradients must be either None, a float, a dictionary, or a list of gradients"
        )

def _references_to_list(references: Union[None, dict[str, float], list[float]], elements: list[str]) -> list[Union[float, None]]:
    """
    Convert references input to a list with one value per element in elements.

    Parameters:
    references (Union[None, dict[str, float], list[float]]): The references input, which can be None, a dictionary, or a list.
    elements (list[str]): The list of elements.

    Returns:
    Optional[list[Optional[float]]]: A list of references corresponding to the elements, or None if references is None.

    Raises:
    ValueError: If the references input is not None, a dictionary, or a list, or if the length of the references list does not match the length of elements.
    """
    if references is None:
        return [None] * len(elements)
    elif isinstance(references, dict):
        if not all(el in references for el in elements):
            raise ValueError(
                "Mismatch between sites and references given."
                f"Missing elements: {set(elements) - set(references.keys())}"
            )
        return [references[el] for el in elements]
    elif isinstance(references, list):
        if len(references) == len(elements):
            return references # type: ignore
        else:
            raise ValueError(
                "Mismatch between number of sites and number of references given."
            )
    else:
        raise ValueError(
            "references must be either None, a dictionary, or a list of references"
        )