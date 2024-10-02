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
from typing import List, Tuple, Union

import numpy as np
import scipy.constants as cnst
from ase.quaternions import Quaternion
from scipy.spatial.transform import Rotation

# Left here for backwards compatibility
from soprano.data.nmr import _get_isotope_data, _get_nmr_data, _el_iso

def _split_species(species: str) -> Tuple[int, str]:
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


def _evals_sort(evals, convention="c", return_indices=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Sort a list of eigenvalue triplets by varios conventions"""
    evals = np.array(evals)
    iso = np.average(evals, axis=1)

    if convention in ("i", "d"):
        to_sort = evals
    elif convention == "h":
        to_sort = np.abs(evals - iso[:, None])
    elif convention == "n":
        # Although iso should be zero for NQR (EFG tensors are traceless),
        # we should really just sort by the absolute values of the eigenvalues

        # We can warn the user if the isotropic value is not zero
        if np.any(np.abs(iso) > 1e-10):
            warnings.warn("Isotropic value(s) are not zero but NQR order is requested.\n"
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


def _anisotropy(haeb_evals, reduced=False):
    """Calculate anisotropy given eigenvalues sorted with Haeberlen
    convention"""

    f = 2.0 / 3.0 if reduced else 1.0

    return (haeb_evals[:, 2] - (haeb_evals[:, 0] + haeb_evals[:, 1]) / 2.0) * f


def _asymmetry(haeb_evals):
    """Calculate asymmetry"""

    aniso = _anisotropy(haeb_evals, reduced=True)
    # Fix the anisotropy zero values
    aniso = np.where(aniso == 0, np.inf, aniso)

    return (haeb_evals[:, 1] - haeb_evals[:, 0]) / aniso


def _span(evals):
    """Calculate span
    
    .. math::
        \\Omega = \\sigma_{33} - \\sigma_{11}

    where :math:`\\sigma_{33}` is the largest, and :math:`\\sigma_{11}` is the
    smallest eigenvalue.
    
    """

    return np.amax(evals, axis=-1) - np.amin(evals, axis=-1)


def _skew(evals):
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


def _dip_constant(Rij, gi, gj):
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


def _J_constant(Kij, gi, gj):
    """J coupling constants for pairs ij, with reduced constant Kij and
    gyromagnetic ratios gi and gj"""

    return cnst.h * gi * gj * Kij / (4 * np.pi ** 2) * 1e19

def _matrix_to_euler(R:Union[List[List[float]], np.ndarray],
                     convention: str = "zyz",
                     passive: bool = False
                     ) -> np.ndarray:
    """Convert a rotation matrix to Euler angles (in radians)
    
    We use the scipy Rotation class to do this, but we need to make sure that
    the angles are in the conventional ranges for NMR. This function does that.

    Note that SciPy uses the convention that upper case letters are extrinsic rotations
    and lower case letters are intrinsic rotations (note this distinction
    is not that same as active vs passive. We enforce extrinsic rotations here
    by converting all to upper case.

    We use an explicit keyword argument to specify passive rotations, and
    transpose the rotation matrix if necessary.

    Args:
        R (Union[List[List[float]], np.ndarray]): Rotation matrix. Note that SciPy
                                                    will convert this to a proper rotation matrix
                                                    if it is not already one.
                                                    (i.e. det(R) = 1)
        convention (str, optional): Euler angle convention. Defaults to "zyz".
                                    This will be converted to upper case to enforce
                                    extrinsic rotations.
        passive (bool, optional): Whether the angles are passive rotations. Defaults to False.

    Returns:
        np.ndarray: Euler angles in radians

    """
    convention = convention.upper()
    R = np.array(R)
    # (Note that SciPy handles converting to proper rotation matrices)
    Rot = Rotation.from_matrix(R)
    R = Rot.as_matrix() # just in case it was converted

    # If passive, we need to transpose the matrix
    if passive:
        Rot = Rot.inv()
    # Use scipy to get the euler angles
    euler_angles = Rot.as_euler(convention, degrees=False)
    # Now we need to make sure the angles are in the right range
    # for NMR
    euler_angles = _normalise_euler_angles(euler_angles, passive=passive)

    return euler_angles

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





def _handle_euler_edge_cases(
        euler_angles: np.ndarray,
        eigenvalues: np.ndarray,
        original_tensor: np.ndarray,
        convention: str = "zyz",
        passive: bool = False,
        eps: float=1e-6):
    """Handle edge cases in the Euler angle conventions for degenerate tensors.

    Args:
        euler_angles (np.ndarray): Euler angles in radians
        eigenvalues (np.ndarray): Eigenvalues of the tensor
        original_tensor (np.ndarray): Original symmetric tensor
        convention (str, optional): Euler angle convention. Defaults to "zyz".
        eps (float, optional): Tolerance for degeneracy. Defaults to 1e-6.
    """
    A = original_tensor.copy() # alias for convenience

    # only handle zyz or zxz for now
    if convention.lower() not in ["zyz", "zxz"]:
        warnings.warn(
            f"Edge cases not handled for {convention} convention. "
            "Returning unmodified Euler angles."
        )
        return euler_angles

    # Check for degeneracy
    degeneracy = np.sum(np.abs(eigenvalues - eigenvalues[0]) < eps)
    e1, e2, e3 = eigenvalues
    if degeneracy == 1:
        # No degeneracy, just check that we're in the right range
        euler_angles = _normalise_euler_angles(euler_angles, passive=passive)
        return euler_angles

    # this is the tricky one - doubly degenerate
    elif degeneracy == 2:
        if np.abs(e1 - e2) < eps:
            # We have the unique axis along z
            # we are free to set gamma to zero
            euler_angles[2] = 0

        elif np.abs(e2 - e3) < eps:
            # We have the unique axis along x
            # we are free to set alpha to zero
            euler_angles[0] = 0

            # But now we have to be careful
            if convention.lower() == "zyz":
                gamma = np.arcsin(np.sqrt((A[1,1] - e2) / (e1 - e2) )) # +/- this
                gamma = abs(gamma) # we can choose the sign to be positive
                if abs(gamma - np.pi/2) < eps:
                    # We're free to choose beta to be zero
                    beta = 0.0
                elif (np.abs([A[1,2], A[0,1]]) < eps).all():
                    beta = np.arcsin(np.sqrt( (A[2,2] - e3) / (e1 - e3 + (e2 - e1)*np.sin(gamma)**2) ))
                    beta = abs(beta) # we can choose the sign to be positive
                else:
                    beta = np.arctan2(
                        -1*A[1,2] / (np.sin(gamma) * np.cos(gamma)*(e1 - e2)),
                           A[0,1] / (np.sin(gamma) * np.cos(gamma)*(e1 - e2))
                        )
                # Done with zyz
                euler_angles[1] = beta
                euler_angles[2] = gamma

                if passive:
                    a,b,c = euler_angles
                    euler_angles = _normalise_euler_angles(
                                        np.array([-c,-b,-a]),
                                        passive=passive)
                return euler_angles

            if convention.lower() == "zxz":
                gamma = np.arcsin(np.sqrt((A[0,0] - e1) / (e2 - e1) )) # +/- this
                gamma = abs(gamma) # we can choose the sign to be positive
                if abs(gamma) < eps or abs(gamma - np.pi) < eps:
                    # We're free to choose beta to be zero
                    beta = 0.0
                elif (np.abs([A[2,0], A[1,0]]) < eps).all():
                    beta = np.arcsin(
                                np.sqrt(
                                    (A[1,1] - e2 + (e2 - e1)*np.sin(gamma)**2) /
                                    (e3 - e2 + (e2 - e1)*np.sin(gamma)**2)
                                )
                            )
                    beta = abs(beta) # we can choose the sign to be positive
                else:
                    beta = np.arctan2(
                        A[2,0] / (np.sin(gamma) * np.cos(gamma)*(e1 - e2)),
                        A[1,0] / (np.sin(gamma) * np.cos(gamma)*(e1 - e2))
                        )
                # Done with zxz
                euler_angles[1] = beta
                euler_angles[2] = gamma

                if passive:
                    a,b,c = euler_angles
                    euler_angles = _normalise_euler_angles(
                                        np.array([-c,-b,-a]),
                                        passive=passive)
                return euler_angles
        else:
            # We shouldn't have the unique axis along y for
            # reasonably sorted eigenvalues
            raise ValueError("Unexpected degeneracy when computing Euler angles.\n"
                             "Eigenvalues are ordered: ", eigenvalues)
    elif degeneracy == 3:
        # All eigenvalues are the same
        # We can set all angles to zero
        euler_angles = np.zeros(3)
    else:
        raise ValueError("Degeneracy must be 1, 2, or 3.")
    return euler_angles


def _normalise_euler_angles(
        euler_angles: np.ndarray,
        passive: bool = False,
        eps: float = 1e-6,
        )-> np.ndarray:
    """
    Normalise Euler angles to standard ranges for NMR as 
    defined in: 
    TensorView for MATLAB :cite:p:`Svenningsson2023`

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


def _equivalent_euler(euler_angles: np.ndarray, passive: bool = False):
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


def _tryallanglestest(
        euler_angles: np.ndarray,
        pas1: np.ndarray,
        pasv2: np.ndarray,
        arel1: np.ndarray,
        convention: str,
        eps: float = 1e-3
        ) -> np.ndarray:
    """
    For relative Euler angles from tensor A to B, if B is axially symmetric,
    we need to try all equivalent Euler angles of the symmetric tensor to
    find the conventional one. 

    Go through the 4 equivalent passive angles since we're using 
    the trick of calculating the A relative angles in the frame of B 
    by calculating the B relative angles in the frame of A with the *passive* convention.

    Credit: function adapted from TensorView for MATLAB: :cite:p:`Svenningsson2023`
    """

    # make copy of the input angles
    euler_angles_out = euler_angles.copy()
    rrel_check = Rotation.from_euler(convention.upper(), euler_angles).as_matrix()
    mcheck = np.round(np.dot(np.dot(rrel_check, pas1), np.linalg.inv(rrel_check)), 14)
    # Define the ways in which the angles should be updated
    alpha, beta, gamma = euler_angles
    updates = [
        lambda: (alpha + np.pi, beta, gamma),
        lambda: (2*np.pi - alpha, np.pi - beta, gamma + np.pi),
        lambda: (np.pi-alpha, np.pi-beta, gamma + np.pi),
    ]
    # if pasv2[0] == pasv2[1]:
    if np.allclose(pasv2[0], pasv2[1], atol=eps):
        # Iterate over the updates, updating only if the angles don't match
        for update in updates:
            if not np.all(np.isclose(arel1, mcheck, atol=eps)):
                alpha_out, beta_out, gamma_out = update()
                euler_angles_out, mcheck = _compute_rotation([alpha_out, beta_out, gamma_out], arel1, pas1, convention, 1)
            else:
                break # If the angles match, we're done

        # If the last condition is still true, print a message
        if not np.all(np.isclose(arel1, mcheck, atol=eps)):
            raise 'Failed isequal check at (_tryallanglestest) please contact the developers for to help resolve this issue.'

    elif np.allclose(pasv2[0], pasv2[1], atol=eps):
        # Iterate over the updates, updating only if the angles don't match
        for update in updates:
            if not np.all(np.isclose(arel1, mcheck, atol=eps)):
                alpha_out, beta_out, gamma_out = update()
                euler_angles_out, mcheck = _compute_rotation([alpha_out, beta_out, gamma_out], arel1, pas1, convention, 2)
            else:
                break

    return euler_angles_out



def _compute_rotation(euler_angles: np.ndarray,
                      arel1: np.ndarray,
                      pas1: np.ndarray,
                      convention: str,
                      rotation_type: int
                      )-> Tuple[np.ndarray, np.ndarray]:
    rrel_check = Rotation.from_euler(convention.upper(), euler_angles).as_matrix()
    mcheck = np.round(np.dot(np.dot(rrel_check, pas1), np.linalg.inv(rrel_check)), 14)

    if rotation_type == 1:
        component1 = arel1[2, 0] + arel1[2, 1]*mcheck[2, 1]/mcheck[2, 0]
        component2 = mcheck[2, 0] + mcheck[2, 1]*mcheck[2, 1]/mcheck[2, 0]
        component3 = arel1[2, 1] - arel1[2, 0]*mcheck[2, 1]/mcheck[2, 0]
        component4 = mcheck[2, 0] + mcheck[2, 1]*mcheck[2, 1]/mcheck[2, 0]
        euler_convention = "ZYZ"
    elif rotation_type == 2:
        component1 = arel1[2, 0] + arel1[1, 0]*mcheck[1, 0]/mcheck[2, 0]
        component2 = mcheck[2, 0] + mcheck[1, 0]*mcheck[1, 0]/mcheck[2, 0]
        component3 = arel1[2, 0] - arel1[1, 0]*mcheck[2, 0]/mcheck[1, 0]
        component4 = mcheck[1, 0] + mcheck[2, 0]*mcheck[2, 0]/mcheck[1, 0]
        euler_convention = "ZXZ"

    symrotang_check = np.arctan2(component3/component4, component1/component2)
    symrot_check = Rotation.from_euler(euler_convention, [0, 0, symrotang_check]).as_matrix()
    mcheck = np.dot(np.dot(symrot_check, mcheck), np.linalg.inv(symrot_check))
    return euler_angles, mcheck
def _frange(a, b, x):
    while a < b:
        yield a
        a += x
