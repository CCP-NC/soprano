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

"""Implementation of AtomsProperties that relate to NMR shieldings/shifts"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import warnings
from soprano.properties import AtomsProperty
from soprano.nmr.utils import (
    _haeb_sort,
    _anisotropy,
    _asymmetry,
    _span,
    _skew,
    _evecs_2_quat,
)


def _has_ms_check(f):
    # Decorator to add a check for the magnetic shieldings array
    def decorated_f(s, *args, **kwargs):
        if not (s.has("ms")):
            raise RuntimeError("No magnetic shielding data found for this" " system")
        return f(s, *args, **kwargs)

    return decorated_f


class MSDiagonal(AtomsProperty):

    """
    MSDiagonal

    Produces an array containing eigenvalues and eigenvectors for the
    symmetric part of each magnetic shielding tensor in the system. By default
    saves them as part of the Atoms' arrays as well.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   save_array (bool): if True, save the diagonalised tensors in the
    |                      Atoms object as an array. By default True.

    | Returns:
    |   ms_diag (np.ndarray): list of eigenvalues and eigenvectors

    """

    default_name = "ms_diagonal"
    default_params = {"save_array": True}

    @staticmethod
    @_has_ms_check
    def extract(s, save_array):

        ms_diag = [np.linalg.eigh((ms + ms.T) / 2.0) for ms in s.get_array("ms")]
        ms_evals, ms_evecs = [np.array(a) for a in zip(*ms_diag)]

        if save_array:
            s.set_array(MSDiagonal.default_name + "_evals", ms_evals)
            # Store also the Haeberlen sorted version
            s.set_array(MSDiagonal.default_name + "_evals_hsort", _haeb_sort(ms_evals))
            s.set_array(MSDiagonal.default_name + "_evecs", ms_evecs)

        return np.array([dict(zip(("evals", "evecs"), ms)) for ms in ms_diag])

class MSShielding(AtomsProperty):

    """
    MSShielding

    Produces an array containing the magnetic shielding isotropies in a system
    (ppm).

    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   save_array (bool): if True, save the ms_shielding array in the
    |                      Atoms object as an array. By default True.

    | Returns:
    |   ms_shielding (np.ndarray): list of shieldings

    """

    default_name = "ms_shielding"
    default_params = {"save_array": True}

    @staticmethod
    @_has_ms_check
    def extract(s, save_array):

        ms_shielding = np.trace(s.get_array("ms"), axis1=1, axis2=2) / 3.0

        if save_array:
            # Save the isotropic shieldings 
            s.set_array(MSShielding.default_name, ms_shielding)

        return ms_shielding
class MSShift(AtomsProperty):

    """
    MSShift

    Produces an array containing the chemical shifts (ppm).
    References must be provided for the chemical shifts to be calculated.
    Optionally, the you can also specify the gradient, m
    
    .. math::
        \\delta = \\sigma_{ref} - m\\sigma
    

    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   ref (list/float/dict): reference frequency per element. Must
    |                          be provided.
    |   gradients float/list/dict: usually around -1. Optional.
    |                              Default: -1 for all elements.
    |   save_array (bool): if True, save the ms_shift array in the
    |                      Atoms object as an array. By default True.

    | Returns:
    |   ms_shift (np.ndarray): list of shifts

    """

    default_name = "ms_shift"
    default_params = {"ref": {}, "grad": -1.0, "save_array": True}

    @staticmethod
    @_has_ms_check
    def extract(s, ref, grad, save_array):
        # make sure we have some references set!
        if not ref:
            raise ValueError("No reference provided for chemical shifts")


        # get shieldings
        ms_shieldings = MSShielding.get(s)
        
        symbols = np.array(s.get_chemical_symbols())
        
        # --- REFERENCES --- #
        # array to store the reference for each site
        # defaults to zeros
        references = np.zeros(len(s))

        #-- FLOAT --#
        # if we have a single float, we use it for all sites
        if isinstance(ref, float):
            references[:] = ref
            # if the strucure has more than one type of element, we
            # a single reference is probably not what we want
            # so we raise a warning.
            if len(set(symbols)) > 1:
                warnings.warn('Using the same reference for all elements.\n'
                    'That is probably not what you want.')
        #-- DICT --#
        ## Assuming a format like {'C': 100.0, 'H': 200.0}
        elif isinstance(ref, dict):
            for el, val in ref.items():
                references[symbols == el] = val
            # throw a warning if there are any elements not in the dictionary
            missing_refs = set(symbols) - set(ref.keys())
            if missing_refs:
                warnings.warn(f"Elements {missing_refs} are not in the references dictionary"
                                " and their reference will be set to zero.")
        #-- LIST --#
        ## Assuming a format like [100.0, 100.0, 200.0] with one item per site
        elif isinstance(ref, list):
            if len(ref) != len(s):
                raise ValueError("Reference list must have one item per site/\n"
                "Alternatively, use a dictionary with the element as key and the reference as value")
            references[:] = ref[:]
        else:
            raise ValueError("Reference must be a dictionary element: reference"
                                " or a float or a list of"
                                " floats")
        
        # --- GRADIENTS --- #
        # default gradient of -1 for each site
        gradients = -1 * np.ones(len(s))

        # Do we have gradients set explicitly?
        if grad:
            ## FLOAT ##
            ## the same gradient for all sites
            if isinstance(grad, float):
                gradients[:] = grad
                if len(set(symbols)) > 1:
                    warnings.warn('You are using the same gradient for all '
                    'elements while referencing the chemical shift.')
            ## DICT ##
            ## Assuming a format like {'C': -1, 'H': -0.98}
            elif isinstance(grad, dict):
                for el, val in grad.items():
                    gradients[symbols == el] = val
            ## LIST ##
            ## Assuming a format like [-1, -1, -0.98] with one item per site
            elif isinstance(grad, list):
                if len(grad) != len(s):
                    raise ValueError("Gradient list must be the same length as the system"
                    "Or use a dictionary with the element as key and the gradient as value")
                gradients[:] = grad[:]
            else:
                raise ValueError("Gradients must be a dictionary element: gradient"
                                " or a float or a list of"
                                " floats")
        # if any of the gradients is outside -1.5 to -0.5 we need to
        # raise a warning
        if np.any(gradients < -1.5) or np.any(gradients > -0.5):
            warnings.warn("Gradients are outside the range: -1.5 to -0.5.\n"
                            "That's a surprising value! Please double check the"
                            "gradients.\n"
                            f"You provided:\n {grad}")


        # Convert from shielding to chemical shift
        ms_shifts = references + gradients * ms_shieldings

        if save_array:
            # Save the isotropic shifts
            s.set_array(MSShift.default_name, ms_shifts)


        return ms_shifts
class MSIsotropy(AtomsProperty):

    """
    MSIsotropy

    Produces an array containing the magnetic shielding isotropies in a system
    (ppm).
    If references are provided, the returned values represent the chemical shift.

    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    Refactored into MSShielding and MSShift. This remains here for backwards
    compatibility.

    | Parameters:
    |   ref (float/dict): reference frequency per element. If provided, the chemical shift
    |                will be returned instead of the magnetic shielding.
    |   gradients float/list/dict: usually around -1. 
    |   save_array (bool): if True, save the diagonalised tensors in the
    |                      Atoms object as an array. By default True.

    | Returns:
    |   ms_iso (np.ndarray): list of shieldings/shifts

    """

    default_name = "ms_isotropy"
    default_params = {"ref": {}, "grad": -1.0, "save_array": True}

    @staticmethod
    @_has_ms_check
    def extract(s, ref, grad, save_array):
        
        if ref:
            # the user wants to use the chemical shift
            ms_iso =  MSShift.extract(s, ref, grad, save_array)
        else:
            # the user wants to use the magnetic shielding
            ms_iso =  MSShielding.extract(s, save_array)

        if save_array:
            # Save the isotropic shifts
            s.set_array(MSIsotropy.default_name, ms_iso)
        return ms_iso    

class MSAnisotropy(AtomsProperty):

    """
    MSAnisotropy

    Produces an array containing the magnetic shielding anisotropies in a
    system (ppm).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of anisotropies

    """

    default_name = "ms_anisotropy"
    default_params = {"force_recalc": False}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if not s.has(MSDiagonal.default_name + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + "_evals_hsort")

        return _anisotropy(ms_evals)


class MSReducedAnisotropy(AtomsProperty):

    """
    MSReducedAnisotropy

    Produces an array containing the magnetic shielding reduced anisotropies
    in a system (ppm).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of reduced anisotropies

    """

    default_name = "ms_red_anisotropy"
    default_params = {"force_recalc": False}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if not s.has(MSDiagonal.default_name + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + "_evals_hsort")

        return _anisotropy(ms_evals, reduced=True)


class MSAsymmetry(AtomsProperty):

    """
    MSAsymmetry

    Produces an array containing the magnetic shielding asymmetries
    in a system (adimensional).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of asymmetries

    """

    default_name = "ms_asymmetry"
    default_params = {"force_recalc": False}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if not s.has(MSDiagonal.default_name + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + "_evals_hsort")

        return _asymmetry(ms_evals)


class MSSpan(AtomsProperty):

    """
    MSSpan

    Produces an array containing the magnetic shielding tensor span
    in a system (ppm).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of spans

    """

    default_name = "ms_span"
    default_params = {"force_recalc": False}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if not s.has(MSDiagonal.default_name + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + "_evals_hsort")

        return _span(ms_evals)


class MSSkew(AtomsProperty):

    """
    MSSkew

    Produces an array containing the magnetic shielding tensor skew
    in a system.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of skews

    """

    default_name = "ms_skew"
    default_params = {"force_recalc": False}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if not s.has(MSDiagonal.default_name + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + "_evals_hsort")

        return _skew(ms_evals)


class MSQuaternion(AtomsProperty):

    """
    MSQuaternion

    Produces a list of ase.Quaternion objects expressing the orientation of
    the MS tensors with respect to the cartesian axes.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_quat (np.ndarray): list of quaternions

    """

    default_name = "ms_quats"
    default_params = {"force_recalc": False}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if not s.has(MSDiagonal.default_name + "_evecs") or force_recalc:
            MSDiagonal.get(s)

        ms_evecs = s.get_array(MSDiagonal.default_name + "_evecs")

        return _evecs_2_quat(ms_evecs)
