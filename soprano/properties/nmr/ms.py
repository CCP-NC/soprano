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


import warnings

import numpy as np

from soprano.nmr import MagneticShielding
from soprano.nmr.utils import (
    _anisotropy,
    _asymmetry,
    _haeb_sort,
    _skew,
    _span,
    _get_tensor_array,
)
from soprano.properties import AtomsProperty

DEFAULT_MS_TAG = "ms"


def _has_ms_check(f):
    # Decorator to add a check for the magnetic shieldings array
    def decorated_f(s, *args, **kwargs):
        tag = kwargs.get('tag', DEFAULT_MS_TAG)
        if not s.has(tag):
            raise RuntimeError(f"The Atoms object does not have a '{tag}' array.")
        return f(s, *args, **kwargs)

    return decorated_f


def tensor_mean_property(property_name):
    """
    Decorator for creating mean methods that extract a specific property from MagneticShielding objects.
    
    Parameters:
      property_name (str): The name of the property to extract from each MagneticShielding object.
                          Must be a valid attribute or property of MagneticShielding.
    
    Returns:
      decorator: A decorator for mean methods
    """
    def decorator(method):
        def wrapper(self, s, axis=None, weights=None, **kwargs):
            # Get the mean MSTensor
            meanTensors = MSTensor().mean(s, axis=axis, weights=weights, **kwargs)
            # If meanTensors is a list of MagneticShielding objects, extract the specified property
            if isinstance(meanTensors, list) and all(isinstance(T, MagneticShielding) for T in meanTensors):
                # Extract the specified property from each tensor
                return np.array([getattr(T, property_name) for T in meanTensors])
            # If meanTensors is a single MagneticShielding object, extract the specified property
            elif isinstance(meanTensors, MagneticShielding):
                # Extract the specified property from the tensor
                return getattr(meanTensors, property_name)
            # If meanTensors is not a list of MagneticShielding objects, raise an error
            else:
                raise ValueError("meanTensors must be a list of MagneticShielding objects")
        return wrapper
    return decorator


class MSTensor(AtomsProperty):
    """
    MSTensor

    Produces a list of MagneticShielding objects containing the magnetic shielding
    tensors for each atom in the system. 
    Requires the Atoms object to have been loaded from a
    .magres file containing the relevant information.

    Parameters:
      order (str):  Order to use for eigenvalues/eigenvectors. Can
                    be 'i' (ORDER_INCREASING), 'd'
                    (ORDER_DECREASING), 'h' (ORDER_HAEBERLEN) or
                    'n' (ORDER_NQR). Default is 'i'.
      tag (str, optional): name of the array containing magnetic shielding tensors.
                    Defaults to 'ms'.

    Returns:
      ms_tensors (list): list of MagneticShielding objects

    """
    default_name = "ms_tensors"
    default_params = {"order": MagneticShielding.ORDER_INCREASING, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, order, tag, **kwargs):
        symbols = s.get_chemical_symbols()
        ms_list = _get_tensor_array(s, tag)

        ref_list = [None] * len(ms_list)
        grad_list = [-1] * len(ms_list)
        if "ref" in kwargs:
            ref = kwargs.pop("ref")
            ref_list = [ref[symbol] if isinstance(ref, dict) else ref for symbol in symbols]
        if "grad" in kwargs:
            grad = kwargs.pop("grad")
            grad_list = [grad[symbol] if isinstance(grad, dict) else grad for symbol in symbols]

        ms_tensors = [MagneticShielding(ms, species=symbol, order=order, reference=ref, gradient=grad)
                        for ms, symbol, ref, grad in zip(ms_list, symbols, ref_list, grad_list)]
        return ms_tensors

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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_diag (np.ndarray): list of eigenvalues and eigenvectors

    """

    default_name = "ms_diagonal"
    default_params = {"save_array": True}

    @staticmethod
    @_has_ms_check
    def extract(s, save_array, tag):

        ms_tensors = _get_tensor_array(s, tag)
        ms_diag = [np.linalg.eigh((ms + ms.T) / 2.0) for ms in ms_tensors]
        ms_evals, ms_evecs = (np.array(a) for a in zip(*ms_diag))

        if save_array:
            s.set_array(f"{tag}_diagonal" + "_evals", ms_evals)
            # Store also the Haeberlen sorted version
            s.set_array(f"{tag}_diagonal" + "_evals_hsort", _haeb_sort(ms_evals))
            s.set_array(f"{tag}_diagonal" + "_evecs", ms_evecs)

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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_shielding (np.ndarray): list of shieldings

    """

    default_name = "ms_shielding"
    default_params = {"save_array": True, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, save_array, tag) -> np.ndarray:

        ms_tensors = _get_tensor_array(s, tag)
        ms_shielding = np.trace(ms_tensors, axis1=1, axis2=2) / 3.0

        if save_array:
            # Save the isotropic shieldings
            s.set_array(f"{tag}_shielding", ms_shielding)

        return ms_shielding

    @tensor_mean_property('isotropy')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSShielding property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_shielding_mean (np.ndarray): The mean of the MSShielding property.
        """


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_shift (np.ndarray): list of shifts

    """

    default_name = "ms_shift"
    default_params = {"ref": {}, "grad": -1.0, "save_array": True, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, ref, grad, save_array, tag)-> np.ndarray:
        # make sure we have some references set!
        if not ref:
            raise ValueError("No reference provided for chemical shifts")


        # get shieldings
        ms_shieldings = MSShielding.get(s, tag=tag)

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
        ms_shifts = references + (np.array(gradients) * np.array(ms_shieldings)) / (1 + references * 1e-6)

        if save_array:
            # Save the isotropic shifts
            s.set_array(f"{tag}_shift", ms_shifts)


        return ms_shifts

    @tensor_mean_property('shift')
    def mean(self, s, axis=None, weights=None, **kwargs):
        """
        Calculate the mean of the MSShift property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.
          **kwParameters: ref and grad parameters for the MSShift calculation. For example,
                    ref={'C': 100.0, 'H': 200.0} and grad=-1.0.

        Returns:
          ms_shift_mean (np.ndarray): The mean of the MSShift property.
        """


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_iso (np.ndarray): list of shieldings/shifts

    """

    default_name = "ms_isotropy"
    default_params = {"ref": {}, "grad": -1.0, "save_array": True, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, ref, grad, save_array, tag) -> np.ndarray:

        if ref:
            # the user wants to use the chemical shift
            ms_iso = MSShift.get(s, ref=ref, grad=grad, save_array=save_array, tag=tag)
        else:
            # the user wants to use the magnetic shielding
            ms_iso = MSShielding.get(s, save_array=save_array, tag=tag)

        if save_array:
            # Save the isotropic shifts
            s.set_array(f"{tag}_isotropy", ms_iso)
        return ms_iso

    def mean(self, s, axis=None, weights=None, **kwargs):
        """
        Calculate the mean of the MSIsotropy property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.
          **kwParameters: ref and grad parameters for the MSIsotropy calculation. For example,
                    ref={'C': 100.0, 'H': 200.0} and grad=-1.0.

        Returns:
          ms_iso_mean (np.ndarray): The mean of the MSIsotropy property.
        """
        # if references are provided in kwargs, we need to calculate the chemical shift
        if kwargs.get("ref"):
            return MSShift().mean(s, axis=axis, weights=weights, **kwargs)
        # otherwise we return the isotropic shielding
        else:
            return MSShielding().mean(s, axis=axis, weights=weights, **kwargs)


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_list (np.ndarray): list of anisotropies

    """

    default_name = "ms_anisotropy"
    default_params = {"force_recalc": False, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc, tag)-> np.ndarray:

        if not s.has(f"{tag}_diagonal" + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s, tag=tag)

        ms_evals = s.get_array(f"{tag}_diagonal" + "_evals_hsort")

        return _anisotropy(ms_evals)

    @tensor_mean_property('anisotropy')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSAnisotropy property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_aniso_mean (np.ndarray): The mean of the MSAnisotropy property.
        """
        # Implementation handled by decorator


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_list (np.ndarray): list of reduced anisotropies

    """

    default_name = "ms_red_anisotropy"
    default_params = {"force_recalc": False, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc, tag)-> np.ndarray:

        if not s.has(f"{tag}_diagonal" + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s, tag=tag)

        ms_evals = s.get_array(f"{tag}_diagonal" + "_evals_hsort")

        return _anisotropy(ms_evals, reduced=True)

    @tensor_mean_property('reduced_anisotropy')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSReducedAnisotropy property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_red_aniso_mean (np.ndarray): The mean of the MSReducedAnisotropy property.
        """
        # Implementation handled by decorator


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_list (np.ndarray): list of asymmetries

    """

    default_name = "ms_asymmetry"
    default_params = {"force_recalc": False, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc, tag)-> np.ndarray:

        if not s.has(f"{tag}_diagonal" + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s, tag=tag)

        ms_evals = s.get_array(f"{tag}_diagonal" + "_evals_hsort")

        return _asymmetry(ms_evals)

    @tensor_mean_property('asymmetry')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSAsymmetry property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_asym_mean (np.ndarray): The mean of the MSAsymmetry property.
        """
        # Implementation handled by decorator


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_list (np.ndarray): list of spans

    """

    default_name = "ms_span"
    default_params = {"force_recalc": False, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc, tag):

        if not s.has(f"{tag}_diagonal" + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s, tag=tag)

        ms_evals = s.get_array(f"{tag}_diagonal" + "_evals_hsort")

        return _span(ms_evals)

    @tensor_mean_property('span')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSSpan property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_span_mean (np.ndarray): The mean of the MSSpan property.
        """
        # Implementation handled by decorator


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
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_list (np.ndarray): list of skews

    """

    default_name = "ms_skew"
    default_params = {"force_recalc": False, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc, tag):

        if not s.has(f"{tag}_diagonal" + "_evals_hsort") or force_recalc:
            MSDiagonal.get(s, tag=tag)

        ms_evals = s.get_array(f"{tag}_diagonal" + "_evals_hsort")

        return _skew(ms_evals)

    @tensor_mean_property('skew')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSSkew property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_skew_mean (np.ndarray): The mean of the MSSkew property.
        """
        # Implementation handled by decorator


class MSEuler(AtomsProperty):

    """
    MSEuler

    Produces an array of Euler angles in radians expressing the orientation of
    the MS tensors with respect to the cartesian axes for each site in the Atoms object.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.


    Parameters:
        order (str):  Order to use for eigenvalues/eigenvectors. Can
                        be 'i' (ORDER_INCREASING), 'd'
                        (ORDER_DECREASING), 'h' (ORDER_HAEBERLEN) or
                        'n' (ORDER_NQR). Default is 'h' for MS tensors.
        convention (str): 'zyz' or 'zxz' accepted - the ordering of the Euler
                        angle rotation axes. Default is ZYZ 
        passive (bool):  active or passive rotations. Default is active (passive=False)
        tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.
         

    Returns:
        ms_eulers (np.array): array of Euler angles in radians

    """

    default_name = "ms_eulers"
    default_params = {"order": MagneticShielding.ORDER_HAEBERLEN,
                      "convention": "zyz",
                      "passive": False,
                      "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, order, convention, passive, tag):
        return np.array([t.euler_angles(convention, passive=passive) for t in MSTensor.get(s, order=order, tag=tag)])

    def mean(self, s, axis=None, weights=None, **kwargs):
        """
        Calculate the mean of the MSEuler property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_euler_mean (np.ndarray): The mean of the MSEuler property.
        """
        # Get the mean MSTensor
        meanTensors = MSTensor().mean(s, axis=axis, weights=weights, **kwargs)

        # If meanTensors is a list of MagneticShielding objects, extract the Euler angles
        if isinstance(meanTensors, list) and all(isinstance(T, MagneticShielding) for T in meanTensors):
            # Extract the Euler angles from each tensor
            return np.array([t.euler_angles(**kwargs) for t in meanTensors])
        # If meanTensors is a single MagneticShielding object, extract the Euler angles
        elif isinstance(meanTensors, MagneticShielding):
            # Extract the Euler angles from the tensor
            return meanTensors.euler_angles(**kwargs)
        # If meanTensors is not a list of MagneticShielding objects, raise an error
        else:
            raise ValueError("meanTensors must be a list of MagneticShielding objects")


class MSQuaternion(AtomsProperty):

    """
    MSQuaternion

    Produces a list of ase.Quaternion objects expressing the orientation of
    the MS tensors with respect to the cartesian axes.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    This is now deprecated in favour of an explicit Euler angle calculation
    that better handles NMR tensors.

    | Parameters:
    |   order (str):  Order to use for eigenvalues/eigenvectors. Can
                        be 'i' (ORDER_INCREASING), 'd'
                        (ORDER_DECREASING), 'h' (ORDER_HAEBERLEN) or
                        'n' (ORDER_NQR). Default is 'i'.
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_quat (list): list of quaternions

    """

    default_name = "ms_quats"
    default_params = {"order": MagneticShielding.ORDER_HAEBERLEN, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, order, tag):
        return [t.quaternion for t in MSTensor.get(s, order=order, tag=tag)]

    @tensor_mean_property('quaternion')
    def mean(self, s, axis=None, weights=None):
        """
        Calculate the mean of the MSQuaternion property.

        Parameters:
          s (AtomsCollection): The collection of structures to calculate the mean for.
          axis (int or None): Axis along which to calculate the mean. Default is None.
          weights (array-like or None): Weights for each structure. Default is None.

        Returns:
          ms_quat_mean (np.ndarray): The mean of the MSQuaternion property.
        """
        # Implementation handled by decorator