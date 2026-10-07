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
    _gradients_to_list,
    _haeb_sort,
    _references_to_list,
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
        references: list/dict/None
                    Specification of the references to convert magnetic shielding tensors to
                    chemical shifts. Can be a dict like {'H': 30, 'C': 170} or a list like
                    [30, 170]. If list, you must specify one reference per site in the system.
                    If None, no conversion is done and the outputs will be shieldings.
        gradients: list/dict/float/None
            Specification of the gradients for linear calibration.
            Default is None, which uses the full NMR formula:
            delta = (reference - shielding) / (1 - reference * 1e-6).
            When a value is provided, the simple linear form is used:
            delta = reference + gradient * shielding.


    Returns:
      ms_tensors (list): list of MagneticShielding objects

    """
    default_name = "ms_tensors"
    default_params = {"order": MagneticShielding.ORDER_INCREASING,
                      "references": None,
                      "gradients": None,
                      "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, order, references, gradients, tag):

        ms_list = _get_tensor_array(s, tag)
        elements = s.get_chemical_symbols()
        reference_list = _references_to_list(references, elements)
        gradient_list = _gradients_to_list(gradients, elements)

        if gradient_list is None:
            # Full NMR formula: pass None for gradient
            ms_tensors = [
                MagneticShielding(ms, species=symbol, order=order, reference=ref, gradient=None)
                for ms, symbol, ref in zip(ms_list, elements, reference_list)
            ]
        else:
            ms_tensors = [
                MagneticShielding(ms, species=symbol, order=order, reference=ref, gradient=grad)
                for ms, symbol, ref, grad in zip(ms_list, elements, reference_list, gradient_list)
            ]
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
    default_params = {"save_array": True, "tag": DEFAULT_MS_TAG}

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

    Two referencing modes are supported, selected by the *gradients* parameter:

    **Full NMR formula (default, gradients=None)**
      The rigorous conversion accounting for the ppm scale definition:

      .. math::
          \\delta = \\frac{\\sigma_{ref} - \\sigma}{1 - \\sigma_{ref} \\times 10^{-6}}

      This is the correct expression when ``gradients`` is omitted or *None*.
      The denominator arises because ppm is a frequency ratio, not a linear
      offset.  For typical solid-state NMR references (|σ_ref| ≲ 500 ppm) the
      correction is < 0.05 % and is often ignored in practice, but the full
      formula is kept as the default for strict correctness.

    **Linear calibration (gradients provided)**
      A simple linear model used when an explicit slope is supplied (e.g. from
      a calibration line):

      .. math::
          \\delta = \\sigma_{ref} + m \\cdot \\sigma

      where *m* is the user-supplied gradient.  This mode is selected whenever
      ``gradients`` is not *None*.

      .. warning::
          When the linear form is used and |σ_ref| > 500 ppm, a warning is
          emitted because the neglected denominator correction exceeds
          ~0.05 %.  For |σ_ref| ≈ 1000 ppm the error is ≈ 0.1 %.

    Parameters
    ----------
    references : list/float/dict
        Reference shielding per element (ppm). Must be provided.
    gradients : float/list/dict or None, optional
        Slope for the linear calibration model.  If *None* (default), the full
        NMR formula is used with an implicit slope of –1.  If a value is
        supplied, the simple linear form above is used instead.
    save_array : bool, optional
        If True, save the ``ms_shift`` array on the Atoms object. Default True.
    tag : str, optional
        Name of the array containing magnetic shielding tensors. Default 'ms'.

    Returns
    -------
    np.ndarray
        Chemical shifts in ppm.
    """

    default_name = "ms_shift"
    default_params = {"references": None, "gradients": None, "save_array": True, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, references, gradients, save_array, tag, **kwargs) -> np.ndarray:

        # Backwards compatibility for ref and grad parameters
        if "ref" in kwargs:
            references = kwargs.pop("ref")
            warnings.warn("The 'ref' parameter is deprecated. Use 'references' instead.", DeprecationWarning)
        if "grad" in kwargs:
            gradients = kwargs.pop("grad")
            warnings.warn("The 'grad' parameter is deprecated. Use 'gradients' instead.", DeprecationWarning)

        # make sure we have some references set!
        if not references:
            raise ValueError("No reference provided for chemical shifts")

        # get shieldings
        ms_shieldings = MSShielding.get(s, tag=tag)

        symbols = s.get_chemical_symbols()

        # --- REFERENCES and GRADIENTS --- #
        reference_list = _references_to_list(references, symbols)
        gradients_list = _gradients_to_list(gradients, symbols)

        # convert to numpy arrays
        references_list = np.array(reference_list)

        # Decide which formula branch to use
        if gradients_list is None:
            # Full NMR formula with implicit gradient = -1
            ms_shifts = (references_list - ms_shieldings) / (1 - references_list * 1e-6)
        else:
            # Linear calibration: δ = reference + gradient * sigma
            gradients_list = np.array(gradients_list)

            # Warn if any gradient is outside the typical -1.5 to -0.5 range
            if np.any(gradients_list < -1.5) or np.any(gradients_list > -0.5):
                warnings.warn(
                    "Gradients are outside the range: -1.5 to -0.5.\n"
                    "That's a surprising value! Please double check the gradients.\n"
                    f"You provided: {gradients}"
                )

            # Warn if |reference| > 500 ppm because the neglected denominator
            # correction becomes significant (> 0.05 %)
            if np.any(np.abs(references_list) > 500):
                max_ref = np.max(np.abs(references_list))
                correction_pct = (1 / (1 - max_ref * 1e-6) - 1) * 100
                warnings.warn(
                    f"Linear calibration used with |reference| = {max_ref:.1f} ppm. "
                    f"The neglected denominator correction is ≈ {correction_pct:.3f} %. "
                    "Use gradients=None (default) to include the full NMR formula."
                )

            ms_shifts = references_list + gradients_list * ms_shieldings

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
          **kwParameters: references and gradients parameters for the MSShift calculation. 
                    For example,
                    references={'C': 100.0, 'H': 200.0} and gradients=-1.0.

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
    |   references (float/dict): reference frequency per element. If provided, the chemical shift
    |                will be returned instead of the magnetic shielding.
    |   gradients float/list/dict: usually around -1. 
    |   save_array (bool): if True, save the diagonalised tensors in the
    |                      Atoms object as an array. By default True.
    |   tag (str): name of the array containing magnetic shielding tensors. Default: 'ms'.

    | Returns:
    |   ms_iso (np.ndarray): list of shieldings/shifts

    """

    default_name = "ms_isotropy"
    default_params = {"references": None, "gradients": None, "save_array": True, "tag": DEFAULT_MS_TAG}

    @staticmethod
    @_has_ms_check
    def extract(s, references, gradients, save_array, tag, **kwargs) -> np.ndarray:

        # Backwards compatibility for ref and grad parameters
        if "ref" in kwargs:
            references = kwargs["ref"]
            warnings.warn("The 'ref' parameter is deprecated. Use 'references' instead.", DeprecationWarning)
        if "grad" in kwargs:
            gradients = kwargs["grad"]
            warnings.warn("The 'grad' parameter is deprecated. Use 'gradients' instead.", DeprecationWarning)

        if references:
            # the user wants to use the chemical shift
            ms_iso = MSShift.get(s, references=references, gradients=gradients, save_array=save_array, tag=tag, **kwargs)
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
          **kwParameters: references and gradients parameters for the MSIsotropy calculation. For example,
                    references={'C': 100.0, 'H': 200.0} and gradients=None.

        Returns:
          ms_iso_mean (np.ndarray): The mean of the MSIsotropy property.
        """
        # if references are provided in kwargs, we need to calculate the chemical shift
        if kwargs.get("references"):
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