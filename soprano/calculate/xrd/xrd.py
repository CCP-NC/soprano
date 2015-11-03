"""
Classes and functions for simulating X-ray diffraction
spectroscopic results from structures.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import numpy as np
from collections import namedtuple
# Internal imports
from soprano.utils import hkl2d2_matgen, minimum_supcell, inv_plane_dist

XraySpectrum = namedtuple("XraySpectrum", ("theta2", "hkl", "hkl_unique",
                                           "invd", "I", "lambdax"))


def xrd_pwd_peaks(latt_abc, lambdax=1.54056, theta2_digits=6):
    """
    Calculate the peaks (without intensities) of a powder
    XRD spectrum given the lattice in ABC form and the
    X-ray wavelength

    | Args:
    |    latt_abc (np.ndarray): periodic lattice in ABC form,
    |                           Angstroms and radians
    |    lambdax (Optional[float]): X-ray wavelength in Angstroms
    |                               (default is 1.54056 Ang)
    |    theta2_digits (Optional[int]): Rounding within which
    |                                  two theta angles (in degrees)
    |                                  are considered to be equivalent
    |                                  (default is 6 digits)

    | Returns:
    |    xpeaks (XraySpectrum): a named tuple containing the peaks
    |                           with theta2, corresponding hkl indices,
    |                           a unique hkl tuple for each peak,
    |                           inverse reciprocal lattice distances,
    |                           intensities and wavelength

    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    # First a sanity check
    latt_abc = np.array(latt_abc, copy=False)
    if latt_abc.shape != (2, 3):
        raise ValueError("Invalid argument latt_abc passed to xrd_pwd_peaks")

    inv_d_max = 2.0/lambdax  # Upper limit to the inverse distance

    # Second, find the matrix linking hkl indices to the inverse distance
    hkl2d2 = hkl2d2_matgen(latt_abc)
    hkl_bounds = minimum_supcell(inv_d_max, r_matrix=hkl2d2)

    hrange = range(-hkl_bounds[0], hkl_bounds[0]+1)
    krange = range(-hkl_bounds[1], hkl_bounds[1]+1)
    lrange = range(-hkl_bounds[2], hkl_bounds[2]+1)

    # We now build a full grid of hkl indices. In this way
    # iteration is performed over numpy arrays and thus faster
    hkl_grid = np.array(np.meshgrid(hrange, krange, lrange)).reshape((3, -1))
    inv_d_grid = np.sqrt(np.sum(hkl_grid *
                                np.tensordot(hkl2d2, hkl_grid,
                                             axes=((1,), (0,))), axis=0))

    # Some will still have inv_d > inv_d_max, we fix that here
    # We also eliminate both 2theta = 0 and 2theta = pi to avoid
    # divergence later in the geometric factor
    valid_i = np.where((inv_d_grid < inv_d_max)*(inv_d_grid > 0))[0]
    hkl_grid = hkl_grid[:, valid_i]
    inv_d_grid = inv_d_grid[valid_i]
    theta_grid = np.arcsin(inv_d_grid/inv_d_max)
    # Now we calculate theta2.
    # Theta needs to be truncated to a certain number of
    # digits to avoid the same peak to be wrongly
    # considered as two.
    # This function also takes care of the sorting
    unique_sorting = np.unique(np.round(2.0*theta_grid*180.0/np.pi,
                                        theta2_digits),
                               return_index=True,
                               return_inverse=True)

    peak_n = len(unique_sorting[0])
    hkl_sorted = [hkl_grid[:,
                           np.where(unique_sorting[2] == i)[0]].T.tolist()
                  for i in range(peak_n)]
    hkl_unique = hkl_grid[:, unique_sorting[1]].T
    invd = inv_d_grid[unique_sorting[1]]
    xpeaks = XraySpectrum(unique_sorting[0],
                          np.array(hkl_sorted),
                          hkl_unique,
                          invd,
                          np.zeros(peak_n),
                          lambdax)

    return xpeaks


def xrd_spec_simul(peaks_th2, peaks_I, th2_axis, peak_func=None,
                   peak_f_args=[], baseline=0.0):
    """
    Simulate an XRD spectrum given positions of peaks, intensities, baseline,
    and a peak function (a Gaussian by default).

    | Args:
    |   peaks_th2 (np.ndarray): 2*theta positions of the peaks to simulate,
    |                           in degrees
    |   peaks_I (np.ndarray): intensities of the peaks to simulate, arbitrary
    |                           units
    |   th2_axis (np.ndarray): theta2 axis along which to simulate the
    |                          spectrum
    |   peak_func (Optional[function<float, float, *kargs>
    |                       => <np.ndarray>]): the function used to simulate
    |                                          peaks. Should take th2 as its
    |                                          first argument, peak centre as
    |                                          its second, and any number of
    |                                          optional arguments. Returns a
    |                                          numpy array containing the peak
    |                                          shape. Should be able to work
    |                                          with numpy arrays as input
    |   peak_f_args (Optional[list<float>]): optional arguments for peak_func.
    |                                        If no peak_func has been supplied
    |                                        by the user, the first value will
    |                                        be used as the Gaussian width
    |   baseline (Optional[float]): baseline to use as starting point for the
    |                               simulated spectrum

    | Returns:
    |   simul_spec (np.ndarray): simulated XRD spectrum
    |   simul_peaks (np.ndarray): simulated spectrum broken by peak
    |                             contribution along axis 1

    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    # Sanity checks here

    peaks_th2 = np.array(peaks_th2, copy=False)
    if len(peaks_th2.shape) != 1:
        raise ValueError("Invalid peaks_th2 passed to xrd_spec_simul")
    peaks_I = np.array(peaks_I, copy=False)
    if peaks_th2.shape != peaks_I.shape:
        raise ValueError("peaks_I should have the same shape as peaks_th2")
    th2_axis = np.array(th2_axis, copy=False)
    if len(th2_axis.shape) != 1:
        raise ValueError("Invalid th2_axis passed to xrd_spec_simul")

    # For peak_func there's little we can do to check, user's responsibility.
    if peak_func is not None:
        if len(inspect.getargspec(peak_func).args) > (2 + len(peak_f_args)):
            raise ValueError("Invalid peak_func passed to xrd_spec_simul")
    else:
        # A gaussian here
        peak_func = _gauss_peak_default
        if len(peak_f_args) == 0:
            # Use default width
            peak_f_args = [0.1]
        else:
            peak_f_args = peak_f_args[:1]

    # So here we are. Actual simulation!
    simul_peaks = peaks_I[None, :]*peak_func(th2_axis[:, None],
                                             peaks_th2[None, :],
                                             *peak_f_args)

    simul_spec = np.sum(simul_peaks, axis=1) + np.ones(len(th2_axis))*baseline

    return simul_spec, simul_peaks


def leBail_Ifit(xpeaks, th2_axis, I_axis,
                peak_func=None, peak_f_args=[], baseline=0.0,
                rwp_tol=1e-2, max_iter=100):
    """
    Perform a refining on an XraySpectrum object's intensities based on
    experimental data with leBail's method.

    | Args:
    |   xpeaks (XraySpectrum): object containing the details of the XRD peaks
    |   th2_axis (np.ndarray): array of 2*theta values for the experimental
    |                          data
    |   I_axis (np.ndarray): array of intensity values for the experimental
    |                        data
    |   peak_func (Optional[function<float, float, *kargs>
    |                       => <np.ndarray>]): the function used to simulate
    |                                          peaks. Should take th2 as its
    |                                          first argument, peak centre as
    |                                          its second, and any number of
    |                                          optional arguments. Returns a
    |                                          numpy array containing the peak
    |                                          shape. Should be able to work
    |                                          with numpy arrays as input
    |   peak_f_args (Optional[list<float>]): optional arguments for peak_func.
    |                                        If no peak_func has been supplied
    |                                        by the user, the first value will
    |                                        be used as the Gaussian width
    |   baseline (Optional[float]): baseline of experimental data, if present
    |   rwp_tol (Optional[float]): tolerance on the Rwp error value between
    |                              two iterations that stops the calculation.
    |                              Default is 1e-2
    |   max_iter (Optional[int]): maximum number of iterations to perform

    | Returns:
    |   xpeaks_scaled (XraySpectrum): a new XraySpectrum object, with
    |                                 intensities properly scaled to match the
    |                                 experimental data
    |   simul_spec (np.ndarray): final simulated XRD spectrum
    |   simul_peaks (np.ndarray): final simulated spectrum broken by peak
    |                             contribution along axis 1
    |   rwp (float): the final value of Rwp (fitness of simulated to
    |                experimental data)

    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    # Sanity check (the rest is up to the other functions)
    if type(xpeaks) != XraySpectrum:
        raise ValueError("Invalid xpeaks passed to leBail_Ifit")

    # Fetch the data
    peaks_th2 = xpeaks.theta2

    # Find a suitable starting height for the peaks
    peaks_I = (xpeaks.I*0.0)+np.amax(I_axis)/4.0

    # Perform the first simulation
    simul_spec, simul_peaks = xrd_spec_simul(peaks_th2, peaks_I, th2_axis,
                                             peak_func=peak_func,
                                             peak_f_args=peak_f_args,
                                             baseline=baseline)
    # Calculate Rwp
    rwp0 = _Rwp_eval(simul_spec, I_axis)

    # Now the iterations
    for i in range(max_iter):
        # Perform a correction
        peaks_I *= _leBail_rescale_I(simul_peaks, simul_spec, I_axis)
        # Recalculate
        simul_spec, simul_peaks = xrd_spec_simul(peaks_th2, peaks_I, th2_axis,
                                                 peak_func=peak_func,
                                                 peak_f_args=peak_f_args,
                                                 baseline=baseline)
        # Calculate Rwp
        rwp = _Rwp_eval(simul_spec, I_axis)
        # Gain?
        delta_rwp = abs((rwp-rwp0)/rwp)
        if delta_rwp < rwp_tol:
            break
        rwp0 = rwp

    xpeaks = XraySpectrum(peaks_th2, xpeaks.hkl,
                          xpeaks.hkl_unique,
                          xpeaks.invd,
                          peaks_I, xpeaks.lambdax)

    return xpeaks, simul_spec, simul_peaks, rwp


# Here be utility functions that are NOT meant to be used by the uninitiated.
# Tread at your own peril.


def _Rwp_eval(simul_spec, exp_spec):
    """Evaluate Rwp for use in LeBail fitting"""

    weights = 1.0/exp_spec

    r_wp = np.sqrt(np.sum(weights*(exp_spec-simul_spec)**2.0) /
                   np.sum(weights*exp_spec**2.0))

    return r_wp


def _leBail_rescale_I(simul_peaks, simul_spec, exp_spec):
    """Returns rescaling factors for intensities in leBail fitting"""

    sum_peaks = np.sum(simul_peaks, axis=0)
    # It could be zero somewhere, fix that
    sum_peaks = np.where(sum_peaks > 0, sum_peaks, np.inf)
    i_scale = np.sum(exp_spec[:, None]*simul_peaks/simul_spec[:, None],
                     axis=0)/sum_peaks

    return i_scale


def _gauss_peak_default(x, x0, w):
    """Gaussian peak function (for spectrum simulation)"""
    return np.exp(-((x-x0)/w)**2.0)
