"""
Classes and functions for simulating X-ray diffraction
spectroscopic results from structures.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from collections import namedtuple
# Internal imports
from soprano.utils import hkl2d2_matgen, minimum_supcell, inv_plane_dist

XraySpectrum = namedtuple("XraySpectrum", ("theta2", "hkl", "I", "lambdax"))


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

    # Now generate a list of peaks based on their hkl values

    xpeaks = XraySpectrum([], [], [], lambdax)

    for h in hrange:
        for k in krange:
            for l in lrange:

                hkl = [h, k, l]
                inv_d = inv_plane_dist(hkl, hkl2d2)

                # Some will still have inv_d > inv_d_max, we fix that here
                # We also eliminate both 2theta = 0 and 2theta = pi to avoid
                # divergence later in the geometric factor
                if inv_d < inv_d_max and inv_d > 0:
                    theta = np.arcsin(inv_d/inv_d_max)
                    # Theta needs to be truncated to a certain number of
                    # digits to avoid the same peak to be wrongly
                    # considered as two
                    theta2 = round(2.0*theta*180/np.pi, theta2_digits)
                    if theta2 not in xpeaks.theta2:
                        xpeaks.theta2.append(theta2)
                        xpeaks.hkl.append([hkl])
                        xpeaks.I.append(0.0)
                    else:
                        theta2_i = xpeaks.theta2.index(theta2)
                        xpeaks.hkl[theta2_i] += [hkl]

    # Now on to sorting by theta2
    sorted_th2, sorted_hkl = zip(*sorted(zip(xpeaks.theta2, xpeaks.hkl),
                                         key=lambda x: x[0]))
    xpeaks = XraySpectrum(sorted_th2, sorted_hkl, xpeaks.I, lambdax)

    return xpeaks
