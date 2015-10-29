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
from soprano.utils import hkl2d2_matgen

XraySpectrum = namedtuple("XraySpectrum", ("theta2", "hkl", "I", "lambdax"))


def xrd_pwd_peaks(latt_abc, lambdax=1.54056, theta2_tol=1e-6):
    """
    Calculate the peaks (without intensities) of a powder
    XRD spectrum given the lattice in ABC form and the
    X-ray wavelength

    | Args:
    |    latt_abc (np.ndarray): periodic lattice in ABC form,
    |                           Angstroms and radians
    |    lambdax (Optional[float]): X-ray wavelength in Angstroms
    |                               (default is 1.54056 Ang)
    |    theta2_tol (Optional[float]): Tolerance within which
    |                                  two theta angles (in degrees)
    |                                  are considered to be equivalent
    |                                  (default is 1e-6 deg)

    | Returns:
    |    xpeaks (XraySpectrum): a named tuple containing the peaks
    |                           with theta2, corresponding hkl indices,
    |                           intensities and wavelength

    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    # First a sanity check
    latt_abc = np.array(latt_abc)
    if latt_abc.shape != (2, 3):
        raise ValueError("Invalid argument latt_abc passed to xrd_pwd_peaks")

    # Second, find the matrix linking hkl indices to the inverse distance
    hkl2d2 = hkl2d2_matgen(latt_abc)