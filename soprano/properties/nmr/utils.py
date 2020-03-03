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

"""Utility functions for NMR-related properties"""

import re
import json
import pkgutil
import numpy as np
import scipy.constants as cnst
from ase.quaternions import Quaternion

# EFG conversion constant.
# Units chosen so that EFG_TO_CHI*Quadrupolar moment*Vzz = Hz
EFG_TO_CHI = cnst.physical_constants['atomic unit of electric field '
                                     'gradient'][0]*cnst.e*1e-31/cnst.h


def _haeb_sort(evals):
    """Sort a list of eigenvalue triplets by Haeberlen convention"""
    evals = np.array(evals)
    iso = np.average(evals, axis=1)
    sort_i = np.argsort(np.abs(evals-iso[:, None]),
                        axis=1)[:, [1, 0, 2]]
    return evals[np.arange(evals.shape[0])[:, None],
                 sort_i]


def _anisotropy(haeb_evals, reduced=False):
    """Calculate anisotropy given eigenvalues sorted with Haeberlen
    convention"""

    f = 2.0/3.0 if reduced else 1.0

    return (haeb_evals[:, 2]-(haeb_evals[:, 0]+haeb_evals[:, 1])/2.0)*f


def _asymmetry(haeb_evals):
    """Calculate asymmetry"""

    return (haeb_evals[:, 1]-haeb_evals[:, 0])/_anisotropy(haeb_evals,
                                                           reduced=True)


def _span(evals):
    """Calculate span"""

    return np.amax(evals, axis=-1)-np.amin(evals, axis=-1)


def _skew(evals):
    """Calculate skew"""

    return 3*(np.median(evals,
                        axis=1) -
              np.average(evals,
                         axis=1))/_span(evals)


def _evecs_2_quat(evecs):
    """Convert a set of eigenvectors to a Quaternion expressing the
    rotation of the tensor's PAS with respect to the Cartesian axes"""

    # First, guarantee that the eigenvectors express *proper* rotations
    evecs = np.array(evecs)*np.linalg.det(evecs)[:, None, None]

    # Then get the quaternions
    return [Quaternion.from_matrix(evs.T) for evs in evecs]


def _dip_constant(Rij, gi, gj):
    """Dipolar constants for pairs ij, with distances Rij and gyromagnetic
    ratios gi and gj"""

    return - (cnst.mu_0*cnst.hbar*gi*gj / (8*np.pi**2*Rij**3))


def _J_constant(Kij, gi, gj):
    """J coupling constants for pairs ij, with reduced constant Kij and
    gyromagnetic ratios gi and gj"""

    return cnst.h*gi*gj*Kij/(4*np.pi**2)*1e19


try:
    _nmr_data = pkgutil.get_data('soprano',
                                 'data/nmrdata.json').decode('utf-8')
    _nmr_data = json.loads(_nmr_data)
except IOError:
    _nmr_data = None


def _get_nmr_data():

    if _nmr_data is not None:
        return _nmr_data
    else:
        raise RuntimeError('NMR data not available. Something may be '
                           'wrong with this installation of Soprano')


def _get_isotope_data(elems, key, isotopes={}, isotope_list=None,
                      use_q_isotopes=False):

    data = np.zeros(len(elems))
    nmr_data = _get_nmr_data()

    for i, e in enumerate(elems):

        if e not in nmr_data:
            # Non-existing element
            raise RuntimeError('No NMR data on element {0}'.format(e))

        iso = nmr_data[e]['iso']
        if use_q_isotopes and nmr_data[e]['Q_iso'] is not None:
            iso = nmr_data[e]['Q_iso']
        if e in isotopes:
            iso = isotopes[e]
        if isotope_list is not None and isotope_list[i] is not None:
            iso = isotope_list[i]

        try:
            data[i] = nmr_data[e][str(iso)][key]
        except KeyError:
            raise RuntimeError('Data {0} does not exist for isotope {1} of '
                               'element {2}'.format(key, iso, e))

    return data


def _el_iso(sym):
    """ Utility function: split isotope and element in conventional
    representation.
    """

    nmr_data = _get_nmr_data()

    match = re.findall('([0-9]*)([A-Za-z]+)', sym)
    if len(match) != 1:
        raise ValueError('Invalid isotope symbol')
    elif match[0][1] not in nmr_data:
        raise ValueError('Invalid element symbol')

    el = match[0][1]
    # What about the isotope?
    iso = str(nmr_data[el]['iso']) if match[0][0] == '' else match[0][0]

    if iso not in nmr_data[el]:
        raise ValueError('No data on isotope {0} for element {1}'.format(iso,
                                                                         el))

    return el, iso
