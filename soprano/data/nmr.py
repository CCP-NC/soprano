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
NMR Data

Data on NMR relevant properties of elements and isotopes - spin, gyromagnetic
ratio and quadrupole moment.
"""

import re
import json
import pkgutil
import numpy as np
import scipy.constants as cnst

# EFG conversion constant.
# Units chosen so that EFG_TO_CHI*Quadrupolar moment*Vzz = Hz
EFG_TO_CHI = cnst.physical_constants['atomic unit of electric field '
                                     'gradient'][0]*cnst.e*1e-31/cnst.h

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


def nmr_gamma(el, iso=None):
    """Gyromagnetic ratio for an element

    Return the gyromagnetic ratio for the given element and isotope, in
    rad/(s*T)

    | Args:
    |   el (str):   element symbol
    |   iso (int):  isotope. Default is the most abundant one.

    | Returns:
    |   gamma (float):  gyromagnetic ratio in rad/(s*T)    
    """

    isotopes = {}
    if iso is not None:
        isotopes[el] = iso

    return _get_isotope_data([el], 'gamma', isotopes=isotopes)


def nmr_spin(el, iso=None):
    """Nuclear spin for an element

    Return the nuclear spin for the given element and isotope, in
    Bohr magnetons

    | Args:
    |   el (str):   element symbol
    |   iso (int):  isotope. Default is the most abundant one.

    | Returns:
    |   I (float):  nuclear spin in Bohr magnetons
    """

    isotopes = {}
    if iso is not None:
        isotopes[el] = iso

    return _get_isotope_data([el], 'I', isotopes=isotopes)


def nmr_quadrupole(el, iso=None):
    """Quadrupole moment for an element

    Return the quadrupole moment for the given element and isotope, in
    barns

    | Args:
    |   el (str):   element symbol
    |   iso (int):  isotope. Default is the most abundant one.

    | Returns:
    |   Q (float):  quadrupole moment in barns
    """

    isotopes = {}
    if iso is not None:
        isotopes[el] = iso

    return _get_isotope_data([el], 'Q', isotopes=isotopes)
