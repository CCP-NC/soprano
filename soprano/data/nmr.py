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

import json
import pkgutil
import numpy as np

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
