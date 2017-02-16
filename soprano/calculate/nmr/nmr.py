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
Classes and functions for simulating approximated NMR spectroscopic results
from structures.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import json
import pkgutil
import numpy as np

try:
    _nmr_data = pkgutil.get_data('soprano',
                                 'data/nmrdata.json').decode('utf-8')
    _nmr_data = json.loads(_nmr_data)
except IOError:
    _nmr_data = None

# Conversion functions to Tesla
# (they take element and isotope as arguments)
_larm_units = {
    'MHz': lambda e, i: 2*np.pi*1.0e6/_nmr_data[e][i]['gamma'],
    'T': lambda e, i: 1.0,
}


def _el_iso(sym):
    """ Utility function: split isotope and element in conventional
    representation
    """

    match = re.findall('([0-9]*)([A-Za-z]+)', sym)
    if len(match) != 1:
        raise ValueError('Invalid isotope symbol')
    elif match[0][1] not in _nmr_data:
        raise ValueError('Invalid element symbol')

    el = match[0][1]
    # What about the isotope?
    iso = _nmr_data[el]['iso'] if match[0][0] == '' else match[0][0]

    if iso not in _nmr_data[el]:
        raise ValueError('No data on isotope {0} for element {1}'.format(iso,
                                                                         el))

    return el, iso


class NMRCalculator(object):

    """NMRCalculator

    An object providing an interface to produce basic simulated NMR spectra
    from .magres files. It should be kept in mind that this is *not* a proper
    spin simulation tool, but merely provides a 'guide for the eye' kind of 
    spectrum to compare to experimental results. What it can simulate:

    - chemical shift of NMR peaks
    - quadrupolar shifts of NMR peaks up to second order corrections
    - effects of crystal orientation (single crystal)
    - powder average (policrystalline/powder)

    What it can NOT simulate:

    - MAS effects
    - J couplings
    - complex multi-spin interactions
    - complex NMR experiments

    | Args:
    |   sample (ase.Atoms): an Atoms object describing the system to simulate
    |                       on. Should be loaded with ASE from a .magres file
    |                       if data on shieldings and EFGs is necessary.
    |   larmor_frequency (float): larmor frequency of the virtual spectrometer
    |                             (referenced to Hydrogen). Default is 400.
    |   larmor_units (str): units in which the larmor frequency is expressed.
    |                       Default are MHz.

    """

    def __init__(self, sample, larmor_frequency=400,
                 larmor_units='MHz'):

        self.sample = sample

        self.set_larmor_frequency(larmor_frequency, larmor_units)

        self._references = {}

    def set_larmor_frequency(self, larmor_frequency=400, larmor_units='MHz',
                             element='1H'):
        """
        Set the Larmor frequency of the virtual spectrometer with the desired
        units and reference element.

        | Args:
        |   larmor_frequency (float): larmor frequency of the virtual
        |                             spectrometer. Default is 400.
        |   larmor_units (str): units in which the larmor frequency is
        |                       expressed. Can be MHz or T. Default are MHz.
        |   element (str): element and isotope to reference the frequency to.
        |                  Should be in the form <isotope><element>. Isotope
        |                  is optional, if absent the most abundant NMR active
        |                  one will be used. Default is 1H.

        """

        if larmor_units not in _larm_units:
            raise ValueError('Invalid units for Larmor frequency')

        # Split isotope and element
        el, iso = _el_iso(element)

        self._B = larmor_frequency*_larm_units[larmor_units](el, iso)

    def set_reference(self, ref, element):

        """
        Set the chemical shift reference (in ppm) for a given element. If not
        provided it will be assumed to be zero.

        | Args:
        |   ref (float): reference shielding value in ppm. Chemical shift will
        |                be calculated as this minus the atom's ms.
        |   element (str): element and isotope whose reference is set.
        |                  Should be in the form <isotope><element>. Isotope
        |                  is optional, if absent the most abundant NMR active
        |                  one will be used.

        """

        el, iso = _el_iso(element)

        if el not in self._references:
            self._references[el] = {}
        self._references[el][iso] = float(ref)

    

