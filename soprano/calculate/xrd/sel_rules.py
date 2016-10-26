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
Providing an interface to selection rules for XRD peaks and various
spacegroups.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import pkgutil

try:
    _xrd_seldata = pkgutil.get_data('soprano',
                                    'data/xrd_sel_rules.json').decode('utf-8')
    xrd_sel_rules = json.loads(_xrd_seldata)
except IOError:
    xrd_sel_rules = None

try:
    _halldata = pkgutil.get_data('soprano',
                                 'data/hall_2_no.json').decode('utf-8')
    hall_2_no = json.loads(_halldata)
except IOError:
    hall_2_no = None


def _ifq(cond, ifT, ifF):
    """ Fortran-style ternary if """
    return ifT if cond else ifF


def _evenq(x):
    """ Returns True if x is even """
    return x % 2 == 0


def _integerq(x):
    """ Returns true if x is integer """
    return int(x) == x


def get_sel_rule_from_international(n, o='all'):
    """ Generate a function object that acts as a selection rule for XRD lines
    for the given symmetry group expressed in international number notation

    | Args:
    |   n (int): International number of the required spacegroup
    |   o (Optional[int]): Sub-option of the required spacegroup

    | Returns:
    |   rule_func (function< list<int> >
    |              => <bool>): a function that can be used to test triples of
    |                          Miller indices h,k,l to verify whether the
    |                          related plane gives rise or not to a peak

    | Raises:
    |   RuntimeError: if the database of XRD selection rules was not properly
    |                 loaded
    |   ValueError: if some of the passed arguments are invalid

    """

    global xrd_sel_rules, _ifq, _evenq, _integerq

    # The JSON keys are strings
    n = str(n)
    o = str(o)

    # First, let's check that we have the correct data,
    # and that the values passed are sensible

    if xrd_sel_rules is None:
        raise RuntimeError("Could not load XRD selection rules")
    if n not in xrd_sel_rules:
        raise ValueError("""Invalid n passed to
                            get_sel_rule_from_international""")

    if o not in xrd_sel_rules[n]:
        if 'all' in xrd_sel_rules[n]:
            o = 'all'
        else:
            if o != 'all':
                raise ValueError("""Invalid o passed to
                                    get_sel_rule_from_international""")
            else:
                raise ValueError("""An option o must be specified for
                                    n = {0}""".format(n))

    xrd_rule_instr = compile(xrd_sel_rules[n][o], '<string>', 'eval')

    return lambda hkl: eval(xrd_rule_instr, {'ifq': _ifq,
                                             'evenq': _evenq,
                                             'integerq': _integerq},
                            {'h': hkl[0],
                             'k': hkl[1],
                             'l': hkl[2]})


def get_sel_rule_from_hall(h):
    """ Generate a function object that acts as a selection rule for XRD lines
    for the given symmetry group expressed in Hall number notation

    | Args:
    |   h (int): Hall number of the required spacegroup

    | Returns:
    |   rule_func (function< list<int> >
    |              => <bool>): a function that can be used to test triples of
    |                          Miller indices h,k,l to verify whether the
    |                          related plane gives rise or not to a peak

    | Raises:
    |   RuntimeError: if the database of XRD selection rules or that of
    |                 Hall numbers was not properly loaded
    |   ValueError: if the passed argument is invalid

    """

    h = str(h)

    if hall_2_no is None:
        raise RuntimeError("""Could not load Hall-to-international conversion
                              table""")
    if h not in hall_2_no:
        raise ValueError("""Invalid h passed to
                            get_sel_rule_from_hall""")

    n = hall_2_no[h]['n']
    o = hall_2_no[h]['o']

    return get_sel_rule_from_international(n, o)
