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
Classes and functions for simulating X-ray diffraction
spectroscopic results from structures.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.calculate.xrd.xrd import XRDCalculator

from soprano.calculate.xrd.sel_rules import (get_sel_rule_from_international,
                                             get_sel_rule_from_hall)
