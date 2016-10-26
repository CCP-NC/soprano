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
Module containing AtomsProperties related specifically to CASTEP calculations.
Some of these can be looked up only in a CASTEP Calculator; others require
passing the path of the .castep file as a parameter and actually parsing its
contents.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.properties.castep.castep import (CastepEnthalpy,)
