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
Module containing  a special set of AtomsProperties that transform an Atoms
object into another (by translating, rotating or mirroring all or some ions,
and so on). These all accept an Atoms object and some parameters and return
an Atoms object as well. Default behaviour for the .get method in most cases
will be to do nothing at all, these properties are meant to be instantiated.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.properties.transform.transform import (Translate, Rotate, Mirror)
