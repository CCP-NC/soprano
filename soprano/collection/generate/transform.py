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

"""Generator producing structures by repeatedly applying a transform"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from ase import Atoms
# Internal imports
import soprano.utils as utils
from soprano.properties import AtomsProperty


def transformGen(struct_0, transform, steps=10):
    """Generator function to create multiple structures with positions
    interpolated linearly between two extremes.

    | Args:
    |   struct_0 (ase.Atoms): the starting structure
    |   transform (function): the transform to apply on the given structure.
    |                         Must accept and return a structure as only
    |                         arguments. Instances of all transforms from the
    |                         soprano.properties.transform module satisfy
    |                         these conditions.
    |   steps (Optional[int]): number of times to apply the transform.
    |                          Default is 10

    | Returns:
    |   linspaceGenerator (generator): an iterator object that yields
    |                                  structures created by linear
    |                                  interpolation.

    """

    # Sanity check
    if not hasattr(transform, '__call__'):
        raise ValueError('Invalid transform function passed to transformGen')

    rootname = struct_0.info['name'] if 'name' in struct_0.info else 'transf'
    struct = struct_0

    for i in range(steps):

        struct = transform(struct)
        if type(struct) is not Atoms:
            raise RuntimeError('Invalid return value from transform')
        struct.info['name'] = '{0}_{1}'.format(rootname, i)

        yield struct
