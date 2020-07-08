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

"""Generator producing structures interpolated between two extremes"""

import os
import numpy as np
# Internal imports
import soprano.utils as utils


def linspaceGen(struct_0, struct_1, steps=10, periodic=False):
    """Generator function to create multiple structures with positions
    interpolated linearly between two extremes.

    | Args:
    |   struct_0 (ase.Atoms): the starting structure
    |   struct_1 (ase.Atoms): the final structure. The atoms should be in the
    |                         same order as the ones in struct_0
    |   steps (Optional[int]): number of interpolated steps to produce
    |                          (extremes included). Default is 10
    |   periodic (Optional[bool]): if True the interpolation will take into
    |                              account periodic boundaries and interpolate
    |                              between positions in struct_0 and the
    |                              closest periodic copy of positions in
    |                              struct_1. By default set to False

    | Returns:
    |   linspaceGenerator (generator): an iterator object that yields
    |                                  structures created by linear
    |                                  interpolation.

    """

    # First, a compatibility check
    chem0 = struct_0.get_chemical_symbols()
    chem1 = struct_1.get_chemical_symbols()

    if chem0 != chem1:
        raise RuntimeError('The two structures passed to linspaceGen do not '
                           'have the same chemical composition')

    pos0 = struct_0.get_positions()
    pos1 = struct_1.get_positions()

    rootname = struct_0.info['name'] if 'name' in struct_0.info else 'linspace'

    # Adjust pos1 to be periodic if asked to
    if periodic:
        dpos = pos1-pos0
        dpos = utils.minimum_periodic(dpos, struct_0.get_cell())[0]
        pos1 = pos0+dpos

    for i, t in enumerate(np.linspace(0, 1, steps)):

        pos = pos0*(1-t)+pos1*t
        struct = struct_0.copy()
        struct.set_positions(pos)
        struct.info['name'] = '{0}_{1}'.format(rootname, i)

        yield struct
