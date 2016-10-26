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

"""Generator producing structures rattled of a given amount"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
# Internal imports
import soprano.utils as utils


def rattleGen(struct, amplitude=0.01, n=100, method='uniform'):
    """Generator function to create multiple structures by randomly displacing
    atoms of a given amount.

    | Args:
    |   struct (ase.Atoms): the starting structure to randomize
    |   amplitude (float or np.ndarray): the amplitude of the random
    |                                    displacement. Can be a single float
    |                                    for all atoms, a 1D numpy array of
    |                                    length N (N being the number of
    |                                    atoms, one value each) or a 2D numpy
    |                                    array of shape (N,3) (one value for
    |                                    each dimension).
    |                                    These values are used as interval for
    |                                    uniform random numbers and as stdev
    |                                    for normal random numbers
    |   n (int): maximum number of structures to generate. If set to None will
    |            generate infinite structures
    |   method (str): must be either 'uniform' or 'normal'. In the first case
    |                 the rattling will be a uniform random number between
    |                 +amplitude and -amplitude. In the second case it will be
    |                 a gaussian random number with +amplitude standard
    |                 deviation.

    | Returns:
    |   rattleGenerator (generator): an iterator that yields copies of the
    |                                base structure with randomly displaced
    |                                atoms.

    """

    if method not in ('uniform', 'normal'):
        raise ValueError('Invalid method argument passed to rattleGen')

    pos = struct.get_positions()

    # Check amplitude shape
    try:
        amplitude = np.array(amplitude)*1.0
    except TypeError:
        raise ValueError('Invalid amplitude argument passed to rattleGen')

    ampsh = amplitude.shape
    if len(ampsh) == 1:
        if ampsh != (pos.shape[0],):
            raise ValueError('Shape mismatch between amplitude and struct '
                             'arguments passed to rattleGen')
        amplitude = amplitude[:, None]
    elif len(ampsh) == 2:
        if ampsh != pos.shape:
            raise ValueError('Shape mismatch between amplitude and struct '
                             'arguments passed to rattleGen')

    i = 0

    n = np.inf if n is None else n

    while i < n:

        rnds = struct.copy()

        # Rattle positions
        if method == 'uniform':
            dxyz = (np.random.random(pos.shape)-0.5)*2*amplitude
        elif method == 'normal':
            dxyz = np.random.normal(size=pos.shape)*amplitude

        rnds.set_positions(pos+dxyz)

        yield rnds
        i += 1
