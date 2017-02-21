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
Functions for use of the POWDER averaging algorithm for NMR spectra,
as described in:

D. W. Alderman, Mark S. Solum, and David M. Grant
Methods for analyzing spectroscopic line shapes. NMR solid powder patterns
[J. Chern. Phys. 84 (7), 1 April 1986]

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def gen_pwd_ang(N, mode='sphere'):
    """
    Generate and return the POWDER angles (in the form of direction cosines),
    weights, and triangles.

    | Args:
    |   N (int): number of divisions on each side of the octahedron used to
    |            generate angles. Higher numbers lead to greater precision.
    |   mode (str): whether the angles should be distributed over the whole
    |               'sphere', over an 'hemisphere' or only on an 'octant'.
    |               Default is 'sphere'.

    | Returns:
    |   points, weights, tris (np.ndarray): arrays containing respectively the
    |                                       direction cosines for each
    |                                       orientation, the weights, and the
    |                                       triangles (in form of triplets of
    |                                       indices of the first array)

    """

    # Use the POWDER algorithm
    # First, generate the positive octant, by row
    points = [np.arange(N-z+1) for z in range(N+1)]
    points = np.concatenate([zip(p, N-z-p, [z]*(N-z+1))
                             for z, p in enumerate(points)])*1.0/N

    z_i = lambda z: int(z*N+1.5*z-z**2/2.0)
    tris = np.array([[x, x+1, x+(N-z+1)]
                     for z in range(N) for x in range(z_i(z), z_i(z+1)-1)] +
                    [[x, x+(N-z), x+(N-z+1)]
                     for z in range(N) for x in range(z_i(z)+1, z_i(z+1)-1)])

    # Point weights
    dists = np.linalg.norm(points, axis=1)
    points /= dists[:, None]
    weights = dists**-3.0

    # Repeat on as many octants as needed
    if mode == 'octant':
        ranges = [[1], [1], [1]]
    elif mode == 'hemisphere':
        ranges = [[1, -1], [1, -1], [1]]
    elif mode == 'sphere':
        ranges = [[1, -1], [1, -1], [1, -1]]
    else:
        raise ValueError("Invalid mode passed to powder_alg")

    octants = np.array(np.meshgrid(*ranges)).T.reshape((-1, 3))

    points = (points[None, :, :]*octants[:, None, :]).reshape((-1, 3))
    weights = np.tile(weights, len(octants))
    tris = np.concatenate([tris]*len(octants)) + \
        np.repeat(np.arange(len(octants))*((N+2)*(N+1))/2,
                  len(tris))[:, None]

    # Some points are duplicated though. Remove them.
    ncols = points.shape[1]
    dtype = points.dtype.descr * ncols
    struct = points.view(dtype)
    uniq, uniq_i, uniq_inv = np.unique(struct, return_index=True,
                                       return_inverse=True)
    points = uniq.view(points.dtype).reshape(-1, ncols)

    # Remap triangles
    weights = weights[uniq_i]
    tris = uniq_inv[tris]

    return points, weights, tris
