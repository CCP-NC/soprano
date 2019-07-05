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
triavg.py

Contains a class to define the the POWDER averaging algorithm for NMR spectra,
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
from soprano.calculate.powder.powder import PowderScheme


class TriAvg(PowderScheme):

    def get_orient_points(self, N):
        """
        Generate and return the TriAvg angles (in the form of direction cosines),
        weights, and triangles.

        | Args:
        |   N (int): number of divisions on each side of the octahedron used to
        |            generate angles. Higher numbers lead to greater precision.

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
        points = np.concatenate([list(zip(p, N-z-p, [z]*(N-z+1)))
                                 for z, p in enumerate(points)])*1.0/N

        def z_i(z):
            return int(z*N+1.5*z-z**2/2.0)

        tris = np.array([[x, x+1, x+(N-z+1)]
                         for z in range(N) for x in range(z_i(z), z_i(z+1)-1)] +
                        [[x, x+(N-z), x+(N-z+1)]
                         for z in range(N) for x in range(z_i(z)+1, z_i(z+1)-1)])

        # Point weights
        dists = np.linalg.norm(points, axis=1)
        points /= dists[:, None]
        weights = dists**-3.0

        # Repeat on as many octants as needed
        if self.mode == 'octant':
            ranges = [[1], [1], [1]]
        elif self.mode == 'hemisphere':
            ranges = [[1, -1], [1, -1], [1]]
        elif self.mode == 'sphere':
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
        tris = uniq_inv[tris.astype(int)]

        return points, weights, tris

    def get_orient_angles(self, N):
        """
        Generate and return the TriAvg angles (in the form of angles in radians),
        weights, and triangles.

        | Args:
        |   N (int): number of divisions on each side of the octahedron used to
        |            generate angles. Higher numbers lead to greater precision.

        | Returns:
        |   points, weights, tris (np.ndarray): arrays containing respectively the
        |                                       direction cosines for each
        |                                       orientation, the weights, and the
        |                                       triangles (in form of triplets of
        |                                       indices of the first array)

        """

        points, weights, tris = self.get_orient_points()

        theta = np.arccos(points[:, 2])
        phi = np.arctan2(points[:, 1], points[:, 0])

        return np.array([theta, phi]).T, weights, tris

    def get_orient_trig(self, N):
        """
        Generate and return the TriAvg angles (in the form of trigonometric fuctions),
        weights, and triangles.

        | Args:
        |   N (int): number of divisions on each side of the octahedron used to
        |            generate angles. Higher numbers lead to greater precision.

        | Returns:
        |   points, weights, tris (np.ndarray): arrays containing respectively the
        |                                       direction cosines for each
        |                                       orientation, the weights, and the
        |                                       triangles (in form of triplets of
        |                                       indices of the first array)

        """

        points, weights, tris = self.get_orient_points()

        ct = points[:, 2]
        st = (1-ct**2)**0.5
        cp = points[:, 0]/st
        sp = points[:, 1]/st

        return np.array([ct, st, cp, sp]).T, weights, tris

    def average(self, x, y, weights, tris):

        triy = np.sort(y[tris], axis=1)
        triweights = np.average(weights[tris], axis=1)
        y = np.zeros(x.shape)

        dx = np.array([-x[1]+x[0], x[1]-x[0]])
        rects = np.repeat(dx[None, :]/2.0, len(y), axis=0) + x[:, None]

        # Make a matrix of 5-arrays (first 3 are tri, latter 2 are rect)
        trirect_mat = np.concatenate([np.repeat(triy[:, None, :], len(rects),
                                                axis=1),
                                      np.repeat(rects[None, :, :], len(triy),
                                                axis=0)],
                                     axis=2)

        # Now compute the contribution for each pair of triangle+rectangle

        with np.errstate(divide='ignore'):
            sl1 = 2.0/((trirect_mat[:, :, 2]-trirect_mat[:, :, 0]) *
                       (trirect_mat[:, :, 1]-trirect_mat[:, :, 0]))
            sl2 = 2.0/((trirect_mat[:, :, 2]-trirect_mat[:, :, 0]) *
                       (trirect_mat[:, :, 2]-trirect_mat[:, :, 1]))

        # Fix NaNs and Infs
        sl1 = np.where(np.isnan(sl1) | np.isinf(sl1), 0, sl1)
        sl2 = np.where(np.isnan(sl2) | np.isinf(sl2), 0, sl2)

        f123 = np.clip(trirect_mat[:, :, :3], trirect_mat[:, :, 3, None],
                       trirect_mat[:, :, 4, None])

        ymat = np.where((trirect_mat[:, :, 0] > trirect_mat[:, :, 4]) |
                        (trirect_mat[:, :, 2] < trirect_mat[:, :, 3]),
                        0.0,
                        ((f123[:, :, 1]-trirect_mat[:, :, 0])**2 -
                         (f123[:, :, 0]-trirect_mat[:, :, 0])**2)*0.5*sl1 +
                        ((-f123[:, :, 1]+trirect_mat[:, :, 2])**2 -
                         (-f123[:, :, 2]+trirect_mat[:, :, 2])**2)*0.5*sl2)
        ymat *= triweights[:, None]

        y = np.sum(ymat, axis=0)

        return y

