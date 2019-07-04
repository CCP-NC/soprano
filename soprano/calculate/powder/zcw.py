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
zcw.py

Contains a class to define the ZCW (Zaremba-Conroy-Wolfsberg) powder averaging scheme.
Implementation taken from:

Edén, M. (2003), Computer simulations in solid‐state NMR. III. Powder averaging. 
Concepts Magn. Reson., 18A: 24-55. doi:10.1002/cmr.a.10065
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.calculate.powder.powder import PowderScheme


class ZCW(PowderScheme):

    def _calc_engine(self, N):
        # Actually computes the two primary quantities
        # obtained from ZCW: phi and costheta

        # Find the correct g
        zcw_g = [8, 13]

        zcw_c = {'sphere': (1, 2, 1),
                 'hemisphere': (-1, 1, 1),
                 'octant': (-1, 1, 4)}[self.mode]

        # Starting M & N
        zcw_M = 2
        zcw_N = 21
        while zcw_N < N:
            zcw_g.append(zcw_N)
            zcw_M += 1
            zcw_N = zcw_g[-1]+zcw_g[-2]

        # If it's over
        zcw_Nf = 1.0*zcw_N
        zcw_g = zcw_g[-1]

        n = np.arange(0, zcw_N)

        phi = 2.0*np.pi/zcw_c[2]*np.mod(n*zcw_g/zcw_Nf, 1.0)
        ct = zcw_c[0]*(zcw_c[1]*np.mod(n/zcw_Nf, 1.0)-1.0)

        weights = np.ones(len(phi))/len(phi)

        return phi, ct, weights

    def get_orient_angles(self, N):
        """
        Generate and return the ZCW angles (in the form of angles in radians)
        and weights.

        | Args:
        |   N (int): lower bound for the number of orientations generated. The
        |            algorithm will generate at least N orientations.

        | Returns:
        |   angles, weights (np.ndarray): arrays containing respectively the 
        |                                 orientations [theta, phi] and the weights.
        """

        phi, ct, weights = self._calc_engine(N)

        return np.array([np.arccos(ct), phi]).T, weights

    def get_orient_trig(self, N):
        """
        Generate and return the ZCW angles (in the form of trigonometric fuctions)
        and weights.

        | Args:
        |   N (int): lower bound for the number of orientations generated. The
        |            algorithm will generate at least N orientations.

        | Returns:
        |   angles, weights (np.ndarray): arrays containing respectively the 
        |                                 orientations [cos(theta), sin(theta),
        |                                 cos(phi), sin(phi)] and the weights.
        """

        phi, ct, weights = self._calc_engine(N)

        cp = np.cos(phi)
        sp = np.sin(phi)
        st = (1.0-ct**2)**0.5

        return np.array([ct, st, cp, sp]).T, weights

    def get_orient_points(self, N):
        """
        Generate and return the ZCW angles (in the form of points on the surface of
        a sphere) and weights.

        | Args:
        |   N (int): lower bound for the number of orientations generated. The
        |            algorithm will generate at least N orientations.

        | Returns:
        |   angles, weights (np.ndarray): arrays containing respectively the 
        |                                 orientations [x, y, z] and the weights.
        """
        orients, weights = self.get_orient_trig(N)
        ct, st, cp, sp = orients.T
        points = np.array([st*cp, st*sp, ct]).T

        return points, weights
