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
shrewd.py

Contains a class to define the SHREWD (Spherical Harmonics Reduction or Elimination by 
a Weighted Distribution) powder averaging scheme.
Implementation taken from:

Mattias Edén, Malcolm H. Levitt,
"Computation of Orientational Averages in Solid-State NMR by Gaussian Spherical Quadrature",
Journal of Magnetic Resonance,
Volume 132, Issue 2,
1998,
Pages 220-239,
ISSN 1090-7807,
https://doi.org/10.1006/jmre.1998.1427.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import warnings
from scipy.special import legendre, sph_harm
from scipy.linalg import lstsq
from scipy.optimize import minimize
from soprano.calculate.powder.zcw import ZCW


class SHREWD(ZCW):

    def _calc_engine(self, N):
        phi, ct, weights = super()._calc_engine(N)

        # Compute weights
        lmax = len(phi)
        lM = np.array([legendre(l)(ct) for l in range(lmax)]).T
        lb = np.array([1] + [0]*(lmax-1))

        alpha = 0.1 # This term helps with keeping the weights in check

        def f(w):
            return np.sum((np.dot(lM, w)-lb)**2 + alpha*w**2)

        def df(w):
            return 2*(np.dot((np.dot(lM, w)-lb), lM)) + 2*alpha*w

        sol = minimize(f, weights, jac=df)
        if sol.status != 0:
            warnings.warn('Optimization of weights for SHREWD did not converge')
        weights = sol.x
        # Normalise them, just for safety
        weights /= np.sum(weights)

        return phi, ct, weights
