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

"""2D mapping algorithms"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.cluster import vq
from scipy import linalg


def _check_dimensionality(mapfunc):
    def mapfunc_decorated(p):
        if p.shape[0] < 2:
            raise ValueError('Dimensionality of the problem'
                             'too small for 2D mapping')
        return mapfunc(p)

    return mapfunc_decorated


@_check_dimensionality
def total_principal_component(p):
    """Total principal component mapping"""

    covmat = np.cov(p.T)
    evals, evecs = np.linalg.eig(covmat)
    A = evecs[:, :2]
    return np.tensordot(A, p, axes=(0, 1))


@_check_dimensionality
def classcond_principal_component(p):
    """Class conditional principal component mapping (Clafic)"""

    # Use k-means clustering to create two classes
    centroids, dist = vq.kmeans(p, 2)
    clusts, cdists = vq.vq(p, centroids)

    # Split the two clusters
    p1 = p[np.where(clusts == 0)]
    p2 = p[np.where(clusts == 1)]

    S1 = np.cov(p1.T)
    S2 = np.cov(p2.T)

    evals1, evecs1 = linalg.eig(a=S1)
    evals2, evecs2 = linalg.eig(a=S2)
    A = np.concatenate((evecs1[:, 0][:, None], evecs2[:, 0][:, None]), axis=1)
    return np.tensordot(A, p, axes=(0, 1))


@_check_dimensionality
def standard_classcond_component(p):
    """Standardized class conditional principal component mapping (Fukunaga-
    Koontz)"""

    # Use k-means clustering to create two classes
    centroids, dist = vq.kmeans(p, 2)
    clusts, cdists = vq.vq(p, centroids)

    # Split the two clusters
    p1 = p[np.where(clusts == 0)]
    p2 = p[np.where(clusts == 1)]

    S1 = np.cov(p1.T)
    S2 = np.cov(p2.T)

    evals, evecs = linalg.eig(a=S1, b=(S1+S2))
    A = evecs[:, :2]
    return np.tensordot(A, p, axes=(0, 1))


@_check_dimensionality
def optimal_discriminant_plane(p):
    """Optimal discriminant plane mapping (using Fischer direction)"""

    # First, split in two classes
    centroids, dist = vq.kmeans(p, 2)
    clusts, cdists = vq.vq(p, centroids)

    p1 = p[np.where(clusts == 0)]
    p2 = p[np.where(clusts == 1)]

    # Find the average points
    m1 = np.average(p1, axis=0)
    m2 = np.average(p2, axis=0)
    mdiff = m1-m2

    B = np.dot(mdiff[:, None], mdiff[None, :])

    S1 = np.cov(p1.T)
    S2 = np.cov(p2.T)
    S = S1+S2

    # So the fisher direction is found analytically...
    A = np.zeros((B.shape[0], 2))
    invS = np.linalg.inv(S)
    A[:, 0] = np.dot(invS, mdiff)
    A[:, 0] /= np.linalg.norm(A[:, 0])  # Normalized
    # And then the second one
    z = np.dot(invS, A[:, 0])
    A[:, 1] = A[:, 0]-np.dot(A[:, 0], z)*z

    return np.tensordot(A, p, axes=(0, 1))
