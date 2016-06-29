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
    A = np.concatenate((evecs1[:,0][:,None], evecs2[:,0][:,None]), axis=1)
    return np.tensordot(A, p, axes=(0, 1))


@_check_dimensionality
def standard_classcond_component(p):
    """Standardized class conditional princial component mapping (Fukunaga-
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
