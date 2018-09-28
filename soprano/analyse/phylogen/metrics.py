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

"""Utility functions to compare clusterings and evaluate similarity"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def confmat(clust1, clust2):
    """Calculate the confusion matrix for two clusterings of the same
    collection. The confusion matrix is defined so that element i,j contains
    the number of elements that are in common between cluster i of the first
    clustering and cluster j of the second:

    M[i,j] = len(set(c[i]).intersection(c[j]))

    | Args:
    |   clust1 (list): a series of clusters in the form of a list of slices,
    |                  like the second value returned by one of the clustering
    |                  methods in PhylogenCluster.
    |   clust2 (list): same as above.

    | Returns:
    |   confmat (np.ndarray): confusion matrix for the two clusterings.
    """

    cmat = [[len(set(c1).intersection(c2)) for c2 in clust2]
            for c1 in clust1]

    return np.array(cmat)


def norm_confmat(clust1, clust2):
    """Calculate the normalised confusion matrix for two clusterings of the
    same collection. The confusion matrix is defined as in the docstring of
    confmat. For the normalisation, each element i,j is divided by the
    geometric mean of the sizes of cluster i of the first clustering and
    cluster j of the second:

    NM[i,j] = M[i,j]/(len(c[i])*len(c[j]))**0.5

    | Args:
    |   clust1 (list): a series of clusters in the form of a list of slices,
    |                  like the second value returned by one of the clustering
    |                  methods in PhylogenCluster.
    |   clust2 (list): same as above.


    | Returns:
    |   nconfmat (np.ndarray): normalised confusion matrix for the two
    |                          clusterings.
    """

    cmat = confmat(clust1, clust2)
    norm = [[(len(c1)*len(c2))**0.5 for c2 in clust2] for c1 in clust1]

    return cmat/np.array(norm)


def fowles_mallows_index(clust1, clust2):
    """Calculate the Fowles-Mallows index, a measure of similarity between two
    clusterings defined as:

    F = (WI*WII)**0.5

    with

    WI = N11/sum_k(n_k*(n_k-1)/2)
    WII = N11/sum_k(n'_k*(n'_k-1)/2)

    with N11 being the number of pairs of points that are in the same cluster
    in both clusterings, and n_k (n'_k) the number of elements in cluster k
    of the first (second) clustering.

    Ref: 
    Fowlkes, E. B.; Mallows, C. L. (1 September 1983). 
    "A Method for Comparing Two Hierarchical Clusterings". 
    Journal of the American Statistical Association. 78 (383): 553. 
    doi:10.2307/2288117

    | Args:
    |   clust1 (list): a series of clusters in the form of a list of slices,
    |                  like the second value returned by one of the clustering
    |                  methods in PhylogenCluster.
    |   clust2 (list): same as above.


    | Returns:
    |   fm_ind (float): the Fowles-Mallows index 
    """

    cmat = confmat(clust1, clust2)

    N11 = (np.sum(cmat**2.0)-np.sum(cmat))/2.0
    n1 = np.array([len(c) for c in clust1])
    n2 = np.array([len(c) for c in clust2])
    WI = N11/np.sum(0.5*n1*(n1-1))
    WII = N11/np.sum(0.5*n2*(n2-1))

    return (WI*WII)**0.5
