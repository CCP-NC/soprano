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
measure.py

Contains utility functions for measuring distances and other quantities at
the level of a single ase.Atoms object.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from soprano.properties.linkage import Bonds
from soprano.utils import (minimum_periodic, get_bonding_graph,
                           get_bonding_distance)


def euclideanDistance(s, i, j, periodic=True):
    """
    Return the distance between two atoms, in Angstroms.

    | Parameters:
    |   s (ase.Atoms): the structure on which to compute the distance
    |   i (int): index of the first atom
    |   j (int): index of the second atom
    |   periodic (bool): whether to account for periodic boundaries when 
    |                    computing the distance; default is True

    | Returns:
    |   r (float): computed distance
    """

    if i == j:
        return 0.0

    cell = s.get_cell()
    pos = s.get_positions()

    r = pos[j]-pos[i]

    if periodic:
        r, _ = minimum_periodic(r[None, :], cell)

    return np.linalg.norm(r)


def bondDistance(s, i, j, bond_matrix=None):
    """
    Return the distance in number of bonds between two atoms. Returns -1 if
    the two atoms are not connected. Requires NetworkX to be installed to
    work.

    | Parameters:
    |   s (ase.Atoms): the structure on which to compute the distance
    |   i (int): index of the first atom
    |   j (int): index of the second atom
    |   bond_matrix (np.ndarray): pre-computed bond matrix to use. If not
    |                             provided, will be calculated with default
    |                             parameters

    | Returns:
    |   r (int): computed bond distance
    """

    if bond_matrix is None:
        bprop = Bonds(return_matrix=True)
        _, bond_matrix = bprop(s)

    graph = get_bonding_graph(bond_matrix)
    r = get_bonding_distance(graph, i, j)

    return r
