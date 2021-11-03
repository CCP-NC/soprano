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


import numpy as np
from scipy.optimize import linear_sum_assignment
from soprano.utils import minimum_periodic
from soprano.properties import AtomsProperty


class RemapIndices(AtomsProperty):

    """
    RemapIndices

    Returns a list of the indices which will remap a given structure to the
    one given as reference. Remapping means creating a one-to-one correspondence
    between atoms based on their distances (which may include the periodic
    boundaries).

    | Parameters:
    |   reference (ase.Atoms):  Reference structure to map to. Required
    |   periodic (bool):        If True, take into account periodic boundaries.
    |                           Default is True.
    |   scaled (bool):          If True, use fractional instead of absolute
    |                           coordinates. Default is False.

    | Returns:
    |   remap_indices ([int]):  List of indices for the structure in the order
    |                           in which they will make it map to the reference
    |                           best 
    """

    default_name: "remap_indices"
    default_params: {
        "reference": None, 
        "periodic": True,
        "scaled": False}

    @staticmethod
    def extract(s, reference, periodic, scaled):

        # First, are these even compatible structures?
        f1 = s.get_chemical_formula()
        f2 = reference.get_chemical_formula()
        if f1 != f2:
            raise ValueError('Structures do not have the same formula')

        n = len(s)
        # Start by computing the distance matrix
        if scaled:
            p1 = s.get_scaled_positions()
            p2 = reference.get_scaled_positions()
        else:
            p1 = s.positions
            p2 = reference.positions

        rmat = p1[None,:]-p2[:,None]

        if periodic:
            if scaled:
                pbc = s.get_pbc()
                rmat = np.where(pbc[None,None,:], (rmat+0.5)%1.0-0.5, rmat)
            else:
                # We need to get a bit clever
                rlist = rmat.reshape((-1,3))
                rlist, _ = minimum_periodic(rlist, reference.cell, 
                                            pbc=s.get_pbc())
                rmat = rlist.reshape((n,n,3))

        dmat = np.linalg.norm(rmat, axis=2)

        return linear_sum_assignment(dmat)[1]

class Remap(AtomsProperty):
    """
    Remap

    Returns a structure remapped to the one given as reference. 
    Remapping means creating a one-to-one correspondence between atoms based
    on their distances (which may include the periodic boundaries).

    | Parameters:
    |   reference (ase.Atoms):  Reference structure to map to. Required
    |   periodic (bool):        If True, take into account periodic boundaries.
    |                           Default is True.
    |   scaled (bool):          If True, use fractional instead of absolute
    |                           coordinates. Default is False.

    | Returns:
    |   remap_indices ([int]):  List of indices for the structure in the order
    |                           in which they will make it map to the reference
    |                           best 
    """

    default_name: "remap"
    default_params: {
        "reference": None, 
        "periodic": True,
        "scaled": False}

    @staticmethod
    def extract(s, reference, periodic, scaled):

        indices = RemapIndices.extract(s, reference, periodic, scaled)

        return s[indices]