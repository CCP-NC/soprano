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

"""Generator producing molecular neighbourhoods"""

import numpy as np
# Internal imports
from soprano.utils import minimum_supcell, supcell_gridgen


def molecularNeighbourhoodGen(struct, mols, central_mol=0, max_R=10,
                              method='com'):
    """Generator function to create a spherical molecular neighbourhood. Given
    a structure and its molecules as returned by the Molecules property, 
    produce supercell structures that contain one molecule each, progressively
    further away from the one indicated as central.

    | Args:
    |   struct (ase.Atoms): original structure
    |   mols ([ase.AtomsSelection]): list of molecules, as returned by the
    |                                soprano.properties.linkage.Molecules 
    |                                class.
    |   central_mol (int): index of the molecule whose centre of mass is 
    |                      considered central. Default is 0.
    |   max_R (float): maximum radius of the neighbourhood sphere. Default is 
    |                  10 Ang.
    |   method (str): method to compute distance between molecules. 'com' 
    |                 means using the center of mass. 'nearest' means using
    |                 the closest atom. Default is 'com'.

    | Returns:
    |   molecularNeighbourhoodGen (generator): an iterator object that yields
    |                                         structures within the given 
    |                                         spherical neighbourhood.                                         

    """

    # Supercell size?
    scell = minimum_supcell(max_R, struct.get_cell())
    fgrid, grid = supcell_gridgen(struct.get_cell(), scell)

    # Center?
    mol_structs = [m.subset(struct) for m in mols]
    mol_coms = np.array([a.get_center_of_mass() for a in mol_structs])

    # Origin
    p0 = mol_coms[central_mol]

    # Positions?
    if method == 'com':
        positions = mol_coms[None, :, :]+grid[:, None, :]-p0
    elif method == 'nearest':
        positions = np.zeros((len(grid), len(mols), 3))
        for i, a in enumerate(mol_structs):
            dp = a.get_positions() - p0
            p = dp[None,:,:] + grid[:,None,:]
            # Closest one?
            positions[:,i,:] = p[range(len(grid)), 
                                 np.argmin(np.linalg.norm(p, axis=-1), axis=1)]
    else:
        raise RuntimeError('Invalid method passed to '
                           'molecularNeighbourhoodGen')

    # Order of appearance?
    distances = np.linalg.norm(positions, axis=-1)

    # Find which ones are 
