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
from ase.geometry import get_distances
import warnings



class RemapIndices(AtomsProperty):

    """
    RemapIndices

    Returns a list of the indices which will remap a given structure to the
    one given as reference. Remapping means creating a one-to-one correspondence
    between atoms based on their distances (which may include the periodic
    boundaries).

    | Parameters:
    |   reference (ase.Atoms):  Reference structure to map to. Required
    |   mic (bool):             If True, take into account periodic boundaries
    |                           via the minimum image convention. We use the general method
    |                           from ASE. It's slow in some cases but robust. 
    |                           Default is True.
    |   check_species (bool):   If True, only compare atoms of the same species. 
    |                           Default is True.
    |   tolerance (float):      Tolerance (in Angstroms) for the distance comparison. If no atom
    |                           is found within this distance, an error is raised.
    |                           Default is 0.1 Angstrom.

    | Returns:
    |   remap_indices ([int]):  List of indices for the structure in the order
    |                           in which they will make it map to the reference
    |                           best 
    """

    default_name = "remap_indices"
    default_params= {
        "reference": None, 
        "mic": True,
        "check_species": True,
        "tolerance": 0.1}

    


    @staticmethod
    def extract(s, reference, mic, check_species, tolerance):
        
        # First, are these even compatible structures?
        f1 = s.get_chemical_formula()
        f2 = reference.get_chemical_formula()
        if f1 != f2:
            raise ValueError('Structures do not have the same formula')

        n = len(s)
        # Check species
        # group indices by species so that we only compare the same species' positions
        if check_species:
            species_list = list(set(s.get_chemical_symbols()))
            s_species_groups = [[atom.index for atom in s if atom.symbol in species] for species in species_list]
            r_species_groups = [[atom.index for atom in reference if atom.symbol in species] for species in species_list]
        else:
            species_list = ['all']
            s_species_groups = [range(n)]
            r_species_groups = [range(n)]

        # Now, loop over the species groups
        assignments = []
        for s_species_group, r_species_group, species in zip(s_species_groups, r_species_groups, species_list):

            # Start by computing the distance matrix
            p1 = reference[r_species_group].positions
            p2 = s[s_species_group].positions
            
            # use the minimum image convention
            if mic:
                pbc = reference.get_pbc()
                cell = reference.cell
                rmat, dmat = get_distances(p1, p2=p2, cell=cell, pbc=pbc)
            else:
                rmat = p1[:, None]-p2[None, :]
                dmat = np.linalg.norm(rmat, axis=2)

            # Solve the linear sum assignment problem.
            row_ind, col_ind = linear_sum_assignment(dmat)

            # Check that the assigned atoms are within the tolerance
            if (dmat[row_ind, col_ind] > tolerance).any():
                # Some atoms are not within the tolerance. raise error
                # indices of atoms that don't match:
                bad_indices = np.where(dmat[row_ind, col_ind] > tolerance)
                raise ValueError(f'Atoms with indices: {bad_indices} are not within the tolerance distance of another atom')
            # append species-specific assignment
            assignments.append([s_species_group[i] for i in col_ind])
            
        # flatten the list of assignments
        new_indices = [item for sublist in assignments for item in sublist]
        # fix the species-sublisted order -> global order
        global_order = np.argsort([item for sublist in r_species_groups for item in sublist])
        new_indices = [new_indices[i] for i in global_order]

        if not all(reference.symbols == s.symbols[new_indices]):
            warnings.warn('Warning: s and reference have some atoms whose species do not match, \n'
            'Set check_species=True if you want to only match atoms of same species')
        return new_indices

class Remap(AtomsProperty):
    """
    Remap

    Returns a structure remapped to the one given as reference. 
    Remapping means creating a one-to-one correspondence between atoms based
    on their distances (which may include the periodic boundaries).

    | Parameters:
    |   reference (ase.Atoms):  Reference structure to map to. Required
    |   mic (bool):             If True, take into account periodic boundaries
    |                           via the minimum image convention. We use the general method
    |                           from ASE. It's slow but robust. 
    |                           Default is True.
    |   check_species (bool):   If True, only compare atoms of the same species. 
    |                           Default is True.
    |   tolerance (float):      Tolerance for the distance comparison. 
    |                           Default is 0.1 Angstrom.
    | Returns:
    |   remap_indices ([int]):  List of indices for the structure in the order
    |                           in which they will make it map to the reference
    |                           best 
    """

    default_name = "remap"
    default_params = {
        "reference": None, 
        "mic": True,
        "check_species": True,
        "tolerance": 0.1}

    @staticmethod
    def extract(s, reference, mic, check_species, tolerance):

        indices = RemapIndices.extract(s, reference, mic, check_species, tolerance)

        return s[indices]