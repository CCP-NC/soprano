"""Implementation of AtomsProperties that relate to linkage of atoms"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import pkgutil
import numpy as np
from ase.quaternions import Quaternion
from soprano.utils import minimum_periodic
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection


# Pre load VdW radii
from ase.data.vdw import vdw_radii as _vdw_radii_ase
_vdw_radii_jmol = np.array(json.loads(pkgutil.get_data('soprano',
                                                       'data/vdw_jmol.json')))

_vdw_radii = {
    'ase': _vdw_radii_ase,
    'jmol': _vdw_radii_jmol
}


class LinkageList(AtomsProperty):

    """
    LinkageList

    Produces an array containing the atomic pair distances in a system,
    reduced to their shortest periodic version and sorted min to max.

    | Parameters: 
    |   size (int): maximum sizeber of distances to include. If not present,
    |               all of them will be included. If present, arrays will be
    |               cut or padded to reach this sizeber.

    | Returns:
    |   link_list ([float]): sorted list of interatomic linkage distances

    """

    default_name = 'linkage_list'
    default_params = {
        'size': 0
    }

    @staticmethod
    def extract(s, size):
        # Get the interatomic pair distances
        v = s.get_positions()
        v = v[:, None, :]-v[None, :, :]
        v = v[np.triu_indices(v.shape[0], k=1)]
        # Reduce them
        v, _ = minimum_periodic(v, s.get_cell())
        # And now compile the list
        link_list = np.linalg.norm(v, axis=-1)
        link_list.sort()
        if size > 0:
            if link_list.shape[0] >= size:
                link_list = link_list[:size]
            else:
                link_list = np.pad(link_list,
                                   (0, size-link_list.shape[0]),
                                   mode=str('constant'),
                                   constant_values=np.inf)

        return link_list


class Molecules(AtomsProperty):

    """
    Molecules

    Produces an array containing multiple AtomSelection objects representing
    molecules in the system as found by connecting atoms closer than the half
    sum of their Van der Waals radii. It will return the entire unit cell if
    the system can not be split in molecules at all.

    | Parameters:
    |   vdw_set({ase, jmol}): set of Van der Waals radii to use. Default is
    |                         the one extracted from JMol.
    |   vdw_scale (float): scaling factor to apply to the base Van der Waals
    |                      radii values. Values bigger than one make for more
    |                      tolerant molecules.
    |   default_vdw (float): default Van der Waals radius for species for
    |                        whom no data is available.
    |   save_info (bool): if True, save the found molecules as part of the 
    |                     Atoms object info. By default True.

    | Returns:
    |   molecules ([AtomSelection]): list of molecules in the form of 
    |                                AtomSelection objects.

    """

    default_name = 'molecules'
    default_params = {
        'vdw_set': 'jmol',
        'vdw_scale': 1.0,
        'default_vdw': 2.0,
        'save_info': True,
    }

    @staticmethod
    def extract(s, vdw_set, vdw_scale, default_vdw, save_info):
        # First, we need the biggest Van der Waals radius
        # So that we know how big the supercell needs to be
        vdw_vals = _vdw_radii[vdw_set][s.get_atomic_numbers()]
        vdw_vals = np.where(np.isnan(vdw_vals), default_vdw, vdw_vals)
        vdw_vals *= vdw_scale
        vdw_max = max(vdw_vals)

        # Get the interatomic pair distances
        atomn = s.get_number_of_atoms()
        triui = np.triu_indices(atomn, k=1)
        v = s.get_positions()
        v = (v[:, None, :]-v[None, :, :])[triui]
        # Reduce them
        v, v_cells = minimum_periodic(v, s.get_cell())
        v = np.linalg.norm(v, axis=-1)

        # Now distance and VdW matrices
        vdw_M = ((vdw_vals[None,:]+vdw_vals[:,None])/2.0)[triui]
        link_M = v <= vdw_M

        mol_sets = []
        unsorted_atoms = range(atomn)

        def get_linked(i):
            inds = np.concatenate((np.where(triui[1] == i)[0],
                                   np.where(triui[0] == i)[0]))
            cells = np.array([v_cells[j] for j in inds if link_M[j]])
            links = np.where(link_M[inds])[0]
            cells = np.where(links[:,None] < i, cells, -cells)
            links = np.where(links < i, links, links + 1)

            return links, cells

        while len(unsorted_atoms) > 0:
            mol_queue = [(unsorted_atoms.pop(0), np.zeros(3))]           
            current_mol = []
            while len(mol_queue) > 0:
                a1, cell1 = mol_queue.pop(0)
                current_mol.append((a1, cell1))
                # Find linked atoms
                links, cells = get_linked(a1)
                for i, l in enumerate(links):
                    if l in unsorted_atoms:
                        mol_queue.append((l, cell1 + cells[i]))
                        unsorted_atoms.remove(l)

            mol_sets.append(current_mol)


        mols = []
        for m in mol_sets:
            m_i, m_cells = zip(*m)
            mols.append(AtomSelection(s, m_i))
            mols[-1].set_array('cell_indices', m_cells)

        if save_info:
            s.info['molecules'] = mols

        return mols

class MoleculeNumber(AtomsProperty):

    """
    MoleculeNumber

    Number of molecules detected in this system. By default will use already
    existing molecules if they're present as a saved array in the system.


    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Returns:
    |   molecule_n (int): number of molecules found

    """

    default_name = 'molecule_n'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    def extract(s, force_recalc):

        if not Molecules.default_name in s.info or force_recalc:
            Molecules.get(s)

        return len(s.info['molecules'])

class MoleculeMass(AtomsProperty):

    """
    MoleculeMass

    Total mass of each of the molecules detected in this system. By default will
    use already existing molecules if they're present as a saved array in the
    system.


    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Returns:
    |   molecule_m ([float]): mass of each of the molecules present, sorted.

    """

    default_name = 'molecule_mass'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    def extract(s, force_recalc):

        if not 'molecules' in s.info or force_recalc:
            Molecules.get(s)

        mol_m = []
        all_m = s.get_masses()
        for mol in s.info[Molecules.default_name]:
            mol_m.append(np.sum(all_m[mol.indices]))

        return sorted(mol_m)

class MoleculeCOMLinkage(AtomsProperty):

    """
    MoleculeCOMLinkage

    Linkage list - following the same criteria as the atomic one - calculated
    for the centers of mass of the molecules present in the system. By default
    will use already existing molecules if they're present as a saved array in
    the system.


    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Returns:
    |   molecule_linkage ([float]): distances between all centers of mass of
    |                               molecules in the system, sorted.

    """

    default_name = 'molecule_com_linkage'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    def extract(s, force_recalc):

        if not Molecules.default_name in s.info or force_recalc:
            Molecules.get(s)

        mol_com = []
        all_m = s.get_masses()
        all_pos = s.get_positions()

        for mol in s.info[Molecules.default_name]:

            mol_pos = all_pos[mol.indices]
            mol_pos += np.tensordot(mol.get_array('cell_indices'),
                                    s.get_cell(),
                                    axes=(1,1))
            mol_ms = all_m[mol.indices]
            mol_com.append(np.sum(mol_pos*mol_ms[:,None],
                                  axis=0)/np.sum(mol_ms))
            
        # Now make the linkage
        v = np.array(mol_com)
        v = v[:, None, :]-v[None, :, :]
        v = v[np.triu_indices(v.shape[0], k=1)]
        # Reduce them
        v, _ = minimum_periodic(v, s.get_cell())
        # And now compile the list
        link_list = np.linalg.norm(v, axis=-1)
        link_list.sort()

        return link_list

class MoleculeRelativeRotation(AtomsProperty):

    """
    MoleculeRelativeRotation

    A list of relative rotations between molecules. Uses the inertia tensor
    eigenvectors to establish a local frame for each molecule, then uses
    quaternions to define a rotational distance between molecules. It then
    produces a list of geodesic distances between these quaternions.


    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Returns:
    |   molecule_relrot ([float]): list of relative rotations, as quaternion
    |                              distances, with the required ordering.

    """

    default_name = 'molecule_rel_rotation'
    default_params = {
        'force_recalc': False,
    }

    @staticmethod
    def extract(s, force_recalc):

        if not Molecules.default_name in s.info or force_recalc:
            Molecules.get(s)

        mol_quat = []
        all_m = s.get_masses()
        all_pos = s.get_positions()

        for mol in s.info[Molecules.default_name]:            
            
            mol_pos = all_pos[mol.indices]
            mol_pos += np.tensordot(mol.get_array('cell_indices'),
                                    s.get_cell(),
                                    axes=(1,1))
            mol_ms = all_m[mol.indices]

            # We still need to correct the positions with the COM
            mol_com = np.sum(mol_pos*mol_ms[:,None],
                             axis=0)/np.sum(mol_ms)
            mol_pos -= mol_com

            tens_i = np.identity(3)[None,:,:]* \
                     np.linalg.norm(mol_pos, axis=1)[:,None,None]**2

            tens_i -= mol_pos[:,None,:]*mol_pos[:,:,None]
            tens_i *= mol_ms[:,None,None]
            tens_i = np.sum(tens_i, axis=0)

            evals, evecs = np.linalg.eigh(tens_i)
            # General ordering convention: we want the component of the 
            # longest position to be positive along evecs_0, and the component
            # of the second longest (and non-parallel) position to be positive
            # along evecs_1, and the triple to be right-handed of course.
            mol_pos = sorted(mol_pos, key=lambda x: -np.linalg.norm(x))
            if len(mol_pos) > 1:
                evecs[0] *= np.sign(np.dot(evecs[0], mol_pos[0]))
            e1dirs = np.where(np.linalg.norm(np.cross(mol_pos,
                                                      mol_pos[0])) > 0)[0]
            if len(e1dirs) > 0:
                evecs[1] *= np.sign(np.dot(evecs[1], mol_pos[e1dirs[0]]))
            evecs[2] *= np.sign(np.dot(evecs[2],
                                       np.cross(evecs[0], evecs[1])))
            # Evecs must be proper
            evecs /= np.linalg.det(evecs)

            quat = Quaternion()
            quat = quat.from_matrix(evecs.T)
            mol_quat.append(quat.q)

        # Now make the linkage
        v = np.array(mol_quat)
        v = np.sum(v[:, None, :]*v[None, :, :], axis=-1)
        v = 2*v**2-1.0
        link_list = np.arccos(v[np.triu_indices(v.shape[0], k=1)])/np.pi
        link_list.sort()

        return np.array(link_list)



















