"""Implementation of AtomsProperties that relate to linkage of atoms"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import pkgutil
import numpy as np
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
        v = minimum_periodic(v, s.get_cell())
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
        v = np.linalg.norm(minimum_periodic(v, s.get_cell()), axis=-1)

        # Now distance and VdW matrices
        vdw_M = ((vdw_vals[None,:]+vdw_vals[:,None])/2.0)[triui]
        link_M = v <= vdw_M

        mol_sets = []
        unsorted_atoms = range(atomn)

        def get_linked(i):
            inds = np.concatenate((np.where(triui[1] == i)[0],
                                   np.where(triui[0] == i)[0]))
            links = np.where(link_M[inds])[0]
            links = np.where(links < i, links, links + 1)

            return links

        while len(unsorted_atoms) > 0:
            mol_queue = [unsorted_atoms.pop(0)]
            current_mol = []
            while len(mol_queue) > 0:
                a1 = mol_queue.pop(0)
                current_mol.append(a1)
                # Find linked atoms
                links = get_linked(a1)
                for l in links:
                    if l in unsorted_atoms:
                        mol_queue.append(l)
                        unsorted_atoms.remove(l)

            mol_sets.append(current_mol)

        mols = [AtomSelection(s, m) for m in mol_sets]

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

        if not 'molecules' in s.info or force_recalc:
            Molecules.get(s)

        return len(s.info['molecules'])








