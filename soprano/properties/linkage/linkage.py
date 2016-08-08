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
_vdw_data = pkgutil.get_data('soprano', 'data/vdw_jmol.json').decode('utf-8')
_vdw_radii_jmol = np.array(json.loads(_vdw_data))

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
    |   size (int): maximum number of distances to include. If not present,
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
        vdw_M = ((vdw_vals[None, :]+vdw_vals[:, None])/2.0)[triui]
        link_M = v <= vdw_M

        mol_sets = []
        unsorted_atoms = list(range(atomn))

        def get_linked(i):
            inds = np.concatenate((np.where(triui[1] == i)[0],
                                   np.where(triui[0] == i)[0]))
            cells = np.array([v_cells[j] for j in inds if link_M[j]])
            links = np.where(link_M[inds])[0]
            cells = np.where(links[:, None] < i, cells, -cells)
            links = np.where(links < i, links, links + 1)

            return links, cells

        while len(unsorted_atoms) > 0:
            mol_queue = [(unsorted_atoms.pop(0), np.zeros(3))]
            current_mol = []
            current_mol_cells = []
            current_mol_bonds = []
            while len(mol_queue) > 0:
                a1, cell1 = mol_queue.pop(0)
                current_mol.append(a1)
                current_mol_cells.append(cell1)
                current_mol_bonds.append([])
                # Find linked atoms
                links, cells = get_linked(a1)
                for i, l in enumerate(links):
                    if l in unsorted_atoms:
                        mol_queue.append((l, cell1 + cells[i]))
                        unsorted_atoms.remove(l)
                    current_mol_bonds[-1].append(l)
                            

            mol_sets.append((current_mol, current_mol_cells,
                             current_mol_bonds))

        mols = []
        for m_i, m_cells, m_bonds in mol_sets:
            mols.append(AtomSelection(s, m_i))
            mols[-1].set_array('cell_indices', m_cells)
            mols[-1].set_array('bonds', m_bonds)

        if save_info:
            s.info[Molecules.default_name] = mols

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

        return len(s.info[Molecules.default_name])


class MoleculeMass(AtomsProperty):

    """
    MoleculeMass

    Total mass of each of the molecules detected in this system. By default will
    use already existing molecules if they're present as a saved array in the
    system.


    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.
    |   size (int): maximum number of distances to include. If not present,
    |               all of them will be included. If present, arrays will be
    |               cut or padded to reach this sizeber.

    | Returns:
    |   molecule_m ([float]): mass of each of the molecules present, sorted.

    """

    default_name = 'molecule_mass'
    default_params = {
        'force_recalc': False,
        'size': 0,
    }

    @staticmethod
    def extract(s, force_recalc, size):

        if not 'molecules' in s.info or force_recalc:
            Molecules.get(s)

        mol_m = []
        all_m = s.get_masses()
        for mol in s.info[Molecules.default_name]:
            mol_m.append(np.sum(all_m[mol.indices]))

        mol_m = np.array(mol_m)
        mol_m.sort()

        if size > 0:
            if mol_m.shape[0] >= size:
                mol_m = mol_m[:size]
            else:
                mol_m = np.pad(mol_m,
                               (0, size-mol_m.shape[0]),
                               mode=str('constant'),
                               constant_values=np.inf)

        return mol_m


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
    |   size (int): maximum number of distances to include. If not present,
    |               all of them will be included. If present, arrays will be
    |               cut or padded to reach this sizeber.

    | Returns:
    |   molecule_linkage ([float]): distances between all centers of mass of
    |                               molecules in the system, sorted.

    """

    default_name = 'molecule_com_linkage'
    default_params = {
        'force_recalc': False,
        'size': 0,
    }

    @staticmethod
    def extract(s, force_recalc, size):

        if not Molecules.default_name in s.info or force_recalc:
            Molecules.get(s)

        mol_com = []
        all_m = s.get_masses()
        all_pos = s.get_positions()

        for mol in s.info[Molecules.default_name]:

            mol_pos = all_pos[mol.indices]
            mol_pos += np.tensordot(mol.get_array('cell_indices'),
                                    s.get_cell(),
                                    axes=(1, 1))
            mol_ms = all_m[mol.indices]
            mol_com.append(np.sum(mol_pos*mol_ms[:, None],
                                  axis=0)/np.sum(mol_ms))

        # Safety check
        if len(mol_com) < 2:
            return [np.inf]*size

        # Now make the linkage
        v = np.array(mol_com)
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
    |   size (int): maximum number of distances to include. If not present,
    |               all of them will be included. If present, arrays will be
    |               cut or padded to reach this sizeber.

    | Returns:
    |   molecule_relrot ([float]): list of relative rotations, as quaternion
    |                              distances, with the required ordering.

    """

    default_name = 'molecule_rel_rotation'
    default_params = {
        'force_recalc': False,
        'size': 0,
    }

    @staticmethod
    def extract(s, force_recalc, size):

        if not Molecules.default_name in s.info or force_recalc:
            Molecules.get(s)

        mol_quat = []
        all_m = s.get_masses()
        all_pos = s.get_positions()

        for mol in s.info[Molecules.default_name]:

            mol_pos = all_pos[mol.indices]
            mol_pos += np.tensordot(mol.get_array('cell_indices'),
                                    s.get_cell(),
                                    axes=(1, 1))
            mol_ms = all_m[mol.indices]

            # We still need to correct the positions with the COM
            mol_com = np.sum(mol_pos*mol_ms[:, None],
                             axis=0)/np.sum(mol_ms)
            mol_pos -= mol_com

            tens_i = np.identity(3)[None, :, :] * \
                np.linalg.norm(mol_pos, axis=1)[:, None, None]**2

            tens_i -= mol_pos[:, None, :]*mol_pos[:, :, None]
            tens_i *= mol_ms[:, None, None]
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

        # Safety check
        if len(mol_quat) < 2:
            return [np.inf]*size

        # Now make the linkage
        v = np.array(mol_quat)
        v = np.tensordot(v, v, axes=(-1, -1))
        link_list = 1.0-np.abs(v[np.triu_indices(v.shape[0], k=1)])
        link_list.sort()

        if size > 0:
            if link_list.shape[0] >= size:
                link_list = link_list[:size]
            else:
                link_list = np.pad(link_list,
                                   (0, size-link_list.shape[0]),
                                   mode=str('constant'),
                                   constant_values=np.inf)

        return np.array(link_list)


class HydrogenBonds(AtomsProperty):

    """
    Hydrogen Bonds

    Produces a dictionary containing the atom indices defining hydrogen bonds
    detected in the system - if required, classified by type. By default only
    O and N atoms are considered for hydrogen bonds (OH..O, OH..N and so on).
    The type is defined as AH..B where A is the symbol of the atom directly
    bonded to the proton and B the one of the hydrogen bonded one.

    | Parameters:
    |   vdw_set({ase, jmol}): set of Van der Waals radii to use. Default is
    |                         the one extracted from JMol.
    |   vdw_scale (float): scaling factor to apply to the base Van der Waals
    |                      radii values. Values bigger than one make for more
    |                      tolerant molecules.
    |   default_vdw (float): default Van der Waals radius for species for
    |                        whom no data is available.
    |   hbond_elems ([str]): chemical symbols of elements considered capable
    |                        of forming hydrogen bonds (by default O and N)
    |   max_length (float): maximum A-B length of the hydrogen bond in
    |                       Angstrom - default is 3.5 Ang
    |   max_angle (float): maximum A-H/A-B angle in the hydrogen bond in
    |                      degrees - default is 45 deg
    |   save_info (bool): if True, save the found hydrogen bonds as part of 
    |                     the Atoms object info. By default True.


    | Returns:
    |   hbondss ([dict]): list of hydrogen bonds detected
    |                     in the system by type (can contain empty arrays).
    |                     For each hydrogen bond we give index of the H
    |                     atom, index and unit cell of the A atom (the one
    |                     directly bonded), index and unit cell of the B atom
    |                     (the one that's hydrogen bonded), length and angle
    |                     in degrees.

    """

    default_name = 'hydrogen_bonds'
    default_params = {
        'vdw_set': 'jmol',
        'vdw_scale': 1.0,
        'default_vdw': 2.0,
        'hbond_elems': ['O', 'N'],
        'max_length': 3.5,
        'max_angle': 45.0,
        'save_info': True
    }

    @staticmethod
    def extract(s, vdw_set, vdw_scale, default_vdw, hbond_elems,
                max_length, max_angle, save_info):

        def elem_inds(s, el):
            return [i for i, cs in enumerate(s.get_chemical_symbols())
                    if cs == el]

        def bname(A, B):
            return '{0}H..{1}'.format(A, B)


        # Define types
        hbonds = {}
        for elA in hbond_elems:
            for elB in hbond_elems:
                hbonds[bname(elA, elB)] = []

        # First, grab the hydrogen atoms
        h_atoms = elem_inds(s, 'H')
        if len(h_atoms) == 0:
            # Nothing to do
            if save_info:
                s.info[HydrogenBonds.default_name] = hbonds
            return hbonds

        bond_atoms = []
        for el_b in hbond_elems:
            bond_atoms += elem_inds(s, el_b)

        if len(bond_atoms) < 2:
            if save_info:
                s.info[HydrogenBonds.default_name] = hbonds
            return hbonds

        # Now to pick the positions
        h_atoms_pos = s.get_positions()[h_atoms]
        bond_atoms_pos = s.get_positions()[bond_atoms]
        # Van der Waals radii length of H-atom bonds
        bonds_vdw = _vdw_radii[vdw_set][s.get_atomic_numbers()[bond_atoms]]
        bonds_vdw = np.where(np.isnan(bonds_vdw), default_vdw, bonds_vdw)
        bonds_vdw = (bonds_vdw+_vdw_radii[vdw_set][1])/2.0
        bonds_vdw *= vdw_scale

        # Now find the shortest and second shortest bonds for each H
        h_links = (h_atoms_pos[:, None, :]-bond_atoms_pos[None, :, :])
        shape = h_links.shape
        h_links, h_cells = minimum_periodic(h_links.reshape((-1, 3)),
                                            s.get_cell())
        h_links = h_links.reshape(shape)
        h_cells = h_cells.reshape(shape)
        h_links_norm = np.linalg.norm(h_links, axis=-1)

        # Now for each hydrogen: first and second closest
        h_closest = np.argsort(h_links_norm,
                               axis=-1)[:, :2]

        # Which ones DO actually form bonds?
        rngh = range(len(h_atoms))
        # Condition one: closest atom, A, is bonded
        h_bonded = h_links_norm[rngh,
                                h_closest[:, 0]] <= bonds_vdw[h_closest[:, 0]]
        # Condition two: furthest atom, B, is NOT bonded...
        h_bonded = np.logical_and(h_bonded,
                                  h_links_norm[rngh,
                                               h_closest[:, 1]
                                               ] > bonds_vdw[h_closest[:, 1]])
        # Condition three: ...but still closer to A than max_length
        links_ab = h_links[rngh, h_closest[:, 0]] - \
            h_links[rngh, h_closest[:, 1]]
        links_ab_norm = np.linalg.norm(links_ab, axis=-1)
        h_bonded = np.logical_and(h_bonded, links_ab_norm <= max_length)
        # Condition four: finally, the angle between AH and AB in A-H..B
        # must be smaller than max_angle
        angles_abah = np.sum(links_ab*h_links[rngh, h_closest[:, 0]], axis=-1)
        angles_abah /= links_ab_norm*h_links_norm[rngh, h_closest[:, 0]]
        angles_abah = np.arccos(angles_abah)*180.0/np.pi
        h_bonded = np.logical_and(h_bonded, angles_abah <= max_angle)

        # The survivors are actual h bonds!
        # Now on to defining them
        h_bonded = np.where(h_bonded)[0]
        if len(h_bonded) == 0:
            return hbonds

        # Moving to structure indices

        for h in h_bonded:

            ai, bi = h_closest[h]
            elA = s.get_chemical_symbols()[bond_atoms[ai]]
            elB = s.get_chemical_symbols()[bond_atoms[bi]]

            bond = {'H': h_atoms[h],
                    'A': (bond_atoms[ai], h_cells[h, ai]),
                    'B': (bond_atoms[bi], h_cells[h, bi]),
                    'length': links_ab_norm[h],
                    'angle': angles_abah[h]}

            btype = bname(elA, elB)
            hbonds[btype].append(bond)

        if save_info:
            s.info[HydrogenBonds.default_name] = hbonds

        return hbonds


class HydrogenBondsNumber(AtomsProperty):

    """
    HydrogenBondsNumber

    Number of hydrogen bonds detected in this system, classified by type.
    By default will use already existing hydrogen bonds if they're present as
    a saved array in the system.


    | Parameters:
    |   force_recalc (bool): if True, always recalculate the hydrogen bonds
    |                        even if already present.

    | Returns:
    |   hbonds_n (int): number of hydrogen bonds found

    """

    default_name = 'hydrogen_bonds_n'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    def extract(s, force_recalc):

        if not HydrogenBonds.default_name in s.info or force_recalc:
            hbonds = HydrogenBonds.get(s)
        else:
            hbonds = s.info[HydrogenBonds.default_name]

        hbonds_n = {}
        for hbt in hbonds:
            hbonds_n[hbt] = len(hbonds[hbt])

        return hbonds_n
