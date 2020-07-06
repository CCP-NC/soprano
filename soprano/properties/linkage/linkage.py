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

"""Implementation of AtomsProperties that relate to linkage of atoms"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import pkgutil
import numpy as np
from ase.data import atomic_numbers
from ase.quaternions import Quaternion
from soprano.selection import AtomSelection
from soprano.properties import AtomsProperty
from soprano.utils import (swing_twist_decomp, is_string, graph_specsort,
                           minimum_periodic, all_periodic, get_bonding_graph)
from soprano.data import vdw_radii


def _compute_bonds(s, vdw_set, vdw_scale=1.0, default_vdw=2.0, vdw_custom={}):
    """Convenience function that covers the core of bond computation"""

    # First, we need the biggest Van der Waals radius
    # So that we know how big the supercell needs to be

    # Build a custom VdW set
    vdw_r = np.array(vdw_radii[vdw_set])*vdw_scale
    vdw_r = np.where(np.isnan(vdw_r), default_vdw, vdw_r)
    for el, r in vdw_custom.items():
        vdw_r[atomic_numbers[el]] = r

    vdw_vals = vdw_r[s.get_atomic_numbers()]
    vdw_max = max(vdw_vals)

    # Get the interatomic pair distances
    atomn = len(s)
    triui = np.triu_indices(atomn, k=1)
    v = s.get_positions()
    v = (v[:, None, :]-v[None, :, :])[triui]
    # Reduce them
    v, v_i, v_cells = all_periodic(v, s.get_cell(), vdw_max, pbc=s.get_pbc())
    v = np.linalg.norm(v, axis=-1)

    # Now distance and VdW matrices
    vdw_M = ((vdw_vals[None, :]+vdw_vals[:, None])/2.0)[triui]
    link_M = v <= vdw_M[v_i]
    linked = np.where(link_M)

    return linked, triui, v, v_i, v_cells


class LinkageList(AtomsProperty):

    """
    LinkageList

    Produces an array containing the atomic pair distances in a system,
    reduced to their shortest periodic version and sorted min to max.

    | Parameters:
    |   size (int): maximum number of distances to include. If not present,
    |               all of them will be included. If present, arrays will be
    |               cut or padded to reach this size.
    |   return_pairs (bool): if True, return the pairs of atoms to which the
    |                        distances correspond, as a list of tuples of
    |                        indices.

    | Returns:
    |   link_list ([float]): sorted list of interatomic linkage distances
    |   pair_list ([(int, int)]): only if return_pairs is True, list of pairs
    |                             corresponding to the distances

    """

    default_name = 'linkage_list'
    default_params = {
        'size': 0,
        'return_pairs': False
    }

    @staticmethod
    def extract(s, size, return_pairs):
        # Get the interatomic pair distances
        v = s.get_positions()
        v = v[:, None, :]-v[None, :, :]
        pair_inds = np.triu_indices(v.shape[0], k=1)
        v = v[pair_inds]
        # Reduce them
        v, _ = minimum_periodic(v, s.get_cell())
        # And now compile the list
        link_list = np.linalg.norm(v, axis=-1)
        sort_i = np.argsort(link_list)
        link_list = link_list[sort_i]
        if size > 0:
            if link_list.shape[0] >= size:
                link_list = link_list[:size]
            else:
                link_list = np.pad(link_list,
                                   (0, size-link_list.shape[0]),
                                   mode=str('constant'),
                                   constant_values=np.inf)

        if not return_pairs:
            return link_list
        else:
            pairs = list(zip(pair_inds[0][sort_i], pair_inds[1][sort_i]))
            return link_list, pairs


class Bonds(AtomsProperty):

    """
    Bonds

    Produces an array of tuples identifying all bonds existing within the
    system (calculated using Van der Waals radii). The tuples are structured
    as:

    (atom_1, atom_2, atom_2_cell, bond_length)

    with atom_1 and atom_2 being indices and atom_2_cell being an array of
    integers identifying the unit cell to which atom_2 belongs with respect
    to atom_1 (which is assumed to be in (0,0,0), the central cell). This is
    to account for the possibility of course that the bond exists through the
    periodic boundary. WARNING: the possibility of an atom bonding with
    another throughout two different periodic boundaries is not accounted for.

    | Parameters:
    |   vdw_set({ase, jmol, csd}): set of Van der Waals radii to use. Default 
    |                              is csd [S. Alvarez, 2013].
    |   vdw_scale (float): scaling factor to apply to the base Van der Waals
    |                      radii values. Values bigger than one make for more
    |                      tolerant bonds.
    |   default_vdw (float): default Van der Waals radius for species for
    |                        whom no data is available.
    |   vdw_custom (dict): a dictionary of custom Van der Waals radii to use,
    |                      overriding the existing ones, expressed as
    |                      {symbol: radius}.
    |   return_matrix (bool): if True, also return an NxN bonding matrix for
    |                         all N atoms in the system
    |   save_info (bool): if True, save the found bonds (and in case matrix)
    |                     as part of the Atoms object info. By default True.

    | Returns:
    |   bonds([tuple]): list of bonds in the form of 3-tuples structured as
    |                   explained above

    """

    default_name = 'bonds'
    default_params = {
        'vdw_set': 'csd',
        'vdw_scale': 1.0,
        'default_vdw': 2.0,
        'vdw_custom': {},
        'return_matrix': False,
        'save_info': True,
    }

    @staticmethod
    def extract(s, vdw_set, vdw_scale, default_vdw, vdw_custom, return_matrix,
                save_info):

        linked, triui, v, v_i, v_cells = _compute_bonds(s,
                                                        vdw_set,
                                                        vdw_scale,
                                                        default_vdw,
                                                        vdw_custom)

        bonds = list(zip(triui[0][v_i[linked]], triui[1][v_i[linked]],
                         -v_cells[linked], v[linked]))

        if save_info:
            s.info[Bonds.default_name] = list(bonds)

        if not return_matrix:
            return list(bonds)  # For Python 3 compatibility
        else:
            bmat = np.zeros((len(s), len(s))).astype(int)
            bmat[triui[0][v_i[linked]], triui[1][v_i[linked]]] = 1
            bmat[triui[1][v_i[linked]], triui[0][v_i[linked]]] = 1

            if save_info:
                s.info[Bonds.default_name + '_matrix'] = bmat

            return list(bonds), bmat


class CoordinationHistogram(AtomsProperty):

    """
    CoordinationHistogram

    Produces an histogram representing, for each pair of species present in
    the system, how many atoms of species 1 have n bonds with species 2, n
    being the histogram bins. The histogram is topped at a 'maximum
    coordination' parameter which is 6 by default but can be user defined;
    the last bin represents all higher values (so by default '6 or more').
    Two species or lists of species can be given if one wants to restrict the
    search; otherwise a full histogram for all pairs of species is returned.

    | Parameters:
    |   vdw_set({ase, jmol, csd}): set of Van der Waals radii to use. Default 
    |                              is csd [S. Alvarez, 2013].
    |   vdw_scale (float): scaling factor to apply to the base Van der Waals
    |                      radii values. Values bigger than one make for more
    |                      tolerant bonds.
    |   default_vdw (float): default Van der Waals radius for species for
    |                        whom no data is available.
    |   vdw_custom (dict): a dictionary of custom Van der Waals radii to use,
    |                      overriding the existing ones, expressed as
    |                      {symbol: radius}.
    |   species_1 (str or [str]): list of species to compute the histogram
    |                             for. By default all of them.
    |   species_2 (str or [str]): list of species whose coordination with
    |                             species_1 should be checked. By default all
    |                             of them.
    |   max_coord (int): what should be the largest coordination number
    |                    considered for an atom (default 6).

    | Returns:
    |   coord_hist (dict): dictionary of dictionaries indexed by species_1
    |                      followed by species_2. The elements are arrays of
    |                      integers constituting the histogram.

    """

    default_name = 'coord_histogram'
    default_params = {
        'vdw_set': 'csd',
        'vdw_scale': 1.0,
        'default_vdw': 2.0,
        'vdw_custom': {},
        'species_1': None,
        'species_2': None,
        'max_coord': 6
    }

    @staticmethod
    def extract(s, vdw_set, vdw_scale, default_vdw, vdw_custom,
                species_1, species_2, max_coord):

        elems = np.array(s.get_chemical_symbols())

        # Get the bonds
        bond_calc = Bonds(vdw_set=vdw_set,
                          vdw_scale=vdw_scale,
                          default_vdw=default_vdw,
                          vdw_custom=vdw_custom)
        bonds = bond_calc(s)
        # What if there are none?
        if len(bonds) == 0:
            # Just return
            print('WARNING: no bonds detected for CoordinationHistogram')
            return {}
        bond_inds = np.concatenate(list(zip(*bonds))[:2])
        bond_elems = elems[bond_inds]
        bN = len(bonds)

        if species_1 is None:
            species_1 = np.unique(elems)
        elif is_string(species_1):
            species_1 = np.array([species_1])

        if species_2 is None:
            species_2 = np.unique(elems)
        elif is_string(species_2):
            species_2 = np.array([species_2])

        # Initialise the histogram
        hist = {s1: {s2: np.zeros(max_coord+1)
                     for s2 in species_2}
                for s1 in species_1}

        for s1 in species_1:
            # Which atoms are of species 1, and what are they bonded to?
            i1 = np.where(bond_elems == s1)[0]
            b1 = bond_inds[i1]
            be1 = bond_elems[(i1-bN).astype(int)]
            for s2 in species_2:
                # Which ones are bonded to species 2?
                i2 = np.where(be1 == s2)
                b2 = b1[i2]
                b2, counts = np.unique(b2, return_counts=True)
                hist_i, hist_n = np.unique(counts, return_counts=True)
                # Fix for numbers that are too high...
                hist_big = np.where(hist_i > max_coord)[0]
                if (len(hist_big) > 0):
                    # In this case find the max_coord index, if absent add it
                    hist_maxc = np.where(hist_i == max_coord)[0]
                    if len(hist_maxc) == 0:
                        hist_i = np.concatenate([hist_i, [max_coord]])
                        hist_n = np.concatenate([hist_n, [0]])
                        hist_maxc = [-1]
                    hist_n[hist_maxc] += np.sum(hist_n[hist_big])
                    # Then slice away, keep only the admissible indices
                    hist_small = np.where(hist_i <= max_coord)[0]
                    hist_i = hist_i[hist_small]
                    hist_n = hist_n[hist_small]
                hist[s1][s2][hist_i] += hist_n

        return hist


class Molecules(AtomsProperty):

    """
    Molecules

    Produces an array containing multiple AtomSelection objects representing
    molecules in the system as found by connecting atoms closer than the half
    sum of their Van der Waals radii. It will return the entire unit cell if
    the system can not be split in molecules at all.

    | Parameters:
    |   vdw_set({ase, jmol, csd}): set of Van der Waals radii to use. Default 
    |                              is csd [S. Alvarez, 2013].
    |   vdw_scale (float): scaling factor to apply to the base Van der Waals
    |                      radii values. Values bigger than one make for more
    |                      tolerant molecules.
    |   default_vdw (float): default Van der Waals radius for species for
    |                        whom no data is available.
    |   vdw_custom (dict): a dictionary of custom Van der Waals radii to use,
    |                      overriding the existing ones, expressed as
    |                      {symbol: radius}.
    |   save_info (bool): if True, save the found molecules as part of the
    |                     Atoms object info. By default True.

    | Returns:
    |   molecules ([AtomSelection]): list of molecules in the form of
    |                                AtomSelection objects.

    """

    default_name = 'molecules'
    default_params = {
        'vdw_set': 'csd',
        'vdw_scale': 1.0,
        'default_vdw': 2.0,
        'vdw_custom': {},
        'save_info': True,
    }

    @staticmethod
    def extract(s, vdw_set, vdw_scale, default_vdw, vdw_custom, save_info):

        N = len(s)
        # Sanity check
        if N < 2:
            # WTF?
            print('WARNING: impossible to calculate molecules on single-atom '
                  'system')
            return None

        # Get the bonds
        bond_calc = Bonds(vdw_set=vdw_set,
                          vdw_scale=vdw_scale,
                          default_vdw=default_vdw,
                          vdw_custom=vdw_custom)
        bonds = bond_calc(s)

        mol_sets = []
        unsorted_atoms = list(range(N))

        def get_linked(i):
            i_bonds = filter(lambda b: i in b[:2], bonds)
            links = map(lambda b: (b[1], b[2]) if b[0] == i else (b[0], -b[2]),
                        i_bonds)
            return links

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
                links = get_linked(a1)
                for l, cl in links:
                    if l in unsorted_atoms:
                        mol_queue.append((l, cell1 + cl))
                        unsorted_atoms.remove(l)
                    current_mol_bonds[-1].append(l)

            mol_sets.append((current_mol, current_mol_cells,
                             current_mol_bonds))

        mols = []
        for m_i, m_cells, m_bonds in mol_sets:
            mols.append(AtomSelection(s, m_i))
            mols[-1].set_array('cell_indices', m_cells)
            # This is necessary to guarantee shape consistency
            m_barr = np.empty((len(m_bonds),), dtype=list)
            for i, m_b in enumerate(m_bonds):
                m_barr[i] = m_b
            mols[-1].set_array('bonds', m_barr)

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

        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        return len(s.info[Molecules.default_name])


class MoleculeMass(AtomsProperty):

    """
    MoleculeMass

    Total mass of each of the molecules detected in this system. By default
    will use already existing molecules if they're present as a saved array in
    the system.


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

        if Molecules.default_name not in s.info or force_recalc:
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


class MoleculeCOM(AtomsProperty):

    """
    MoleculeCOM

    List of centers of mass for the molecules present in the system. By
    default will use already existing molecules if they're present as a saved
    array in the system.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Return:
    |   mol_com (np.ndarray): list of centers of mass for the system's
    |                         molecules

    """

    default_name = 'molecule_com'
    default_params = {
        'force_recalc': False,
    }

    @staticmethod
    def extract(s, force_recalc):

        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        mol_com = []
        all_m = s.get_masses()
        all_pos = s.get_positions()

        for mol in s.info[Molecules.default_name]:

            mol_pos = all_pos[mol.indices]
            mol_pos += np.tensordot(mol.get_array('cell_indices'),
                                    s.get_cell(),
                                    axes=(1, 0))
            mol_ms = all_m[mol.indices]
            mol_com.append(np.sum(mol_pos*mol_ms[:, None],
                                  axis=0)/np.sum(mol_ms))

        return np.array(mol_com)


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

        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        mol_com = MoleculeCOM.get(s)

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


class MoleculeQuaternion(AtomsProperty):

    """
    MoleculeQuaternion

    A list of quaternions expressing the rotation of the molecule's intertia
    tensor principal frame with respect to the cartesian axes.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Returns:
    |   mol_quat ([ase.Quaternion]): list of quaternions

    """

    default_name = 'molecule_quaternion'
    default_params = {
        'force_recalc': False,
    }

    @staticmethod
    def extract(s, force_recalc):

        if Molecules.default_name not in s.info or force_recalc:
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
            mol_quat.append(quat)

        return mol_quat


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
    |               cut or padded to reach this size.
    |   twist_axis ([float]): if present, only compare the Twist component of
    |                         quaternion along the given axis. The Twist/Swing
    |                         decomposition splits a quaternion in a rotation
    |                         around an axis and one around an orthogonal
    |                         direction. Only one between this and swing_plane
    |                         can be present.
    |   swing_plane ([float]): if present, only compare the Swing component of
    |                         quaternion along the given axis. The Twist/Swing
    |                         decomposition splits a quaternion in a rotation
    |                         around an axis and one around an orthogonal
    |                         direction. Only one between this and twist_axis
    |                         can be present.

    | Returns:
    |   molecule_relrot ([float]): list of relative rotations, as quaternion
    |                              distances, with the required ordering.

    """

    default_name = 'molecule_rel_rotation'
    default_params = {
        'force_recalc': False,
        'size': 0,
        'swing_plane': None,
        'twist_axis': None,
    }

    @staticmethod
    def extract(s, force_recalc, size, swing_plane, twist_axis):

        # Sanity check
        if swing_plane is not None and twist_axis is not None:
            raise RuntimeError('Only one between swing_plane and twist_axis '
                               'can be passed to MoleculeRelativeRotation')

        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        mol_quat = [q.q for q in MoleculeQuaternion.get(s)]

        for i, quat in enumerate(mol_quat):
            if swing_plane is not None:
                mol_quat[i], dummy = swing_twist_decomp(quat, swing_plane)
            elif twist_axis is not None:
                dummy, mol_quat[i] = swing_twist_decomp(quat, twist_axis)

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


class MoleculeSpectralSort(AtomsProperty):

    """
    MoleculeSpectralSort

    Reorder molecules to have their indices sorted using a spectral 
    sorting method based on the Fiedler vector of their bonding graph. This
    sorting should be equivalent for equivalent molecules - except for the
    arbitrary ordering of equivalent atoms.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.

    | Returns:
    |   mol_specsort ([np.ndarray]): list of Molecules with indices sorted by
                                     spectral method.

    """

    default_name = 'molecule_specsort'
    default_params = {
        'force_recalc': False,
    }

    @staticmethod
    def extract(s, force_recalc):

        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        mol_specsort = []

        for mol in s.info[Molecules.default_name]:
            # Start by getting the adjacency and degree matrices
            N = len(mol)
            bonds = mol.get_array('bonds')
            A = np.zeros((N, N))
            D = A.copy()
            for i, b in enumerate(bonds):
                D[i, i] = len(b)
                A[i, [list(mol.indices).index(j) for j in b]] = 1

            L = D - A

            sort = graph_specsort(L)

            mol = mol[sort]
            mol_specsort.append(mol)

        return mol_specsort


class HydrogenBonds(AtomsProperty):

    """
    Hydrogen Bonds

    Produces a dictionary containing the atom indices defining hydrogen bonds
    detected in the system - if required, classified by type. By default only
    O and N atoms are considered for hydrogen bonds (OH..O, OH..N and so on).
    The type is defined as AH..B where A is the symbol of the atom directly
    bonded to the proton and B the one of the hydrogen bonded one.

    | Parameters:
    |   vdw_set({ase, jmol, csd}): set of Van der Waals radii to use. Default 
    |                              is csd [S. Alvarez, 2013].
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
        'vdw_set': 'csd',
        'vdw_scale': 1.0,
        'default_vdw': 2.0,
        'vdw_custom': {},
        'hbond_elems': ['O', 'N'],
        'max_length': 3.5,
        'max_angle': 45.0,
        'save_info': True
    }

    @staticmethod
    def extract(s, vdw_set, vdw_scale, default_vdw, vdw_custom, hbond_elems,
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

        vdw_r = np.array(vdw_radii[vdw_set])*vdw_scale
        vdw_r = np.where(np.isnan(vdw_r), default_vdw, vdw_r)
        for el, r in vdw_custom.items():
            vdw_r[atomic_numbers[el]] = r

        bonds_vdw = vdw_r[s.get_atomic_numbers()[bond_atoms]]
        bonds_vdw = (bonds_vdw+vdw_r[1])/2.0

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
            if save_info:
                s.info[HydrogenBonds.default_name] = hbonds
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

        if HydrogenBonds.default_name not in s.info or force_recalc:
            hbonds = HydrogenBonds.get(s)
        else:
            hbonds = s.info[HydrogenBonds.default_name]

        hbonds_n = {}
        for hbt in hbonds:
            hbonds_n[hbt] = len(hbonds[hbt])

        return hbonds_n


class DihedralAngleList(AtomsProperty):

    """
    DihedralAngleList

    Produces a list of dihedral angles found in the system, identified by
    looking for a bonding pattern. The amount of said angles can vary from
    zero (if the pattern is not present) to an arbitrary number. They will be
    returned sorted from lowest to highest. Periodic boundary conditions are
    taken into account. If no pattern is provided, HCCH groups are searched by
    default.

    | Parameters:
    |   dihedral_pattern ([str]*4): a list of four chemical symbols
    |                               identifying the dihedral pattern to look
    |                               for
    |   bonds_params (dict): parameters to pass to the Bonds property used to
    |                        compute bonds. See the Bonds docstring for
    |                        details. If not provided, defaults are used

    | Returns:
    |   dihedral_angles (np.ndarray): sorted list of dihedral angles found

    """

    default_name = 'dihedral_angle_list'
    default_params = {
        'dihedral_pattern': ['H', 'C', 'C', 'H'],
        'bonds_params': {}
    }

    @staticmethod
    def extract(s, dihedral_pattern, bonds_params):

        # First, compute bonds
        bonds = Bonds(**bonds_params)
        bonds = bonds(s)
        # Chemical symbols
        elems = np.array(s.get_chemical_symbols())

        # Build a network
        dp = np.array(dihedral_pattern)
        nodes = np.where(np.any(elems[:, None] == dp[None, :],
                                axis=1))[0]

        bond_table = {n: [] for n in nodes}
        for b in bonds:
            if np.all(np.any(nodes[:, None] == b[:2], axis=0)):
                bond_table[b[0]].append((b[1], b[2], b[3]))
                bond_table[b[1]].append((b[0], -b[2], b[3]))

        # Now prepare a series of 'pointers' to traverse the network
        pointers = np.where(elems == dihedral_pattern[0])[0]
        pointer_memory = [[] for p in pointers]
        pointer_traversal = np.zeros((len(pointers), 1, 3))
        for el in dihedral_pattern[1:]:
            new_pointers = []
            new_pointer_memory = []
            new_pointer_traversal = []
            for p_i, p in enumerate(pointers):
                # Is it bonded to anything with that element?
                # Did we visit it already?
                for b in bond_table[p]:
                    if elems[b[0]] == el and b[0] not in pointer_memory[p_i]:
                        # If so, spawn new pointers and save the corresponding
                        # memory of the previous path
                        new_pointers.append(b[0])
                        new_pointer_memory.append(
                            list(pointer_memory[p_i]) + [p])
                        new_pointer_traversal.append(
                            np.concatenate((pointer_traversal[p_i],
                                            np.array(b[1])[None, :])))
            pointers = new_pointers
            pointer_memory = new_pointer_memory
            pointer_traversal = np.array(new_pointer_traversal)

        # Now build the dihedra array
        dihedra = np.array([pointer_memory[i] + [p]
                            for i, p in enumerate(pointers)])

        # Symmetry check, epurate the repeated ones if needed
        is_symm = np.all(dihedral_pattern == dihedral_pattern[::-1])

        if is_symm:
            duplicate = []
            for i, d1 in enumerate(dihedra):
                for j, d2 in enumerate(dihedra[i+1:]):
                    if np.all(d2 == d1[::-1]):
                        duplicate.append(j+i+1)
            unique = list(set(range(len(dihedra))) - set(duplicate))
            dihedra = dihedra[unique]
            pointer_traversal = pointer_traversal[unique]

        # If it's empty we may quit now
        if len(dihedra) == 0:
            return np.array([])
        # Cumulate traversals
        pointer_traversal = np.cumsum(pointer_traversal, axis=1)

        # Now that we have them let's find the positions
        posv = s.get_positions()[dihedra.reshape((-1,))]
        posv += np.dot(pointer_traversal.reshape((-1, 3)), s.get_cell())

        # And finally the dihedral angles
        links = np.diff(posv.reshape((-1, 4, 3)), axis=1)
        bxa = np.cross(links[:, 1, :], links[:, 0, :])
        bxa /= np.linalg.norm(bxa, axis=-1)[:, None]
        cxb = np.cross(links[:, 2, :], links[:, 1, :])
        cxb /= np.linalg.norm(cxb, axis=-1)[:, None]
        angles = np.arccos(np.clip(np.sum(bxa*cxb, axis=-1), -1, 1))
        angles = np.where(np.sum(bxa*links[:, 2, :], axis=-1) > 0,
                          2*np.pi-angles, angles)

        # And return!
        return angles


class BondGraph(AtomsProperty):
    """
    BondGraph

    Bond graph returns a networkx graph of the molecules in the structure. To
    use this property you must have the networkx library installed.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the bond graph
    |                        even if already present.
    |   save_info (bool): if True, save the bond graph as part of the Atoms
    |                     object info. By default True.

    | Returns:
    |   graph (nx.Graph): the bond graph for the structure

    """

    default_name = 'bond_graph'
    default_params = {
        'force_recalc': False,
        'save_info': True,
    }

    @staticmethod
    def extract(s, force_recalc, save_info):
        if BondGraph.default_name not in s.info or force_recalc:
            bprop = Bonds(return_matrix=True)
            _, bond_matrix = bprop(s)
            graph = get_bonding_graph(bond_matrix)
        else:
            graph = s.info[BondGraph.default_name]

        if save_info:
            s.info[BondGraph.default_name] = graph

        return graph
