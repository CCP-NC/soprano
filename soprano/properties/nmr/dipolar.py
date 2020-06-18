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

"""Implementation of AtomsProperties that relate to NMR dipole-dipole
couplings"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy import constants as cnst
from soprano.utils import minimum_periodic, minimum_supcell, supcell_gridgen
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection
from soprano.nmr.utils import _dip_constant
from soprano.data.nmr import _get_isotope_data


class DipolarCoupling(AtomsProperty):

    """
    DipolarCoupling

    Produces a dictionary of dipole-dipole coupling constants for atom pairs
    in the system. For each pair, the closest periodic copy will be considered.
    The constant for a pair of nuclei i and j is defined as:

    .. math::

        d_{ij} = -\\frac{\\mu_0\\hbar\\gamma_i\\gamma_j}{8\\pi^2r_{ij}^3}

    where the gammas represent the gyromagnetic ratios of the nuclei and the
    r is their distance. The full tensor of the interaction is then defined as

    .. math::

         D_{ij} = 
         \\begin{bmatrix}
          -\\frac{d_{ij}}{2} & 0 & 0 \\\\
          0 & -\\frac{d_{ij}}{2} & 0 \\\\
          0 & 0 & d_{ij} \\\\
         \\end{bmatrix}

    where the z-axis is aligned with :math:`r_{ij}` and the other two can be any
    directions in the orthogonal plane.

    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with its
    |                         own closest periodic copy. Otherwise excluded.
    |                         Default is False.
    |   block_size (int): maximum size of blocks used when processing large
    |                     chunks of pairs. Necessary to avoid memory problems
    |                     for very large systems. Default is 1000.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings in Hz and r_{ij} versors,
    |                    pointing from i to j, by atomic index pair.

    """

    default_name = 'dip_coupling'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'block_size': 1000,
    }

    @staticmethod
    def extract(s, sel_i, sel_j, isotopes, isotope_list, self_coupling,
                block_size):

        # Selections
        if sel_i is None:
            sel_i = AtomSelection.all(s)
        elif not isinstance(sel_i, AtomSelection):
            sel_i = AtomSelection(s, sel_i)

        if sel_j is None:
            sel_j = sel_i
        elif not isinstance(sel_j, AtomSelection):
            sel_j = AtomSelection(s, sel_j)

        # Find gammas
        elems = s.get_chemical_symbols()

        gammas = _get_isotope_data(elems, 'gamma', isotopes, isotope_list)

        # Viable pairs
        pairs = [(i, j) for i in sel_i.indices
                 for j in sel_j.indices]
        if not self_coupling:
            pairs = [p for p in pairs if p[0] != p[1]]

        pairs = np.array(pairs).T
        # Need to sort them and remove any duplicates, also take i < j as
        # convention
        pairs = np.array(list(zip(*set([tuple(x)
                                        for x in np.sort(pairs, axis=0).T]))))

        pos = s.get_positions()

        # Split this in blocks to make sure we don't clog the memory

        d_ij = np.zeros((0,))
        v_ij = np.zeros((0, 3))

        npairs = pairs.shape[1]

        for b_i in range(0, npairs, block_size):
            block = pairs.T[b_i:b_i+block_size]
            r_ij = pos[block[:, 1]] - pos[block[:, 0]]
            # Reduce to NN
            r_ij, _ = minimum_periodic(r_ij, s.get_cell(), exclude_self=True)
            # Distance
            R_ij = np.linalg.norm(r_ij, axis=1)
            # Versors
            v_ij = np.concatenate([v_ij, r_ij/R_ij[:, None]], axis=0)
            # Couplings
            d_ij = np.concatenate([d_ij,
                                   _dip_constant(R_ij*1e-10,
                                                 gammas[block[:, 0]],
                                                 gammas[block[:, 1]])
                                   ])

        return {tuple(ij): [d_ij[l], v_ij[l]] for l, ij in enumerate(pairs.T)}


class DipolarTensor(AtomsProperty):

    """
    DipolarTensor

    Produces a dictionary of dipole-dipole coupling tensors for atom pairs
    in the system. For each pair, the closest periodic copy will be considered.
    The coupling constant for a pair of nuclei i and j is defined as:

    .. math::

        d_{ij} = -\\frac{\\mu_0\\hbar\\gamma_i\\gamma_j}{8\\pi^2r_{ij}^3}

    where the gammas represent the gyromagnetic ratios of the nuclei and the
    r is their distance. The full tensor of the interaction is then defined as

    .. math::

         D_{ij} = d_{ij}\\frac{3\\hat{r}_{ij}\\otimes \\hat{r}_{ij}-\\mathbb{I}}{2}

    where :math:`\\hat{r}_{ij} = r_{ij}/|r_{ij}|` and the Kronecker product is
    used.

    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with its
    |                         own closest periodic copy. Otherwise excluded.
    |                         Default is False.
    |   block_size (int): maximum size of blocks used when processing large
    |                     chunks of pairs. Necessary to avoid memory problems
    |                     for very large systems. Default is 1000.

    | Returns: 
    |   dip_dict (dict): Dictionary of tensors in Hz by atomic index pair.

    """

    default_name = 'dip_coupling'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'block_size': 1000,
    }

    @staticmethod
    def extract(s, sel_i, sel_j, isotopes, isotope_list, self_coupling,
                block_size):

        dip_dict = DipolarCoupling.extract(s,
                                           sel_i=sel_i, sel_j=sel_j,
                                           isotopes=isotopes,
                                           isotope_list=isotope_list,
                                           self_coupling=self_coupling,
                                           block_size=block_size)

        # Now build the tensors
        tdict = {}
        for ij, (d, r) in dip_dict.items():
            tdict[ij] = d*(3*r[:, None]*r[None, :]-np.eye(3))/2

        return tdict


class DipolarDiagonal(AtomsProperty):

    """
    DipolarDiagonal

    Produces a dictionary of dipole-dipole tensors as eigenvalues and
    eigenvectors for atom pairs in the system. For each pair, the closest
    periodic copy will be considered.

    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling with the ones i sel_i. By
    |                                   default is None (= same as sel_i).
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with its
    |                         own closest periodic copy. Otherwise excluded.
    |                         Default is False.
    |   block_size (int): maximum size of blocks used when processing large
    |                     chunks of pairs. Necessary to avoid memory problems
    |                     for very large systems. Default is 1000.

    | Returns: 
    |   dip_tens_dict (dict): Dictionary of dipolar eigenvalues (in Hz) and
    |                         eigenvectors, by atomic index pair.

    """

    default_name = 'dip_diagonal'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'block_size': 1000
    }

    @staticmethod
    def extract(s, sel_i, sel_j, isotopes, isotope_list, self_coupling,
                block_size):

        # First, just get the values
        dip_dict = DipolarCoupling.extract(s, sel_i, sel_j,
                                           isotopes, isotope_list,
                                           self_coupling, block_size)

        # Now build the tensors
        dip_tens_dict = {}

        for ij, (d, v) in dip_dict.items():

            evals = np.array([-d/2, -d/2, d])
            # Eigenvectors
            evecs = np.zeros((3, 3))
            # Z is equal to v
            evecs[:, 2] = v
            # Y is any random orthogonal vector
            rv = np.random.random(3)
            evecs[:, 1] = np.cross(v, rv)
            evecs[:, 1] /= np.linalg.norm(evecs[:, 1])
            # X = Y cross Z
            evecs[:, 0] = np.cross(evecs[:, 1], v)

            dip_tens_dict[ij] = {'evals': evals, 'evecs': evecs}

        return dip_tens_dict


class DipolarRSS(AtomsProperty):

    """
    DipolarRSS

    Compute the Dipolar constant Root Sum Square for each atom in a system,
    including periodicity, within a cutoff.

    | Parameters:
    |   cutoff (float): cutoff radius in Angstroms at which the sum stops. By
    |                   default 5 Ang.
    |   isonuclear (bool): if True, only nuclei of the same species will be
    |                      considered. By default is False.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.

    | Returns:
    |   dip_rss (np.ndarray): dipolar constant RSS for each atom in the system

    """

    default_name = 'dip_rss'
    default_params = {
        'cutoff': 5.0,
        'isonuclear': False,
        'isotopes': {},
        'isotope_list': None,
    }

    @staticmethod
    def extract(s, cutoff, isonuclear, isotopes, isotope_list):

        # Supercell size
        scell_shape = minimum_supcell(cutoff, s.get_cell())
        _, scell = supcell_gridgen(s.get_cell(), scell_shape)

        pos = s.get_positions()
        elems = np.array(s.get_chemical_symbols())

        gammas = _get_isotope_data(elems, 'gamma', isotopes, isotope_list)

        dip_rss = []

        for i, el in enumerate(elems):

            # Distances?
            if not isonuclear:
                rij = pos.copy()
                gj = np.tile(gammas, len(scell))
            else:
                rij = pos[np.where(elems == el)]
                gj = gammas[i]
            rij = rij[None, :, :]+scell[:, None, :]-pos[i, None, None]
            Rij = np.linalg.norm(rij.reshape((-1, 3)), axis=-1)
            # Valid indices?
            ij = np.where((Rij > 0) & (Rij <= cutoff))
            Rij = Rij[ij]*1e-10
            try:
                gj = gj[ij]
            except IndexError:
                pass

            dip = _dip_constant(Rij, gammas[i], gj)
            dip_rss.append(np.sqrt(np.sum(dip**2)))

        return np.array(dip_rss)
