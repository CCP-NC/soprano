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
from soprano.utils import minimum_periodic
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection
from soprano.properties.nmr.utils import _get_nmr_data


class DipolarCoupling(AtomsProperty):

    """
    DipolarCoupling

    Produces a dictionary of dipole-dipole coupling constants for atom pairs
    in the system. For each pair, the closest periodic copy will be considered.
    The constant for a pair of nuclei i and j is defined as:

               - \mu_0 \hbar \gamma_i \gamma_j
    d_{ij} =  ---------------------------------
                      8\pi^2 r_{ij}^3

    where the gammas represent the gyromagnetic ratios of the nuclei and the
    r is their distance. The full tensor of the interaction is then defined as

             |-d/2  0   0 |
             |            |
    D_{ij} = |  0 -d/2  0 |
             |            |
             |  0   0   d |

    where the z-axis is aligned with r_{ij} and the other two can be any
    directions in the orthogonal plane.

    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to compute the dipolar
    |                                   coupling. By default is None
    |                                   (= all of them).
|   |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
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

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings in Hz and r_{ij} versors,
    |                    by atomic index pair.


    """

    default_name = "dip_coupling"
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'isotopes': {},
        'isotope_list': None
    }

    @staticmethod
    def extract(s, sel_i, sel_j, isotopes, isotope_list):

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
        _nmr_data = _get_nmr_data()

        gammas = np.zeros(len(elems))
        for i, e in enumerate(elems):

            if e not in _nmr_data:
                # Non-existing element
                raise RuntimeError('No NMR data on element {0}'.format(e))

            iso = _nmr_data[e]['iso']
            if e in isotopes:
                iso = isotopes[e]
            if isotope_list is not None and isotope_list[i] is not None:
                iso = isotope_list[e]

            try:
                gammas[i] = _nmr_data[e][str(iso)]['gamma']
            except KeyError:
                raise RuntimeError('Isotope {0} does not exist for '
                                   'element {1}'.format(iso, e))

        # Viable pairs
        pairs = np.array([(i, j) for i in sel_i.indices
                          for j in sel_j.indices if i < j]).T
        pos = s.get_positions()

        r_ij = pos[pairs[1]] - pos[pairs[0]]
        # Reduce to NN
        r_ij, _ = minimum_periodic(r_ij, s.get_cell())
        # Distance
        R_ij = np.linalg.norm(r_ij, axis=1)
        # Versors
        v_ij = r_ij/R_ij[:, None]
        # Couplings
        d_ij = - (cnst.mu_0*cnst.hbar*gammas[pairs[0]]*gammas[pairs[1]] /
                  (8*np.pi**2*(R_ij*1e-10)**3))

        return {tuple(ij): [d_ij[l], v_ij[l]] for l, ij in enumerate(pairs.T)}
