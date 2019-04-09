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

"""Implementation of AtomsProperties that relate to bond order parameters"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.special import sph_harm
from soprano.utils import wigner_3j, minimum_periodic
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection


def _valid_triples(l):
    """Valid triples for angular momentum l for which m1 + m2 + m3 = 0"""

    # m1 can be anything
    m1 = np.arange(-l, l+1)
    # m2 can be anything such that |m1+m2| <= l
    m2 = [np.arange(-l-min(m, 0), l+1-max(m, 0)) for m in m1]

    return np.array([[m, mm, -m-mm] for i, m in enumerate(m1)
                     for mm in m2[i]])


def _steinhardt_pars(vecs, l_channels, compute_W=False, weights=None):
    """Compute the Steinhardt bond order parameters (Q and optionally W)"""

    Qlm = []
    Q = np.zeros(len(l_channels))
    W = np.zeros(len(l_channels))

    # Translate vectors in polar form
    vecs = np.array(vecs)/np.linalg.norm(vecs, axis=1)[:, None]
    pols = np.array([np.arccos(vecs[:, 2]),
                     np.arctan2(vecs[:, 1], vecs[:, 0])]).T

    # What about weights?
    if weights is None:
        weights = np.ones(vecs.shape[0])/len(vecs)

    for i, l in enumerate(l_channels):
        mvals = np.arange(-l, l+1)
        Qlm.append(np.sum(sph_harm(mvals[None, :], l, pols[:, 1, None],
                                   pols[:, 0, None])*weights[:, None], axis=0))
        Qsum = np.sum(np.conj(Qlm[i])*Qlm[i])
        Q[i] = np.abs(np.sqrt(Qsum*4*np.pi/(2*l+1.0)))
        if compute_W:
            m1, m2, m3 = _valid_triples(l).T
            ls = np.array([l]*len(m1))
            w3j = wigner_3j(ls, m1, ls, m2, ls, m3)
            W[i] = np.abs(np.sum(w3j*Qlm[i][m1+l]*Qlm[i][m2+l]*Qlm[i][m3+l]
                                 )/(Qsum)**1.5)

    if not compute_W:
        return Q
    else:
        return Q, W


class BondOrder(AtomsProperty):

    """
    BondOrder

    Computes the order parameters defined by Steinhardt et al. [PRB 28, 784
    (1983)] for an atom or group of atoms in the system. Here for each atom
    all closest periodic copies of the others are used; a sigmoidal cutoff is
    used to reduce the impact of those beyond a certain distance while keeping
    the behaviour smooth upon small fluctuations.

    | Parameters:
    |   l_channels (list[int]): list of angular momentum channels to be used
    |                           for the calculation. Must be integers >= 1.
    |                           Default is all channels 1 to 10.
    |   center_atoms (AtomSelection): AtomSelection, integer index or list of
    |                                 indices of all atoms around which the
    |                                 bond order parameters are to be computed
    |                                 and summed over. Default is None, which
    |                                 means all atoms of the system.
    |   environment_atoms (AtomSelection): AtomSelection, integer index or
    |                                      list of indices of all atoms that
    |                                      used as environment to compute the
    |                                      bond order parameters. For each
    |                                      'center' atom, all its links to
    |                                      'environment' atoms other than
    |                                      itself will be considered. Default
    |                                      is None, which means all atoms of
    |                                      the system.
    |   cutoff_radius (float): radius at which to cut the smoothing sigmoid
    |                          function, in Angstroms. Default is 2.0.
    |   cutoff_width (float): width parameter of the sigmoid, controlling the
    |                         steepness with which it falls to zero, in 
    |                         Angstroms. Default is 0.05.
    |   compute_W (bool): whether to also compute the W parameter (third
    |                     order) besides the Q (second order).

    | Returns:
    |   bond_order_pars (dict): dictionary containing arrays with Q and, if
    |                           required, W for each l in l_channels.

    """

    default_name = 'bond_order'
    default_params = {
        'l_channels': np.arange(1, 11),
        'center_atoms': None,
        'environment_atoms': None,
        'cutoff_radius': 2.0,
        'cutoff_width': 0.05,
        'compute_W': False
    }

    @staticmethod
    def extract(s, l_channels, center_atoms, environment_atoms, cutoff_radius,
                cutoff_width, compute_W):

        # Check that l_channels are valid
        l_channels = np.array(l_channels)
        if ((l_channels < 1) | ((l_channels % 1) != 0)).any():
            raise ValueError('Invalid angular momentum channels')

        # Turn center_atoms and environment_atoms into AtomSelections
        if not isinstance(center_atoms, AtomSelection):
            if center_atoms is None:
                center_atoms = range(len(s))
            elif not isinstance(center_atoms, list):
                center_atoms = [center_atoms]
            center_atoms = AtomSelection(s, center_atoms)

        if not isinstance(environment_atoms, AtomSelection):
            if environment_atoms is None:
                environment_atoms = range(len(s))
            elif not isinstance(environment_atoms, list):
                environment_atoms = [environment_atoms]
            environment_atoms = AtomSelection(s, environment_atoms)

        xyz = s.get_positions()

        # Now to build the total list of vectors and weights
        all_vecs = np.zeros((0, 3))
        all_weights = np.zeros((0,))

        for ci in center_atoms.indices:
            # What would the environment be?
            env_inds = list(set(environment_atoms.indices) - set([ci]))
            # So what are the 'bonds'?
            bonds = minimum_periodic(xyz[env_inds] - xyz[ci], s.get_cell())[0]
            bnorms = np.linalg.norm(bonds, axis=1)

            # Compute sigmoid weights
            bweights = 0.5*(((cutoff_radius-bnorms)/cutoff_width) /
                            (((cutoff_radius-bnorms)/cutoff_width)**2+1)**0.5
                            + 1)
            bweights /= np.sum(bweights)  # Norm to 1

            all_vecs = np.concatenate((all_vecs, bonds))
            all_weights = np.concatenate((all_weights, bweights))

        # Norm to 1 all weights:
        all_weights /= len(center_atoms.indices)

        # Now compute the order parameters!
        stp = _steinhardt_pars(all_vecs, l_channels, weights=all_weights,
                               compute_W=compute_W)

        if compute_W:
            return {'Q': stp[0], 'W': stp[1]}
        else:
            return {'Q': stp}
