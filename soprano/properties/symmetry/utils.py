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

"""Utility functions for symmetry calculations"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from soprano.optional import requireSpglib


@requireSpglib('spglib')
def _get_symmetry_dataset(s, symprec, spglib=None):
    symdata = spglib.get_symmetry_dataset(s, symprec=symprec)
    return symdata


@requireSpglib('spglib')
def _get_symmetry_ops(hall_no, spglib=None):
    symdata = spglib.get_symmetry_from_database(hall_no)
    return symdata['rotations'], symdata['translations']


def _find_wyckoff(a, symprec=1e-5):
    """Find and return all Wyckoff points for a given atomic system,
    as well as the operations that each Wyckoff point is stable
    under"""

    dset = _get_symmetry_dataset(a, symprec)

    R, T = dset['rotations'], dset['translations']
    P, o = dset['transformation_matrix'], dset['origin_shift']
    iP = np.linalg.inv(P)

    rR = np.einsum('ij,ajk,kl', P, R, iP)
    rT = np.dot(T, P.T)

    # A grid of candidate Wyckoff points: they always come in
    # fractions n/24
    wgrid = np.reshape(np.meshgrid(*[range(24)]*3, indexing='ij'),
                       (3, -1)).T

    # Identify the points that match under transformation and modulo 24
    matching_ops = np.all(np.isclose((np.tensordot(rR, (wgrid-o),
                                                   axes=(2, 1)) +
                                      rT[:, :, None]*24) % 24,
                                     (wgrid.T[None]-o[None, :, None]) % 24),
                          axis=1)

    # Find the Wyckoff points
    wp_indices = np.where(np.any(matching_ops[1:], axis=0))[0]
    # Find their positions in fractional coordinates
    wp_fxyz = (np.dot(wgrid[wp_indices]/24.0, iP.T)-np.dot(iP, o))%1
    # And the corresponding operations
    wp_ops = []
    for i in wp_indices:
        ops_inds = np.where(matching_ops[:,i])[0]
        wp_ops.append([(R[j], T[j]) for j in ops_inds])

    return wp_indices, wp_fxyz, wp_ops
