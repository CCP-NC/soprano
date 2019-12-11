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


def _loci_intersect(l1, l2):
    """Find the locus of points at the intersection of l1 and l2"""

    # Make sure l1 has the lowest dof
    l1, l2 = sorted([l1, l2], key=lambda x: x[0])

    if l1[0] == 0 or l2[0] == 3:
        return l1
    elif l1[0] == 1:
        d = np.dot(l1[1], l2[1])
        if l2[0] == 1:
            if abs(abs(d) - 1) < 1e-5:
                return (1, l1[1])
            else:
                return (0, None)
        else:
            if abs(d) > 1e-5:
                return (0, None)
            else:
                return (1, l1[1])
    elif l1[0] == 2:
        # l2 has to be a plane too
        v = np.cross(l1[1], l2[1])
        return (1, v/np.linalg.norm(v))


def _find_wyckoff_points(a, symprec=1e-5):
    """Find and return all Wyckoff points for a given atomic system,
    as well as the operations that each Wyckoff point is stable
    under"""

    dset = _get_symmetry_dataset(a, symprec)
    hno = dset['hall_number']

    # Operations in the non-transformed frame
    R, T = _get_symmetry_ops(hno)
    # Transformation
    P, o = dset['transformation_matrix'], dset['origin_shift']
    iP = np.linalg.inv(P)

    # Operations in the transformed frame
    rR = np.einsum('ij,ajk,kl', iP, R, P)
    rT = np.dot(T, iP.T)

    # Invariant loci for the operations
    invLoci = []
    for ro in R:
        evals, evecs = np.linalg.eig(ro)
        ev1 = np.isclose(evals, 1)
        dof = np.sum(ev1)
        if dof == 0:
            v = None
        elif dof == 1:
            i = np.where(ev1)[0][0]
            v = np.real(evecs[:, i])
        elif dof == 2:
            i = np.where(ev1)[0]
            v = np.real(np.cross(*evecs[:, i].T))
            v /= np.linalg.norm(v)
        elif dof == 3:
            v = None
        else:
            raise RuntimeError('Invalid symmetry operation')
        invLoci.append((dof, v))

    invLoci = np.array(invLoci)

    # A grid of candidate Wyckoff points: they always come in
    # fractions n/24
    wgrid = np.reshape(np.meshgrid(*[range(24)]*3, indexing='ij'),
                       (3, -1)).T

    # Identify the points that match under transformation and modulo 24
    matching_ops = np.all(np.isclose((np.tensordot(R, wgrid,
                                                   axes=(2, 1)) +
                                      T[:, :, None]*24) % 24,
                                     wgrid.T[None] % 24),
                          axis=1)
    # Indices of all matching points
    w_indices = np.where(np.any(matching_ops[1:], axis=0))[0]

    # Now for each find how many degrees of freedom there are
    wp_ops = []
    wp_indices = []
    for i in w_indices:
        ops_inds = np.where(matching_ops[:,i])[0]
        loci = sorted(invLoci[ops_inds], key=lambda x: x[0])
        ltot = loci[0]
        for l in loci[1:]:
            if ltot[0] == 0:
                break
            ltot = _loci_intersect(ltot, l)
        if ltot[0] == 0:
            wp_indices.append(i)
            wp_ops.append(list(zip(rR[ops_inds], rT[ops_inds])))

    # Find their positions in fractional coordinates
    wp_fxyz = (np.dot(wgrid[wp_indices]/24.0, iP.T)-np.dot(iP, o)) % 1
    # Remove any identical rows    
    wp_fxyz, uinds = np.unique(wp_fxyz, axis=0, return_index=True)
    wp_ops = np.array(wp_ops)[uinds]

    return wp_fxyz, wp_ops
