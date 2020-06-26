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
from scipy.linalg import null_space
from soprano.optional import requireSpglib


@requireSpglib('spglib')
def _get_symmetry_dataset(s, symprec, spglib=None):
    lattice = s.get_cell()
    positions = s.get_scaled_positions()
    numbers = s.get_atomic_numbers()
    symdata = spglib.get_symmetry_dataset((lattice, positions, numbers),
                                          symprec=symprec)
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
    under, and whether the Hessian has local radial symmetry or is
    definite in them."""

    dset = _get_symmetry_dataset(a, symprec)
    hno = dset['hall_number']

    # First, check the standard cell
    stdcell = dset['std_lattice']
    # And the transformation properties
    icell = np.linalg.inv(stdcell)
    ic2 = np.dot(icell.T, icell)
    evals, _ = np.linalg.eig(ic2)
    e0 = evals[0]
    is_iso = np.isclose(evals, e0).all()
    is_def = (np.sign(evals) == np.sign(e0)).all()

    # Operations in the non-transformed frame
    R, T = _get_symmetry_ops(hno)
    # Transformation
    P, o = dset['transformation_matrix'], dset['origin_shift']
    iP = np.linalg.inv(P)

    # Operations in the transformed frame
    rR = np.einsum('ij,ajk,kl', iP, R, P)
    rT = np.dot(T, iP.T) % 1

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

    invLoci = np.array(invLoci, dtype=object)

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
    wp0_ops = []
    wp_indices = []
    for i in w_indices:
        ops_inds = np.where(matching_ops[:, i])[0]
        loci = sorted(invLoci[ops_inds], key=lambda x: x[0])
        ltot = loci[0]
        for l in loci[1:]:
            if ltot[0] == 0:
                break
            ltot = _loci_intersect(ltot, l)
        if ltot[0] == 0:
            wp_indices.append(i)
            wp0_ops.append(zip(R[ops_inds], T[ops_inds]))
            wp_ops.append(zip(rR[ops_inds], rT[ops_inds]))

    # Find their positions in fractional coordinates
    wp_fxyz = (np.dot(wgrid[wp_indices]/24.0, iP.T)-np.dot(iP, o)) % 1
    # Remove any identical rows
    if wp_fxyz.shape[0] > 0:
        wp_fxyz, uinds = np.unique(np.round(wp_fxyz*24).astype(int), axis=0, return_index=True)
        wp_fxyz = wp_fxyz/24.0
    else:
        uinds = []
    wp0_ops = np.array(wp0_ops)[uinds]
    wp_ops = np.array(wp_ops)[uinds]

    isohess = [] # Here the meanings are:
    # 0 - no constraints
    # 1 - is positive/negative definite
    # 2 - is isotropic
    for ops in wp0_ops:
        wh = _wyckoff_isohess(ops)
        isohess.append(wh*(2*is_def+is_iso-1))
    isohess = np.array(isohess)

    return wp_fxyz, wp_ops, isohess


def _wyckoff_isohess(ops):
    """For each set of operations in wp_ops, representing a Wyckoff
    point that is symmetric under them, find whether the symmetry is
    such that any symmetric 3x3 tensor valued function ought to be 
    isotropic in that point"""

    """
    How it works:
     - For any operation, tensors must respect the following:

                    T = O'TO

        with O' meaning the transpose. This can be rewritten as

                    kron(O', O')t = t

       where t is a reshaped vector version of T. In other words,
       t has to belong to the nullspace of kron(O', O')-I.
     - While kron(O', O') is a 9x9 matrix, if we assume that T is
       symmetric, then it has 6 independent components. We want to
       prove that T is isotropic (T = xI, with x a scalar), which
       means we only worry about 5 independent components.
     - We can split T as such:

                    T = xI + S = xO'O + O'SO

       where S is a traceless symmetric tensor. Now, in what 
       conditions can we say that S has to be zero?
     - The first is that O'O = I. This is true of rotations
       and reflections, but not of all symmetry operations.
       If it's not true for any operation in wp_ops, then S
       can not be zero.
     - The second is that the intersection of the null
       spaces of P2kron(O',O')P1 is empty. Here P1 is a 9x5 
       matrix such that P1s = t-xi, where s is the vectorised
       form of S and i the same for I, and P2 is the inverse.

    Note that this has to hold for the operations expressed in an *orthogonal*
    basis. Otherwise it can fail. However, if it works in one basis, it's 
    valid in all of them. 
    """

    # Rotation/transformation matrices
    R = list(zip(*ops))[0]

    # First: check that all matrices are orthogonal
    ortho = all([np.isclose(np.dot(r.T, r), np.eye(3)).all() for r in R])

    if not ortho:
        return False

    # These matrices go back and fro to the 'reduced' form of the
    # unfolded symmetric traceless tensor (9 to only 5 components)

    P1 = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1],
                   [-1, 0, 0, -1, 0]])

    P2 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
    ])

    ns = np.eye(5)
    for r in R:
        r5 = np.linalg.multi_dot([P2, np.kron(r, r).T, P1])
        ns1 = null_space(r5-np.eye(5))
        ns = np.dot(ns, null_space(np.concatenate([ns, -ns1],
                                                  axis=1))[:ns.shape[1]])
        if ns.shape[1] == 0:
            return True
        ns /= np.linalg.norm(ns, axis=0)[None, :]

    return False
