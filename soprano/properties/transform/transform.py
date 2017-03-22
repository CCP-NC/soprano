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

"""Implementation of AtomsProperties that transform the instance in some
way"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from ase import Atoms
from ase.quaternions import Quaternion
from soprano.utils import periodic_center
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection


def _transform_sel_check(extrfunc):

    def decorated_extrfunc(s, selection, **kwargs):

        # Perform basic checks on selection
        if selection is None:
            selection = AtomSelection.all(s)
        elif not selection.validate(s):
            raise ValueError('Selection passed to transform does not apply to'
                             ' system.')

        return extrfunc(s, selection, **kwargs)

    return decorated_extrfunc


class Translate(AtomsProperty):

    """
    Translate

    Returns an Atoms object with some or all the atoms translated by a given
    vector. Absolute or scaled coordinates may be used.

    | Parameters:
    |   selection (AtomSelection): selection object defining which atoms to
    |                              act on. By default, all of them.
    |   vector ([float]*3): vector by which to translate the atoms.
    |   scaled (bool): if True, treat the input vector as expressed in scaled,
    |                  not absolute, coordinates.

    | Returns:
    |   translated (ase.Atoms): Atoms object with the translation performed.

    """

    default_name = "translated"
    default_params = {
        'selection': None,
        'vector': [0, 0, 0],
        'scaled': False
    }

    @staticmethod
    @_transform_sel_check
    def extract(s, selection, vector, scaled):

        vector = np.array(vector)
        if vector.shape != (3,):
            raise ValueError('Invalid vector passed to Translate.')

        sT = s.copy()

        if not scaled:
            pos = sT.get_positions()
        else:
            pos = sT.get_scaled_positions()

        pos[selection.indices] += vector

        if not scaled:
            sT.set_positions(pos)
        else:
            sT.set_scaled_positions(pos)

        return sT


class Rotate(AtomsProperty):

    """
    Rotate

    Returns an Atoms object with some or all the atoms rotated by a given
    quaternion and with a given center. Absolute or scaled coordinates may be
    used.

    | Parameters:
    |   selection (AtomSelection): selection object defining which atoms to
    |                              act on. By default, all of them.
    |   center ([float]*3): center around which the rotation should take
    |                       place. By default the origin of the axes.
    |   quaternion (ase.quaternions.Quaternion): quaternion expressing the
    |                                            rotation that should be
    |                                            applied.
    |   scaled (bool): if True, treat the input vector as expressed in scaled,
    |                  not absolute, coordinates.

    | Returns:
    |   rotated (ase.Atoms): Atoms object with the rotation performed.

    """

    default_name = "rotated"
    default_params = {
        'selection': None,
        'center': [0, 0, 0],
        'quaternion': None,
        'scaled': False
    }

    @staticmethod
    @_transform_sel_check
    def extract(s, selection, center, quaternion, scaled):

        center = np.array(center)
        if center.shape != (3,):
            raise ValueError('Invalid center passed to Rotate.')

        if quaternion is None:
            quaternion = Quaternion()

        sT = s.copy()

        if not scaled:
            pos = sT.get_positions()
        else:
            pos = sT.get_scaled_positions()

        pos -= center
        pos[selection.indices] = quaternion \
            .rotate(pos[selection.indices].T).T
        pos += center

        if not scaled:
            sT.set_positions(pos)
        else:
            sT.set_scaled_positions(pos)

        return sT


class Mirror(AtomsProperty):

    """
    Mirror

    Returns an Atoms object with some or all the atoms reflected with either a
    given center or a given plane. Absolute or scaled coordinates may be used.

    | Parameters:
    |   selection (AtomSelection): selection object defining which atoms to
    |                              act on. By default, all of them.
    |   center ([float]*3): center around which the reflection should take
    |                       place. By default the origin of the axes. Can't be
    |                       present at the same time as plane.
    |   plane ([float]*4): plane with respect to which the reflection should
    |                      take place, in the form [a, b, c, d] parameters of
    |                      the plane equation.
    |                      By default is not used. Can't be present at the
    |                      same time as center.
    |   scaled (bool): if True, treat the input vector as expressed in scaled,
    |                  not absolute, coordinates.

    | Returns:
    |   reflected (ase.Atoms): Atoms object with the reflection performed.

    """

    default_name = "reflected"
    default_params = {
        'selection': None,
        'center': None,
        'plane': None,
        'scaled': False
    }

    @staticmethod
    @_transform_sel_check
    def extract(s, selection, center, plane, scaled):

        if plane is not None and center is not None:
            raise ValueError('Can\'t pass both center and plane to Mirror')

        if center is not None:
            center = np.array(center)
            if center.shape != (3,):
                raise ValueError('Invalid center passed to Mirror.')
        elif plane is not None:
            plane = np.array(plane)
            if plane.shape != (4,):
                raise ValueError('Invalid plane passed to Mirror.')
        else:
            center = np.zeros(3)

        sT = s.copy()

        if not scaled:
            pos = sT.get_positions()
        else:
            pos = sT.get_scaled_positions()

        if center is not None:

            pos -= center
            pos[selection.indices] *= -1
            pos += center

        else:

            # Find the components of the position vectors normal to the plane
            nu = plane[:3]/np.linalg.norm(plane[:3])
            norm_pos = (np.dot(pos[selection.indices],
                               plane[:3])+plane[3])[:, None]*nu
            pos[selection.indices] = pos[selection.indices] - 2*norm_pos

        if not scaled:
            sT.set_positions(pos)
        else:
            sT.set_scaled_positions(pos)

        return sT


class Regularise(AtomsProperty):

    """
    Regularize

    Perform a translation by a vector calculated to cancel out the effect of
    global translational symmetry. In theory, given two copies of the same
    system that only differ by a translation of all atoms in the unit cell,
    this should produce two systems that overlap perfectly. Can be used to
    compare slightly different systems if they're similar enough. If a
    selection is given, only those atoms will be used to calculate the center,
    but the translation will still be applied to all atoms. The same atoms
    have to be used in all systems for comparisons to make sense (for example
    one might use all the heavy atoms and not include hydrogens).

    | Parameters:
    |   selection (AtomSelection): selection object defining which atoms to
    |                              act on. By default, all of them.

    | Returns:
    |   regularised (ase.Atoms): Atoms object translated by the regularizing
    |                            vector.

    """

    default_name = "regularised"
    default_params = {
        'selection': None,
    }

    @staticmethod
    @_transform_sel_check
    def extract(s, selection):

        # Calculate the vector to shift by
        v_frac = s.get_scaled_positions()
        reg_v = 0.5-periodic_center(v_frac[selection._indices])

        sT = s.copy()
        sT.set_scaled_positions((v_frac+reg_v) % 1)

        return sT
