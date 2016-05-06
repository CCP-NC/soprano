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
from soprano.properties import AtomsProperty


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
    def extract(s, selection, vector, scaled):

        # Some necessary checks
        if not selection.validate(s):
            raise ValueError('Selection passed to Translate does not apply to'
                             ' system.')

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

    default_name = "translated"
    default_params = {
        'selection': None,
        'center': [0, 0, 0],
        'quaternion': None,
        'scaled': False
    }

    @staticmethod
    def extract(s, selection, center, quaternion, scaled):

        # Some necessary checks
        if not selection.validate(s):
            raise ValueError('Selection passed to Rotate does not apply to'
                             ' system.')

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

    default_name = "translated"
    default_params = {
        'selection': None,
        'center': None,
        'plane': None,
        'scaled': False
    }

    @staticmethod
    def extract(s, selection, center, plane, scaled):

        # Some necessary checks
        if not selection.validate(s):
            raise ValueError('Selection passed to Mirror does not apply to'
                             ' system.')

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
            norm_pos = (np.dot(pos[selection.indices], plane[:3])+plane[3])*nu
            pos[selection.indices] = pos[selection.indices] - 2*norm_pos

        if not scaled:
            sT.set_positions(pos)
        else:
            sT.set_scaled_positions(pos)

        return sT
