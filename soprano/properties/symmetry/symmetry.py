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

"""Implementation of AtomProperties that relate to symmetry"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from collections import namedtuple
from soprano.properties import AtomsProperty
from soprano.properties.symmetry.utils import (_get_symmetry_dataset,
                                               _find_wyckoff_points)

WyckoffPoint = namedtuple('WyckoffPoint', ['fpos', 'pos', 'operations',
                                           'hessian'])


class SymmetryDataset(AtomsProperty):

    """
    SymmetryDataset

    Extracts SPGLIB's standard symmetry dataset from a given system, including
    spacegroup symbol, symmetry operations etc.

    | Parameters:
    |   symprec (float): distance tolerance, in Angstroms, applied when
    |                    searching symmetry.

    | Returns:
    |   symm_dataset (dict): dictionary of symmetry information

    """

    default_name = 'symmetry_dataset'
    default_params = {
        'symprec': 1e-5,
    }

    @staticmethod
    def extract(s, symprec):
        return _get_symmetry_dataset(s, symprec=symprec)


class WyckoffPoints(AtomsProperty):

    """
    WyckoffPoints

    Returns a list of the found high symmetry points for a given system,
    including information about their point group operations, and the 
    properties of Hessian-like quantities at that point, namely, if they are
    constrained to be isotropic, definite (positive/negative), or
    can be anything.


    | Parameters:
    |   symprec (float): distance tolerance, in Angstroms, applied when
    |                    searching symmetry.

    | Returns:
    |   wyckoff_points (list): a list of WyckoffPoint named tuples, containing
    |                          the members 'fpos' (fractional coordinates),
    |                          'pos' (Cartesian coordinates), 'operations'
    |                          (point group operations) and 'isotropic'
    |                          (whether Hessian-like tensors are isotropic at
    |                          the point).

    """

    default_name = 'wyckoff_points'
    default_params = {
        'symprec': 1e-5,
    }

    @staticmethod
    def extract(s, symprec):

        hprops = ['saddle', 'none', 'definite', 'isotropic']

        fpos, ops, hess = _find_wyckoff_points(s, symprec)
        pos = np.dot(fpos, s.get_cell())

        wpoints = [WyckoffPoint(fp, p, o, hprops[h+1])
                   for (fp, p, o, h) in zip(fpos, pos, ops, hess)]

        return wpoints
