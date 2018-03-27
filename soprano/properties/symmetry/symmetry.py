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


from soprano.properties import AtomsProperty
from soprano.optional import requireSpglib


@requireSpglib('spglib')
def _get_symmetry_dataset(s, symprec, spglib=None):
    symdata = spglib.get_symmetry_dataset(s, symprec=symprec)
    return symdata


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
