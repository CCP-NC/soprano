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

"""
Van der Waals radii
"""

import json
import pkgutil
import numpy as np
from ase.data.vdw import vdw_radii as _vdw_radii_ase


def _load_vdw(name):

    _vdw_data = pkgutil.get_data('soprano', 'data/vdw_{0}.json'.format(name)
                                 ).decode('utf-8')
    _vdw_radii = np.array(json.loads(_vdw_data))

    return _vdw_radii


vdw_radii = {
    'ase': _vdw_radii_ase,
    'jmol': _load_vdw('jmol')
}
