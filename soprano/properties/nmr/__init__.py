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
Module containing AtomsProperties that relate to NMR properties of a system.
Some of these are valid only for Atoms objects loaded from a .magres file.
"""


from soprano.properties.nmr.dipolar import (
    DipolarCoupling,
    DipolarDiagonal,
    DipolarEuler,
    DipolarRSS,
    DipolarTensor,
)
from soprano.properties.nmr.efg import (
    EFGNQR,
    EFGAnisotropy,
    EFGAsymmetry,
    EFGDiagonal,
    EFGEuler,
    EFGQuadrupolarConstant,
    EFGQuadrupolarProduct,
    EFGQuaternion,
    EFGReducedAnisotropy,
    EFGSkew,
    EFGSpan,
    EFGTensor,
    EFGVzz,
)
from soprano.properties.nmr.isc import (
    JCDiagonal,
    JCIsotropy,
)
from soprano.properties.nmr.ms import (
    MSAnisotropy,
    MSAsymmetry,
    MSDiagonal,
    MSEuler,
    MSIsotropy,
    MSQuaternion,
    MSReducedAnisotropy,
    MSSkew,
    MSSpan,
    MSTensor,
)
