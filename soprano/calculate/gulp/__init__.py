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

"""Classes and functions to carry out calculations using the bindings to GULP
(General Utility Lattice Program), a software providing a lot of useful
calculations with empirical force fields, partial charge calculations, Ewald
summation of Coulombic interactions and more. GULP can be found at:

http://nanochemistry.curtin.edu.au/gulp/

It needs to be installed on your system to use any of the functionality
provided here.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.calculate.gulp.w99 import get_w99_energy, W99Error
from soprano.calculate.gulp.charges import get_gulp_charges
