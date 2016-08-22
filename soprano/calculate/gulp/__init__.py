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

from soprano.calculate.gulp.w99 import get_w99_energy
