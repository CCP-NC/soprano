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

""" A very simple script to convert from all ASE-accepted formats to CASTEP's
.cell format"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import argparse as ap
from ase import io


def __main__():

    parser = ap.ArgumentParser()
    # Main argument
    parser.add_argument('input_folders', type=str, nargs='+', default=None,
                        help="Folders of VASP input files to convert to CELL")

    args = parser.parse_args()

    _inpfiles = ('POSCAR', 'CONTCAR')

    for inpf in args.input_folders:
        # Check that these folders have the right files
        a = None
        for ft in _inpfiles:
            try:
                a = io.read(os.path.join(inpf, ft))
            except IOError as e:
                if "Could not determine chemical symbols" in str(e):
                    # Wrong header format!
                    print("WARNING - ASE could not read elements in "
                          "POSCAR file.\n"
                          "By convention, elements should be elencated in the "
                          "first line of the POSCAR file in the order "
                          "in which they appear in the line in which the "
                          "number of each atom type is given.")
                continue
        if a is None:
            print("No valid structure files found in folder "
                  "{0}\nSkipping...".format(inpf))
            continue

        name = os.path.split(inpf)[-1]
        io.write(name + '.cell', a)