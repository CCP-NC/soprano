#!/usr/bin/env python
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
                print("WARNING - ASE could not read elements in POSCAR file. "
                      "\nBy convention, elements should be elencated in the "
                      "first line of the POSCAR file in the order in which "
                      "they appear in the line in which the number of each "
                      "atom type is given.")
            continue
    if a is None:
        print("No valid structure files found in folder {0}\n".format(inpf) +
              "Skipping...")
        continue

    name = os.path.split(inpf)[-1]
    io.write(name + '.cell', a)





