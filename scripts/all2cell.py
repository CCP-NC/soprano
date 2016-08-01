#!/usr/bin/env python

""" A very simple script to convert from all ASE-accepted formats to CASTEP's
.cell format"""

import os
import argparse as ap
from ase import io

parser = ap.ArgumentParser()
# Main argument
parser.add_argument('input_files', type=str, nargs='+', default=None,
                    help="Files to convert to CELL")

args = parser.parse_args()

for inpf in args.input_files:
    try:
        a = io.read(inpf)
    except Exception as e:
        print("WARNING: file reading failed with exception:\n"
              ">\t{0}\nSkipping...".format(e))
        continue
    cname = os.path.splitext(inpf)[0] + '.cell'
    io.write(cname, a)
