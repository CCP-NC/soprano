#!/usr/bin/env python

""" A very simple script to convert from all ASE-accepted formats to CASTEP's
.cell format"""

import os
import glob
import argparse as ap
from ase import io

parser = ap.ArgumentParser()
# Main argument
parser.add_argument('input_folders', type=str, nargs='+', default=None,
                    help="Folders of VASP input files to convert to CELL")

args = parser.parse_args()

for inpf in args.input_folders:
    # Check that these folders have the right files
    foldf = glob.glob(os.path.join(inpf, '*'))
    
