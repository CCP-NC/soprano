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

""" A script to extract NMR properties from a (list of) .magres files"""

# TODO:
# - Add support for more properties
# - Add selection by X
#    - got sumo-style atom selection down. What else would be good to implement now? 
#  ~~Add support for custom isotopes~~
# - Add function to symmetry-reduce the output -- merge symmetry-equivalent sites
#    - e.g. get asymmetric unit cell for molecular crystals

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import argparse as ap
import os
import sys
from ase import io
from soprano.properties.linkage import Bonds
from soprano.properties.nmr import *
from soprano.selection import AtomSelection

import warnings

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "March 21, 2022"



def atoms_selection(selection_string):
    """Parse the atom string. Adapted from [sumo](https://github.com/SMTG-UCL/sumo)
    Args:
        selection_string (str): The atoms to extract, in the form ``"C.1.2.3,"``.
    Returns:
        dict: The atomic indices for which to extract the NMR properties. Formatted as::
            {Element: [atom_indices]}.
        Indices are 1-indexed. If an element symbol
        is included with an empty list, then all sites for that species are considered.
    """
    selection = {}
    for split in selection_string.split(","):
        sites = split.split(".")
        el = sites.pop(0)
        sites = list(map(int, sites))
        selection[el] = np.array(sites)
    return selection

def isotope_selection(selection_string):
    """Parse the isotope string. Adapted from [sumo](https://github.com/SMTG-UCL/sumo)
    Args:
        selection_string (str): The isotopes specify, in the form ``"H.2,N.15" for deuterium and N 15``.
    Returns:
        dict: The atomic indices for which to extract the NMR properties. Formatted as::
            {Element: Isotope}.
    """
    selection = {}
    for split in selection_string.split(","):
        isotopes = split.split(".")
        el = isotopes.pop(0)
        isotopes = list(map(int, isotopes))
        assert len(isotopes) == 1 # anything else doesn't make sense
        selection[el] = isotopes[0]
    return selection

def __main__():

    parser = ap.ArgumentParser(
        description="""
        Processes .magres files containing NMR-related properties 
        and prints a summary. It defaults to printing all NMR properties 
        present in the file for all the atoms. 
        
        See the below arguments for how to extract specific information.
        """,
        epilog=f"""
        Author: {__author__} ({__email__})
        Last updated: {__date__}
        """
    )
    # Main argument
    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        default=None,
        help="Magres files for which to extract the summary data. "
        "You can use standard shell wildcards such as *.magres",
    )
    # Optional arguments
    parser.add_argument(
        "-s",
        "--select",
        type=atoms_selection,
        metavar="A",
        help=('element/atoms to include (e.g. "C" for only carbon or '
        '"C.1.2.3,H.1.2" for carbons 1,2,3 and hydrogens 1 and 2)'),
    )
    parser.add_argument(
        "-i",
        "--isotope",
        type=isotope_selection,
        metavar="I",
        default={},
        help=('Isotopes specification (e.g. "C.13" carbon 13 '
        '"H.2,N.15" deuterium and N 15). '
        'When nothing is specified it defaults to the most common NMR active isotope.'),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action='store_true',
        default=False,
        help="If present, suppress print headers and append filename to each line",
    )
    parser.add_argument(
        "-p",
        "--properties",
        type=str,
        metavar="P",
        nargs="+",
        default=["ms", "efg"],
        help="Properties for which to extract and summarise e.g. '-p ms efg'"
         "(default: ms and efg)",
    )
    ## output csv? default False 
    parser.add_argument('--csv',    action='store_true', 
        help="output data to a CSV file for each .magres file"
        "(default: False)")
    parser.add_argument('--no-csv', action='store_false')
    parser.set_defaults(csv=False)

    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix added to CSV output file"
    )

    args = parser.parse_args()
    properties = [p.lower() for p in args.properties]


    # how many files are to be summarised? 
    nfiles = len(args.input_files)

    # comma or tab separated?
    if args.csv:
        sep = ',   '
    else:
        sep = '\t'

    for f in args.input_files:
        dirname = os.path.dirname(f)
        basename = os.path.basename(f)


        MS_HEADER  = '\n'+'\n'.join([
            '#'+120*'-'+'#', 
            '# {:^118} #'.format(f'Magnetic Shielding:  {f}') ,
            '#'+120*'-'+'#'])

        EFG_HEADER = '\n'+'\n'.join([
            '#'+120*'-'+'#', 
            '# {:^118} #'.format(f'Electric Field Gradient:  {f}') ,
            '#'+120*'-'+'#'])

        try:
            atoms = io.read(f)
        except IOError:
            print("File {0} not found, skipping".format(f))
            continue


        # Do they actually have any magres data?
        if not any([atoms.has(k) for k in properties]):
            print("File {0} has no {1} data extract, skipping".format(f, ' '.join(properties)))
            continue

        
        # select subset of atoms
        if args.select:
            indices = atoms.get_array('indices') # these are 1-indexed per jmol convention
            
            # this works but is pretty ugly... 
            keep = [atom.index 
                   for atom in atoms 
                   if (atom.symbol in args.select) and len(args.select[atom.symbol]) == 0 \
                   or atom.symbol in args.select and indices[atom.index] in args.select[atom.symbol]]
            atoms = AtomSelection(atoms, keep).subset(atoms)

        # select by other? 
        
        
        
        # Obtain nice labels for subset atoms:
        indices = atoms.get_array('indices')
        labels  = ["{0}_{1}".format(l, i) for l, i in zip(atoms.get_array('labels'), indices)]



        # --- ms --- #
        if 'ms' in properties:
            # list to hold the summary for this file
            summary = []
            try:
                # quick check that ms data is present
                mstensor = atoms.get_array('ms')

                # Isotropy, Anisotropy and Asymmetry (Haeberlen convention)
                iso   = MSIsotropy.get(atoms)
                aniso = MSAnisotropy.get(atoms)
                asymm = MSAsymmetry.get(atoms)
                # Span and skew
                span = MSSpan.get(atoms)
                skew = MSSkew.get(atoms)
                # quaternion
                quat = MSQuaternion.get(atoms)


                # if not quiet, print header
                if not args.quiet:
                    if not args.csv:
                        summary += [MS_HEADER.format(fname=f)]
                    summary += [f'Label{sep} Isotropy/ppm{sep} Anisotropy/ppm{sep} Asymmetry {sep}' + \
                                f'Span/ppm{sep} Skew{sep}' + \
                                f'alpha/deg{sep} beta/deg{sep} gamma/deg']
                for i, jl in enumerate(labels):
                    a, b, c = quat[i].euler_angles(mode='zyz')*180/np.pi # rad to degrees
                    summary +=[f'{jl: <5}{sep} {iso[i]:12.2f}{sep} {aniso[i]:14.2f}{sep} {asymm[i]:9.2f}{sep}' + \
                               f'{span[i]:8.2f}{sep}{skew[i]:5.2f}{sep}' + \
                               f'{a:9.2f}{sep} {b:9.2f}{sep} {c:9.2f}']
                    if args.quiet:
                        # add file path to end if in quiet mode
                        summary += [f]
                # output ms summary 
                if args.csv:
                    fileout = 'MS_' + basename.replace('.magres', '.csv')
                    prefix = args.prefix
                    if prefix != "":
                        fileout = f"{prefix}_{fileout}"
                    fileout = os.path.join(dirname, fileout)
                    with open(fileout, 'w') as fd:
                        fd.write('\n'.join(summary))
                else:
                    print('\n'.join(summary))
            except KeyError:
                warnings.warn(f'No MS data found in {f}\n'
                'Set argument `-p efg` if the file(s) only contains EFG data ')
                pass
            except:
                warnings.warn('Failed to load MS data from .magres')
                raise
                
        # end ms

        # --- EFG --- #
        if 'efg' in properties:
            # list to hold the summary for this file
            summary = []
            
            try:

                vzz = EFGVzz.get(atoms)
                # For quadrupolar constants, isotopes become relevant. This means we need to create custom Property instances to
                # specify them. There are multiple ways to do so - check the docstrings for more details - but here we set them
                # by element. When nothing is specified it defaults to the most common NMR active isotope.

                isotopes = args.isotope
                qP = EFGQuadrupolarConstant(isotopes=isotopes) # Deuterated; for the others use the default
                qC = qP(atoms)/1e6 # To MHz

                quat = EFGQuaternion.get(atoms)
                eta  = EFGAsymmetry.get(atoms)

                # if not quiet, print header
                if not args.quiet:
                    if not args.csv:
                        summary += [EFG_HEADER.format(fname=f)]
                    table_headers = ['Label', 'Vzz/au', 'Cq/MHz', 'Eta', 'alpha/deg', 'beta/deg', 'gamma/deg']
                    summary += [f'{sep}'.join([f'{lab:>9}' for lab in table_headers])]

                for i, jl in enumerate(labels):
                    a, b, c = quat[i].euler_angles(mode='zyz')*180/np.pi # rad to degrees
                    summary +=[f'{jl:<9}{sep} {vzz[i]:9.2f}{sep} {qC[i]:12.4e}{sep}' + \
                               f'{eta[i]:8.3f}{sep}' + \
                               f'{a:9.2f}{sep} {b:9.2f}{sep} {c:9.2f}']
                    if args.quiet:
                        # add file path to end if in quiet mode
                        summary[-1] += f'  {f}'
                # output EFG summary 
                if args.csv:
                    fileout = 'EFG_' + basename.replace('.magres', '.csv')
                    prefix = args.prefix
                    if prefix != "":
                        fileout = f"{prefix}_{fileout}"
                    fileout = os.path.join(dirname, fileout)
                    with open(fileout, 'w') as fd:
                        fd.write('\n'.join(summary))
                else:
                    print('\n'.join(summary))
            except KeyError:
                warnings.warn(f'No EFG data found in {f}\n'
                'Set argument `-p ms` if the file(s) only contains MS data ')
                pass
            except:
                warnings.warn(f'Failed to load EFG data from .magres, {f}')
                raise
        
        # end EFG section
        ###############




if __name__ == '__main__':
    sys.exit(__main__())