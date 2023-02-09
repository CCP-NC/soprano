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
# - ~~Add function to symmetry-reduce the output -- merge symmetry-equivalent sites~~
#    - e.g. get asymmetric unit cell for molecular crystals

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import argparse as ap
import re
import os
import sys
from ase import io
from ase.spacegroup import get_spacegroup
from soprano.properties.labeling import UniqueSites
from soprano.properties.linkage import Bonds
from soprano.properties.nmr import *
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels
from collections import OrderedDict

import warnings

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "March 21, 2022"



def isotope_selection(selection_string):
    """Parse the isotope string. Adapted from [sumo](https://github.com/SMTG-UCL/sumo)
    Args:
        selection_string (str): The isotopes specification, in the form ``"2H,15N" for deuterium and 15N``.
    Returns:
        dict: The isotope for each element specified. Formatted as::
            {Element: Isotope}.
    """
    isotope_dict = {}
    for split in selection_string.split(","):
        matchobj = re.match(r"(\d+)([A-Z][a-z]?)", split)
        el = matchobj.group(2)
        isotope = int(matchobj.group(1))
        isotope_dict[el] = isotope
    return isotope_dict

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
        type=str,
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
        help=('Isotopes specification (e.g. "13C" for carbon 13 '
        '"2H,15N" for deuterium and N 15). '
        'When nothing is specified it defaults to the most common NMR active isotope.'),
    )
    # add argument to reduce by symmetry
    parser.add_argument(
        "-r",
        "--reduce",
        action="store_true",
        help=("Reduce the output by symmetry-equivalent sites. "
        "Note that this doesn't take into account magnetic symmetry!"),
    )
    #add option to average over reduced groups
    parser.add_argument(
        "-a",
        "--average",
        action="store_true",
        help=("Average over symmetry-equivalent sites. "    
        "Note that this doesn't take into account magnetic symmetry!"),
    )
    # add option to set the symprec
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-4,
        help=("Set the symprec for the symmetry reduction. "
        "Default is 1e-4."),
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
    # optional argument for the precision of the output
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Precision of the output (default: 3) -- number of decimal places",
    )
    ## add optional argument for euler angle convention
    parser.add_argument(
        "--euler",
        type=str,
        default="zyz",
        help="Euler angle convention (default: zyz). Can be either 'zyz' or 'zxz'",
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

    # label precision
    PREC = args.precision

    # comma or tab separated?
    if args.csv:
        sep = ',   '
    else:
        sep = '\t'

    for f in args.input_files:
        dirname = os.path.dirname(f)
        basename = os.path.basename(f)


        MS_HEADER  = '\n'+'\n'.join([
            '#'+155*'-'+'#', 
            '# {:^153} #'.format(f'Magnetic Shielding:  {f}') ,
            '#'+155*'-'+'#'])

        EFG_HEADER = '\n'+'\n'.join([
            '#'+155*'-'+'#', 
            '# {:^153} #'.format(f'Electric Field Gradient:  {f}') ,
            '#'+155*'-'+'#'])

        # try to read in the file:
        try:
            atoms = io.read(f)
        except IOError:
            print("File {0} not found, skipping".format(f))
            continue


        # Do they actually have any magres data?
        if not any([atoms.has(k) for k in properties]):
            print("File {0} has no {1} data extract, skipping".format(f, ' '.join(properties)))
            continue

        # create new array for multiplicities
        multiplicities = np.ones(len(atoms))
        atoms.set_array('multiplicities', multiplicities)

        # reduce by symmetry?
        if args.reduce:
            symprec = args.symprec
            tags = UniqueSites.get(atoms, symprec=symprec)
            groups = [np.where(tags == i)[0] for i in range(max(tags))]

            # update multiplicities
            group_multiplicities = [len(g) for g in groups]
            for i, g in enumerate(groups):
                multiplicities[g] = group_multiplicities[i]
            atoms.set_array('multiplicities', multiplicities)
            # average properties within each group
            if args.average: 
                for p in properties:
                    arr = atoms.get_array(p)
                    for group in groups:
                        # print(f"Average {p} over symmetry-equivalent sites {group}")
                        arr[group] = np.average(arr[group], axis=0)
                    atoms.set_array(p, arr)

            # create a new atoms object with only the unique sites
            # taking the first site from each group
            uniqueinds = [groups[i][0] for i in range(len(groups))]
            atoms = AtomSelection(atoms, uniqueinds).subset(atoms)


            print(atoms.get_array('multiplicities'))
            # atoms = AtomSelection.unique(atoms, symprec=symprec).subset(atoms)

        # select subset of atoms based on selection string
        if args.select:
            sel_selectionstring = AtomSelection.from_selection_string(atoms, args.select)
            atoms = sel_selectionstring.subset(atoms)

        # Note we could have combined selections as in Tutorial 3, but then 
        # we lose the nice ordering of the atoms so better to apply selections successively...        

        # Obtain labels for subset atoms:
        indices = atoms.get_array('indices')
        labels = atoms.get_array('labels')



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
                red_aniso = MSReducedAnisotropy.get(atoms)
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
                    table_headers = ['Label', 
                                     'Index',
                                     'Isotropy/ppm', 
                                     'Anisotropy/ppm', 
                                     'Red.anisotropy/ppm', 
                                     'Asymmetry', 
                                     'Span/ppm', 
                                     'Skew', 
                                     'alpha/deg', 
                                     'beta/deg', 
                                     'gamma/deg']
                    summary += [f'{sep}'.join([f'{lab:<9}' for lab in table_headers])]
                for i, jl in enumerate(labels):
                    a, b, c = quat[i].euler_angles(mode=args.euler)*180/np.pi # rad to degrees
                    summary +=[f'{jl: <5}{sep} {indices[i]:>9d}{sep} \t' + \
                               f'{iso[i]:12.{PREC}f}{sep} ' +\
                               f'{aniso[i]:13.{PREC}f}{sep} {red_aniso[i]:17.{PREC}f}{sep}' + \
                               f'{asymm[i]:9.{PREC}f}{sep}' + \
                               f'{span[i]:8.{PREC}f}{sep}{skew[i]:5.{PREC}f}{sep}' + \
                               f'\t{a:9.{PREC}f}{sep} {b:9.{PREC}f}{sep} {c:9.{PREC}f}']
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
                    table_headers = ['Label', 'Index', 'Vzz/au', 'Cq/MHz', 'Eta', 'alpha/deg', 'beta/deg', 'gamma/deg']
                    summary += [f'{sep}'.join([f'{lab:^9}' for lab in table_headers])]

                for i, jl in enumerate(labels):
                    a, b, c = quat[i].euler_angles(mode=args.euler)*180/np.pi # rad to degrees
                    summary +=[f'{jl:<5}{sep} {indices[i]:>9d}{sep}\t' +\
                               f'{vzz[i]:9.{PREC}f}{sep} {qC[i]:12.{PREC}e}{sep}' + \
                               f'{eta[i]:8.{PREC}f}{sep}' + \
                               f'{a:9.{PREC}f}{sep} {b:9.{PREC}f}{sep} {c:9.{PREC}f}']
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