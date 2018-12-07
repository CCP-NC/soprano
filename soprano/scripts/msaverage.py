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

""" A script to compute averages of NMR tensors over CH3 and NH3 groups"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import argparse as ap

from ase import io
from soprano.properties.linkage import Bonds


def __main__():

    parser = ap.ArgumentParser(description="""
        Processes .magres files containing chemical groups of the form XH3
        (by default, CH3 and NH3), to average the required NMR tensors for the
        hydrogen atoms connected to such groups.
        """)
    # Main argument
    parser.add_argument('input_files', type=str, nargs='+', default=None,
                        help="Magres files on which to carry out averages")
    # Optional arguments
    parser.add_argument('-X', type=str, nargs='+', default=['C', 'N'],
                        help="Nuclei to consider for XH3 groups "
                        "(default: C and N)")
    parser.add_argument('-vdws', type=float, default=1.0,
                        help="Van der Waals radius scale for bond calculation."
                        " Increase for higher tolerance")
    parser.add_argument('-avg', type=str, nargs='+', default=['ms', 'efg'],
                        help="Arrays to average over XH3 groups "
                        "(default: ms and efg)")
    parser.add_argument('-prefix', type=str, default='avg',
                        help="Prefix added to output files")

    args = parser.parse_args()

    bcalc = Bonds(vdw_scale=args.vdws, return_matrix=True)

    for f in args.input_files:
        try:
            a = io.read(f)
        except IOError:
            print('File {0} not found, skipping'.format(f))
            continue

        # Do they have magres data?
        if not any([a.has(k) for k in args.avg]):
            print('File {0} has no data to average, skipping'.format(f))
            continue

        # Find what to average
        bonds, bmat = bcalc(a)

        # Find XH3 groups
        symbs = np.array(a.get_chemical_symbols())
        hinds = np.where(symbs == 'H')[0]
        h3groups = []

        for xsymb in args.X:

            xinds = np.where(symbs == xsymb)[0]
            xinds = xinds[np.where(np.sum(bmat[xinds][:, hinds],
                                          axis=1) == 3)[0]]
            if len(xinds) > 0:
                h3groups.append(np.where(bmat[xinds][:, hinds] == 1)[1])

        # Now average
        avg_a = a.copy()

        for k in args.avg:
            arr = a.get_array(k)
            for h3 in h3groups:
                arr[h3] = np.average(arr[h3], axis=0)
            avg_a.set_array(k, arr)

        io.write('{0}_{1}'.format(args.prefix, f), avg_a)
