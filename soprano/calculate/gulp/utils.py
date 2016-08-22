"""
Utility functions for operating with GULP

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re


def _gulp_cell_definition(s, syms=None):
    """Create a cell definition for a GULP input file. Will use syms if 
    passed, otherwise standard chemical symbols"""

    gcell = "vectors\n{0}\n".format('\n'.join(['\t'.join([str(x)
                                                          for x in row])
                                               for row in s.get_cell()]))
    gcell += "frac\n"
    syms = s.get_chemical_symbols() if syms is None else syms
    pos = s.get_scaled_positions()
    for i, s in enumerate(syms):
        gcell += "{0} {1} {2} {3}\n".format(s, *pos[i])

    return gcell


def _gulp_parse_energy(lines):
    """Parse energy out of a GULP output split in lines"""
    for l in lines[::-1]:
        if 'Total lattice energy       =' in l and 'eV' in l:
            return float(l.split()[4])

    return None


def _gulp_parse_charges(lines):
    """Parse charges out of a GULP output split in lines"""
    q_re = re.compile('Final charges from\s+([a-zA-Z\-]+)\s+:')
    qline_re = re.compile('([0-9]+)\s+([0-9]+)\s+([0-9\.\-]+)')
    for i, l in enumerate(lines[::-1]):
        q_type = q_re.findall(l)
        if len(q_type) == 1:
            # Found it!
            charges = {'type': q_type[0],
                       'q': [],
                       'Z': []}
            # Go forward until you find the first valid line
            in_block = False
            for l2 in lines[(len(lines)-i):]:
                parsed = qline_re.findall(l2)
                if len(parsed) == 0:
                    if in_block:
                        return charges
                    else:
                        continue
                else:
                    in_block = True
                    charges['q'].append(float(parsed[0][2]))
                    charges['Z'].append(int(parsed[0][1]))

    return None
