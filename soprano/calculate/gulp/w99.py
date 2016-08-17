"""
Classes and functions for using the W99 force field in GULP.
This force field only applies to organic molecules. More information can be 
found in the original paper by Donald E. Williams:

D.E. Williams,
"Improved Intermolecular Force Field for Molecules Containing H, C, N, and O
Atoms, with Application to Nucleoside and Peptide Crystals"
Journal of Computational Chemistry, Vol. 22, No. 11, 1154-1166 (2001)

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pkgutil
import numpy as np
from soprano.properties.linkage import Molecules

_w99_data = pkgutil.get_data('soprano',
                             'data/w99_parameters.json').decode('utf-8')


class W99Error(Exception):
    pass

def find_w99_atomtypes(s, force_recalc=False):
    """Calculate the W99 force field atom types for a given structure

    | Parameters:
    |   s (ase.Atoms): the structure to calculate the atomtypes on
    |   force_recalc (bool): whether to recalculate the molecules even if 
    |                        already present. Default is False.    
    """

    # First, check that W99 even applies to this system
    chsyms = np.array(s.get_chemical_symbols())

    if not set(chsyms).issubset(set(['C', 'H', 'O', 'N'])):
        raise W99Error('Invalid system for use of W99 force field'
                       ' - System can only contain H, C, O and N!')

    if Molecules.default_name not in s.info or force_recalc:
        Molecules.get(s)

    mols = s.info['molecules']

    # Now the types!
    w99types = ['' for sym in chsyms]

    for m_i, m in enumerate(mols):

        inds = m.indices
        bonds = m.get_array('bonds')
        msyms = chsyms[inds]

        for a_mi, a_i in enumerate(inds):
            # What element is it?
            el_i = msyms[a_mi]
            bnd_i = bonds[a_mi]
            if len(bnd_i) < 1:
                # Something's wrong
                raise W99Error('ERROR - W99 can not be applied to single'
                               ' atom molecular fragments')
            if el_i == 'H':
                # Hydrogen case
                if len(bnd_i) != 1:
                    raise W99Error('ERROR - Hydrogen with non-1 valence '
                                   'found')
                # What is it bonded to?
                nn_el = chsyms[bnd_i[0]]
                if nn_el == 'C':
                    w99types[a_i] = 'H_1'
                elif nn_el == 'N':
                    w99types[a_i] = 'H_4'
                elif nn_el == 'O':
                    # Alcoholic or carboxylic?
                    nn_mi = inds.index(bnd_i[0])
                    carbs = [b for b in bonds[nn_mi] if chsyms[b] == 'C']
                    if len(carbs) != 1:
                        raise W99Error('ERROR - Anomalous chemical group '
                                       'found')
                    c_mi = inds.index(carbs[0])
                    oxys = [b for b in bonds[c_mi]
                            if chsyms[b] == 'O' and b != bnd_i[0]]
                    # What is it?
                    if len(oxys) == 0:
                        # Alcoholic!
                        w99types[a_i] = 'H_2'
                    elif len(oxys) == 1:
                        # Carboxylic!
                        w99types[a_i] = 'H_3'
                    else:
                        raise W99Error('ERROR - Anomalous chemical group '
                                       'found')
            elif el_i == 'C':
                # Carbon case
                pass





