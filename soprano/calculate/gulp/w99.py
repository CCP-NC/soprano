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

"""
Classes and functions for using the W99 force field in GULP.
This force field only applies to organic molecules. More information can be
found in the original paper by Donald E. Williams:

D.E. Williams,
*Improved Intermolecular Force Field for Molecules Containing H, C, N, and O
Atoms, with Application to Nucleoside and Peptide Crystals*  - Journal
of Computational Chemistry, Vol. 22, No. 11, 1154-1166 (2001)

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import json
import pkgutil
import numpy as np
import subprocess as sp
from ase.calculators.singlepoint import SinglePointCalculator
from soprano.utils import safe_communicate
from soprano.properties.linkage import Molecules
from soprano.calculate.gulp._utils import (_gulp_cell_definition,
                                           _gulp_parse_energy,
                                           _gulp_parse_charges)

_w99_data = pkgutil.get_data('soprano',
                             'data/w99_parameters.json').decode('utf-8')
_w99_data = json.loads(_w99_data)


class W99Error(Exception):
    pass


def find_w99_atomtypes(s, force_recalc=False):
    """Calculate the W99 force field atom types for a given structure.

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
    w99types = np.array(['XXX' for sym in chsyms])

    for m_i, m in enumerate(mols):

        inds = list(m.indices)
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
                    if len(oxys) == 1 and len(bonds[c_mi]) == 3:
                        # Carboxylic!
                        w99types[a_i] = 'H_3'
                    else:
                        # Alcoholic!
                        w99types[a_i] = 'H_2'
            elif el_i == 'C':
                # Carbon case
                val = len(bnd_i)
                if val > 1 and val < 5:
                    w99types[a_i] = 'C_{0}'.format(val)
                else:
                    raise W99Error('ERROR - Anomalous chemical group found')
            elif el_i == 'O':
                # Oxygen case
                val = len(bnd_i)
                if val > 0 and val < 3:
                    w99types[a_i] = 'O_{0}'.format(val)
                else:
                    raise W99Error('ERROR - Anomalous chemical group found')
            elif el_i == 'N':
                # Nitrogen case
                val = len(bnd_i)
                if val == 3:
                    w99types[a_i] = 'N_1'
                else:
                    # Count hydrogens
                    hydros = [b for b in bnd_i if chsyms[b] == 'H']
                    if len(hydros) == 0:
                        w99types[a_i] = 'N_2'
                    elif len(hydros) == 1:
                        w99types[a_i] = 'N_3'
                    else:
                        w99types[a_i] = 'N_4'

        m.set_array('w99_types', w99types[inds])

    s.set_array('w99_types', w99types)


def _w99_field_definition(s, etol):
    """Output the W99 field definition for GULP input. System must already
    have w99 types calculated"""

    w99types = s.get_array('w99_types')

    # Which types are present?
    w99types = list(set(w99types))

    field_def = ''
    # Now take care of all permutations
    for i, t1 in enumerate(w99types):
        for t2 in w99types[i:]:
            # Calculate parameters
            if t1 == t2:
                A, rho, C = _w99_data[t1]
            else:
                A1, rho1, C1 = _w99_data[t1]
                A2, rho2, C2 = _w99_data[t2]
                A = np.sqrt(A1*A2)
                rho = 2.0/(1.0/rho1+1.0/rho2)
                C = np.sqrt(C1*C2)

            # Calculate the optimal cutoff to meet the required energy
            # tolerance
            expcut = -rho*np.log(etol/abs(A))
            r6cut = (abs(C)/etol)**(1.0/6.0)
            cut = max(expcut, r6cut, 0)
            field_def += 'buck inter\n{0} {1} {2} {3} {4} {5}\n'.format(t1,
                                                                        t2,
                                                                        A,
                                                                        rho,
                                                                        C,
                                                                        cut)

    return field_def


def get_w99_energy(s, charge_method='eem', Etol=1e-6,
                   gulp_command='gulp',
                   gulp_path=None,
                   save_charges=False):
    """Calculate the W99 force field energy using GULP.

    | Parameters:
    |   s (ase.Atoms): the structure to calculate the energy of
    |   charge_method (Optional[str]): which method to use for atomic partial
    |                                  charge calculation. Can be any of
    |                                  'eem', 'qeq' and 'pacha'.
    |                                  Default is 'eem'.
    |   Etol (Optional[float]): tolerance on energy for intermolecular
    |                           potential cutoffs (relative to single
    |                           interaction energy). Default is 1e-6 eV.
    |   gulp_command (Optional[str]): command required to call the GULP
    |                                 executable.
    |   gulp_path (Optional[str]): path where the GULP executable can be
    |                              found. If not present, the GULP command
    |                              will be invoked directly (assuming the
    |                              executable is in the system PATH).
    |   save_charges (Optional[bool]): whether to retrieve also the charges
    |                                  and save them in the Atoms object.
    |                                  False by default.

    | Returns:
    |   energy (float): the calculated energy

    """

    # Sanity check
    if charge_method not in ['eem', 'qeq', 'pacha']:
        raise ValueError('Invalid charge_method passed to get_w99_energy')

    # First, atom types
    find_w99_atomtypes(s)

    # Now define the input
    gin = "molq {0} dipole\n".format(charge_method)
    gin += _gulp_cell_definition(s, syms=s.get_array('w99_types'))

    # Finally, the potential definition
    gin += _w99_field_definition(s, Etol)

    # AND GO!
    if gulp_path is None:
        gulp_path = ''

    gulp_cmd = [os.path.join(gulp_path, gulp_command)]

    try:
        gulp_proc = sp.Popen(gulp_cmd,
                             universal_newlines=True,
                             stdin=sp.PIPE,
                             stdout=sp.PIPE,
                             stderr=sp.PIPE)
        stdout, stderr = safe_communicate(gulp_proc, gin)
    except OSError:
        raise RuntimeError('GULP not found on this system with the given '
                           'command')

    # Necessary for compatibility in Python2
    try:
        stdout = unicode(stdout)
    except NameError:
        pass

    # Now parse the energy
    gulp_lines = stdout.split('\n')
    E = _gulp_parse_energy(gulp_lines)
    if E is None:
        raise RuntimeError('ERROR - GULP run failed to return energy')

    # Remember it with a mock ASE calculator
    calc = SinglePointCalculator(s, energy=E)
    s.set_calculator(calc)

    if save_charges:
        qs = _gulp_parse_charges(gulp_lines)
        s.set_initial_charges(qs['q'])

    return E
