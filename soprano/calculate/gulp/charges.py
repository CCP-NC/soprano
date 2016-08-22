"""
Get charges using GULP

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import subprocess as sp
from soprano.calculate.gulp.utils import (_gulp_cell_definition,
                                          _gulp_parse_charges)


def get_gulp_charges(s, charge_method="eem", save_charges=True,
                     gulp_command='gulp',
                     gulp_path=None):
    """Calculate the atomic partial charges using GULP.

    | Parameters:
    |   s (ase.Atoms): the structure to calculate the energy of
    |   charge_method (Optional[str]): which method to use for atomic partial
    |                                  charge calculation. Can be any of
    |                                  'eem', 'qeq' and 'pacha'.
    |                                  Default is 'eem'.
    |   save_charges (Optional[bool]): whether to save or not the charges in 
    |                                  the given ase.Atoms object. Default is
    |                                  True.
    |   gulp_command (Optional[str]): command required to call the GULP
    |                                 executable.
    |   gulp_path (Optional[str]): path where the GULP executable can be
    |                              found. If not present, the GULP command
    |                              will be invoked directly (assuming the
    |                              executable is in the system PATH).

    | Returns:
    |   charges(np.array(float)): per-atom partial charges

    """

    # Sanity check
    if charge_method not in ['eem', 'qeq', 'pacha']:
        raise ValueError('Invalid charge_method passed to get_gulp_charges')

    # Now define the input
    gin = "{0}\n".format(charge_method)
    gin += _gulp_cell_definition(s)

    # AND GO!
    if gulp_path is None:
        gulp_path = ''

    gulp_cmd = [os.path.join(gulp_path, gulp_command)]

    # Run the thing...
    try:
        stdout, stderr = sp.Popen(gulp_cmd,
                                  universal_newlines=True,
                                  stdin=sp.PIPE,
                                  stdout=sp.PIPE,
                                  stderr=sp.PIPE).communicate(gin)
    except OSError:
        raise RuntimeError('GULP not found on this system with the given '
                           'command')

    # Necessary for compatibility in Python2
    try:
        stdout = unicode(stdout)
    except NameError:
        pass

    # And parse the output
    gulp_lines = stdout.split('\n')
    charges = _gulp_parse_charges(gulp_lines)
    if charges is None:
        raise RuntimeError('ERROR - GULP run failed to return charges')

    # Run a security check
    if not np.all(s.get_atomic_numbers() == charges['Z']):
        raise RuntimeError('ERROR - Invalid charges parsed from GULP output')

    charges = charges['q']

    if save_charges:
        s.set_initial_charges(charges)

    return charges
