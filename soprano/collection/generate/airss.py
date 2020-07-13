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

"""Bindings for AIRSS Buildcell program for random structure generation"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import copy
import hashlib
import subprocess as sp
from ase import io as ase_io
# Internal imports
from soprano.utils import seedname, safe_communicate

# Python 2-to-3 compatibility
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


def airssGen(input_file,
             n=100,
             buildcell_command='buildcell',
             buildcell_path=None,
             clone_calc=True):
    """Generator function binding to AIRSS' Buildcell.

    This function searches for a buildcell executable and uses it to
    generate multiple new Atoms structures for a collection.

    | Args:
    |   input_file (str or file): the .cell file with appropriate comments
    |                             specifying the details of buildcell's
    |                             construction work.
    |   n (int): number of structures to generate. If set to None the
    |            generator goes on indefinitely.
    |   buildcell_command (str): command required to call the buildcell
    |                            executable.
    |   buildcell_path (str): path where the buildcell executable can be
    |                         found. If not present, the buildcell command
    |                         will be invoked directly (assuming the
    |                         executable is in the system PATH).
    |   clone_calc (bool): if True, the CASTEP calculator in the input file
    |                      will be copied and attached to the new structures.
    |                      This means that for example any additional CASTEP
    |                      keywords/blocks in the input file will be carried
    |                      on to the new structures. Default is True.

    | Returns:
    |   airssGenerator (generator): an iterable object that yields structures
    |                               created by buildcell.

    """

    # First: check that AIRSS is even installed
    if buildcell_path is None:
        buildcell_path = ''
    airss_cmd = [os.path.join(buildcell_path, buildcell_command)]

    try:
        stdout, stderr = sp.Popen(airss_cmd + ['-h'],
                                  stdout=sp.PIPE,
                                  stderr=sp.PIPE).communicate()
    except OSError:
        # Not even installed!
        raise RuntimeError('No instance of Buildcell found on this system')

    # Now open the given input file
    try:
        input_file = open(input_file)   # If it's a string
    except TypeError:
        pass                            # If it's already a file
    template = input_file.read()
    # Now get the file name
    basename = seedname(input_file.name)
    input_file.close()

    # Calculator (if needed)
    calc = None
    if clone_calc:
        calc = ase_io.read(input_file.name).calc

    # And keep track of the count!
    # (at least if it's not infinite)
    i = 0

    while True:
        if n is not None:
            if i >= n:
                return
            i += 1

        # Generate a structure
        subproc = sp.Popen(airss_cmd,
                           universal_newlines=True,
                           stdin=sp.PIPE,
                           stdout=sp.PIPE,
                           stderr=sp.PIPE)
        stdout, stderr = safe_communicate(subproc, template)

        # Now turn it into a proper Atoms object
        # To do this we need to make it look like a file to ASE's io.read
        try:
            newcell = ase_io.read(StringIO(stdout), format='castep-cell')
        except:
            # If ANYTHING happens, let's consider that stdout might be wrong
            raise RuntimeError(('Invalid output from buildcell:\nstdout:\n{0}'
                                '\nstderr:\n{1}')
                               .format(stdout, stderr))
        if clone_calc:
            newcell.calc = copy.deepcopy(calc)
        # Generate it a name, function of its properties
        postfix = hashlib.md5(str(newcell.get_positions()
                                  ).encode()).hexdigest()
        newcell.info['name'] = '{0}_{1}'.format(basename, postfix)
        yield newcell
