"""Bindings for AIRSS Buildcell program for random structure generation"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import hashlib
import subprocess as sp
from ase import io as ase_io
# Internal imports
import soprano.utils as utils

# Python 2-to-3 compatibility
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


def airssGen(input_file,
             n=100,
             buildcell_command='buildcell',
             buildcell_path=None):
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
    basename = utils.seedname(input_file.name)
    input_file.close()

    # And keep track of the count!
    # (at least if it's not infinite)
    i = 0

    while True:
        if n is not None:
            if i >= n:
                return
            i += 1

        # Generate a structure
        stdout, stderr = sp.Popen(airss_cmd,
                                  universal_newlines=True,
                                  stdin=sp.PIPE,
                                  stdout=sp.PIPE,
                                  stderr=sp.PIPE).communicate(template)

        # Necessary for compatibility in Python2
        try:
            stdout = unicode(stdout)
        except NameError:
            pass

        # Now turn it into a proper Atoms object
        # To do this we need to make it look like a file to ASE's io.read
        newcell = ase_io.read(StringIO(stdout), format='castep-cell')
        # Generate it a name, function of its properties
        postfix = hashlib.md5(str(newcell.get_positions()
                                  ).encode()).hexdigest()
        newcell.info['name'] = '{0}_{1}'.format(basename, postfix)
        yield newcell
