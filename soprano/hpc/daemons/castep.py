"""A basic Daemon to run CASTEP calculations
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import glob
import shutil
import tempfile

import numpy as np
import subprocess as sp
from ase.io.castep import read_cell, write_castep_cell
from ase.calculators.castep import create_castep_keywords
from soprano.hpc.daemons import DaemonHPC
from soprano.utils import seedname, replace_folder


class CastepDaemon(DaemonHPC):

    def set_parameters(self, folder_in, folder_out, castep_command):
        """Set a collection to draw Atoms objects from, and an index keeping
        track of where we are in scanning it.

        | Args:
        |   collection (AtomsCollection): AtomsCollection object containing
        |                                 the atoms to perform calculations
        |                                 on. These must already have a CASTEP
        |                                 calculator set.

        """

        self.folder_in = folder_in
        self.folder_out = folder_out
        self.castep_command = castep_command

        # Initialize the CASTEP keywords file in the IN folder
        self.kwdir = tempfile.mkdtemp()
        create_castep_keywords(castep_command,
                               os.path.join(self.kwdir, 
                                            'castep_keywords.py'))
        sys.path.append(self.kwdir)

    def next_processes(self, n=1):

        # Grab n cell files from the folder
        cfile_list = glob.glob(os.path.join(self.folder_in, '*.cell'))
        np.random.shuffle(cfile_list)

        # Try grabbing all of them - if they're not there, don't make a fuss
        procs = []
        for a in cfile_list[:n]:
            try:
                procs.append({'atoms': read_cell(a),
                              'seedname': seedname(a)})                
            except IOError:
                pass
        # It is better to reset the calculators as they don't serialize well
        # Also, move the files to the out folder!
        for i, p in enumerate(procs):
            p['atoms'].set_calculator(None)
            shutil.move(cfile_list[i], 
                        replace_folder(cfile_list[i], self.folder_out))

        return procs

    def on_complete(self, rval):

        if len(glob.glob(os.path.join(self.folder_in, '*.cell'))) > 0:
            self.start_processes()

    def run_process(self, loop_id, atoms, seedname):

        # Make a temporary folder, store everything there, run
        calcfold = tempfile.mkdtemp()
        write_castep_cell(open(os.path.join(calcfold,
                                            seedname + '.cell'),
                               'w'), atoms)
        # Run the calculation
        stdout, stderr = sp.Popen([self.castep_command, seedname],
                                  cwd=calcfold,
                                  stdout=sp.PIPE,
                                  stderr=sp.PIPE).communicate()
        # Copy all output files
        outfiles = glob.glob(os.path.join(calcfold, '*'))
        for of in outfiles:
            shutil.move(of, replace_folder(of, self.folder_out))
        shutil.rmtree(calcfold)

        return "Complete"

    def terminate(self):

        # It's over!
        # Remove the temporary directory
        try:
            shutil.rmtree(self.kwdir)
        except OSError:
            self.log("Could not delete temporary castep_keywords.py"
                     "directory at {0}".format(self.kwdir))
        sys.path.remove(self.kwdir)

        super(CastepDaemon, self).terminate()

