"""
Definition of CastepSubmitter class.

A basic "rolling" submitter for Castep calculations, grabbing from one folder
and depositing results in another.
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

from soprano import utils
from soprano.hpc.submitter import Submitter
from ase.calculators.castep import create_castep_keywords


class CastepSubmitter(Submitter):

    def set_parameters(self, folder_in, folder_out, castep_command,
                       castep_path=None,
                       copy_extensions=['.castep'],
                       pspot_files=[],
                       dryrun_test=False):
        """Set the parameters of the CASTEP Submitter

        | Args:
        |   folder_in (str): path of the folder to extract cell files from
        |   folder_out (str): path of the folder where the results will be
        |                     saved
        |   castep_command (str): command used to call the CASTEP executable on
        |                         this system
        |   castep_path (Optional[str]): folder where the CASTEP executable is 
        |                                located (if not part of the system
        |                                PATH)
        |   pspot_files (Optional[list[str]]): additional pseudopotential files
        |                                      to be copied in the input
        |                                      temporary folders
        |   copy_extensions (Optional[list[str]]): extensions of output files to
        |                                          copy to the output folder (by
        |                                          default only .castep file)
        |   dryrun_test (Optional[bool]): run a dryrun test on files before 
        |                                 actually running the calculation. Off
        |                                 by default.

        """

        # Check for existence of the folders, if not present create them
        try:
            os.mkdir(folder_in)
        except OSError:
            pass

        try:
            os.mkdir(folder_out)
        except OSError:
            pass

        self.folder_in = folder_in
        self.folder_out = folder_out
        self.castep_command = castep_command
        self.pspots = [os.path.abspath(pp) for pp in pspot_files]
        self.cp_ext = copy_extensions
        self.drun = dryrun_test

        self.castpath = castep_path

        # Finally add the castep_path to PATH if needed
        if castep_path is not None:
            sys.path.append(castep_path)

    def start_run(self):
        # Initialize the CASTEP keywords file in a dedicated temporary folder
        self.kwdir = tempfile.mkdtemp()
        create_castep_keywords(self.castep_command,
                               os.path.join(self.kwdir,
                                            'castep_keywords.py'))
        sys.path.append(self.kwdir)

    def next_job(self):
        """Grab the next job from folder_in"""

        cfile_list = glob.glob(os.path.join(self.folder_in, '*.cell'))
        if len(cfile_list) == 0:
            return None

        cfile = cfile_list[0]
        name = utils.seedname(cfile)
        files = [cfile]
        # Check if .param file is available too
        if os.path.isfile(os.path.join(self.folder_in, name + '.param')):
            files += [os.path.join(self.folder_in, name + '.param')]
        job = {
            'name': name,
            'args': {
                'files': files
            }
        }

        return job

    def setup_job(self, name, args, folder):
        """Copy files to temporary folder to prepare for execution"""

        for f in args['files']:
            shutil.move(f, folder)
        # Also copy the pseudopotentials
        for pp in self.pspots:
            shutil.copy(pp, folder)

        success = True

        # Perform dryrun test if required
        if self.drun:
            stdout, stderr = sp.Popen([self.castep_command, name, '--dryrun'],
                                      cwd=folder, stdout=sp.PIPE,
                                      stderr=sp.PIPE).communicate()
            # When it's finished...
            try:
                castfile = open(os.path.join(folder, name + '.castep'))
            except IOError:
                return False
            # Does the file contain the required lines?
            drline1 = "|       DRYRUN finished ...                       |"
            drline2 = "|       No problems found with input files.       |"

            castlines = castfile.readlines()
            success = False
            for i, l in enumerate(castlines):
                if drline1 in l:
                    if drline2 in castlines[i+1]:
                        success = True
                        break

            # Ok, now remove the CASTEP file
            castfile.close()
            os.remove(castfile.name)

        return success

    def finish_job(self, name, args, folder):
        """Save required output files to the output folder"""

        for cext in self.cp_ext:
            files = glob.glob(os.path.join(folder, '*'+cext))
            for f in files:
                shutil.move(f, os.path.join(self.folder_out,
                                            os.path.basename(f)))

    def finish_run(self):
        """Try removing the temporary keywords directory"""
        try:
            shutil.rmtree(self.kwdir)
        except OSError:
            self.log("Could not delete temporary castep_keywords.py"
                     "directory at {0}".format(self.kwdir))
        sys.path.remove(self.kwdir)

        if self.castpath is not None:
            sys.path.remove(castep_path)
