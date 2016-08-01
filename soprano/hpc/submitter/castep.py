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

from soprano import utils
from soprano.hpc.submitter import Submitter
from ase.calculators.castep import create_castep_keywords


class CastepSubmitter(Submitter):

    def set_parameters(self, folder_in, folder_out, castep_command,
                             castep_path=None,
                             copy_extensions=['.castep']):
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
        |   copy_extensions (Optional[list[str]]): extensions of output files to
        |                                          copy to the output folder (by
        |                                          default only .castep file)

        """

        self.folder_in = folder_in
        self.folder_out = folder_out
        self.castep_command = castep_command
        self.cp_ext = copy_extensions

        # Initialize the CASTEP keywords file in a dedicated temporary folder
        self.kwdir = tempfile.mkdtemp()
        self.castpath = castep_path
        create_castep_keywords(castep_command,
                               os.path.join(self.kwdir, 
                                            'castep_keywords.py'))
        sys.path.append(self.kwdir)

        # Finally add the castep_path to PATH if needed
        if castep_path is not None:
            sys.path.append(castep_path)

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

    def finish_job(self, name, args, folder):
        """Save required output files to the output folder"""

        for cext in self.cp_ext:
            files = glob.glob(os.path.join(folder, '*'+cext))
            for f in files:
                shutil.move(f, self.folder_out)                

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

