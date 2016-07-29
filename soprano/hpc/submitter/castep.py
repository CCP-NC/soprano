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

import sys
import numpy as np

from soprano.hpc.submitter import Submitter


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
        create_castep_keywords(castep_command,
                               os.path.join(self.kwdir, 
                                            'castep_keywords.py'))
        sys.path.append(self.kwdir)

        # Finally add the castep_path to PATH if needed
        if castep_path is not None:
            sys.path.append(castep_path)

    

