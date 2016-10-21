#!/usr/bin/env python
"""
Test code for QueueInterface and Submitters
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import stat
import glob
import time
import shutil
import tempfile
import subprocess as sp

from soprano.hpc.submitter import QueueInterface, Submitter, CastepSubmitter
from soprano.hpc.submitter.debug import DebugQueueInterface

import unittest
import numpy as np

_TESTCMD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_cmds")
_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")


class TestSubmit(unittest.TestCase):

    def test_queueint(self):

        # Clean up the mock queue for any eventuality
        try:
            os.remove(os.path.join(_TESTCMD_DIR, 'queue.pkl'))
        except OSError:
            pass

        qInt = QueueInterface(sub_cmd='mocksub.py',
                              list_cmd='mocklist.py',
                              kill_cmd='mockkill.py',
                              sub_outre='\<(?P<job_id>[0-9]+)\>',
                              list_outre='(?P<job_id>[0-9]+)[^(RUN|PEND)]*'
                                         '(?P<job_status>RUN|PEND)')

        test_id = qInt.submit('test_job 1 .')
        # Is the id correct?
        jobs = qInt.list()

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs.keys()[0], test_id)

        # Now delete it
        qInt.kill(test_id)
        # And check
        jobs = qInt.list()

        self.assertEqual(len(jobs), 0)


if __name__ == "__main__":

    # For this to work we need to be sure that all files in _TESTCMD_DIR
    # are executable
    mockfiles = ['mocksub.py', 'mocklist.py', 'mockkill.py']
    for mf in mockfiles:
        mfpath = os.path.join(_TESTCMD_DIR, mf)
        try:
            st = os.stat(mfpath)
        except OSError:
            sys.exit('Mock queue system for test not found')
        os.chmod(mfpath, st.st_mode | stat.S_IEXEC)

    # Remove the pipe if left over
    try:
        os.remove('.test_sub.fifo')
    except OSError:
        pass
    # Then add the folder to the system's PATH temporarily and we're good to go
    os.environ['PATH'] += ":"+_TESTCMD_DIR

    unittest.main()
