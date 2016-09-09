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
import subprocess as sp

from soprano.hpc.submitter import QueueInterface, Submitter, CastepSubmitter

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

    def test_submitter(self):

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

        # Now create a Submitter sub class

        subm = Submitter('test_sub', qInt, '<name> 1', max_time=10,
                         check_time=0.2)
        subm.start()
        print("\nSubmitter launched")
        time.sleep(2)
        # Now kill it
        Submitter.stop('test_sub')
        print("Submitter stopped")
        
    """
    def test_castep_submitter(self):

        from ase import Atoms
        from ase.io.castep import write_castep_cell

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

        # Create some folders
        in_fold = os.path.join(_TESTDATA_DIR, 'hpc_casteptest_in')
        out_fold = os.path.join(_TESTDATA_DIR, 'hpc_casteptest_out')

        try:
            shutil.rmtree(in_fold)
        except OSError:
            pass
        try:
            shutil.rmtree(out_fold)
        except OSError:
            pass

        os.makedirs(in_fold)
        os.makedirs(out_fold)

        # Now create the input files and save them
        n_tests = 5
        for i, l in enumerate(np.linspace(2, 4, n_tests)):
            a = Atoms(str('Si'), cell=[l]*3, pbc=[True]*3,
                      scaled_positions=[[0.5]*3])
            write_castep_cell(open(os.path.join(in_fold,
                                                'test{0}.cell'.format(i)),
                                   'w'), a)

        ctime = 0.2
        subm = CastepSubmitter('test_sub', qInt, '<name> 0.4 <folder>',
                               max_time=20, check_time=ctime)

        subm.set_parameters(in_fold, out_fold, 'castep.serial')

        subm.start()
        # Wait it out
        time.sleep(ctime*n_tests*2)
        # Stop it
        Submitter.stop('test_sub')
        # Check the results        
        self.assertEqual(len(glob.glob(os.path.join(out_fold, '*.castep'))), 
                         n_tests)
    """

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