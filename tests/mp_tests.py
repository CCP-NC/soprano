#!/usr/bin/env python
"""
Test code for HPC Daemons and the ilk
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
from ase import io, Atoms
sys.path.insert(0, os.path.abspath(
                   os.path.join(os.path.dirname(__file__), "../")))  # noqa
from soprano.collection import AtomsCollection
from soprano.hpc.daemons import DaemonRunner
from soprano.hpc.daemons.castep import CastepDaemon
import unittest
import numpy as np


_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_data")


class TestMP(unittest.TestCase):

    def test_castep(self):

        from ase.io.castep import write_castep_cell

        n_tests = 5

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
        for i, l in enumerate(np.linspace(2, 4, n_tests)):
            a = Atoms(str('Si'), cell=[l]*3, pbc=[True]*3,
                      scaled_positions=[[0.5]*3])
            write_castep_cell(open(os.path.join(in_fold,
                                                'test{0}.cell'.format(i)),
                                   'w'), a)

        # Load some test files
        reslist=glob.glob(
            os.path.join(_TESTDATA_DIR, 'rescollection', '*.res'))
        testcoll=AtomsCollection(reslist)

        # Create a Daemon runner containing this collection
        testdaemon=DaemonRunner(CastepDaemon,
                                  daemon_args={'folder_in': in_fold,
                                               'folder_out': out_fold,
                                               'castep_command':
                                               'castep.serial'},
                                  proc_num=3,
                                  verbose=True)


if __name__ == '__main__':
    unittest.main()
