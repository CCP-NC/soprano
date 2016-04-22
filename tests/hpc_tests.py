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

class TestHPC(unittest.TestCase):

    def test_castep(self):

        # Load some test files
        reslist = glob.glob(os.path.join(_TESTDATA_DIR, 'rescollection', '*.res'))
        testcoll = AtomsCollection(reslist)

        # Create a Daemon runner containing this collection
        testdaemon = DaemonRunner(CastepDaemon,
                                  daemon_args={'structures':
                                               testcoll.structures},
                                  proc_num=3,
                                  verbose=True)


if __name__ == '__main__':
    unittest.main()
