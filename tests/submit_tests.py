#!/usr/bin/env python
"""
Test code for QueueInterface and Submitters
"""


import os
import stat
import sys
import unittest
from pathlib import Path

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

# Use Path for cross-platform compatibility
_TESTCMD_DIR = Path(__file__).parent / "test_cmds"
_TESTDATA_DIR = Path(__file__).parent / "test_data"


class TestSubmit(unittest.TestCase):
    @pytest.mark.skipif(sys.platform == "win32", reason="Queue tests not supported on Windows")
    def test_queueint(self):
        from soprano.hpc.submitter import QueueInterface

        # Clean up the mock queue for any eventuality
        try:
            (_TESTCMD_DIR / "queue.pkl").unlink(missing_ok=True)
        except OSError:
            pass

        qInt = QueueInterface(
            sub_cmd=str(_TESTCMD_DIR / "mocksub.py"),
            list_cmd=str(_TESTCMD_DIR / "mocklist.py"),
            kill_cmd=str(_TESTCMD_DIR / "mockkill.py"),
            sub_outre="\\<(?P<job_id>[0-9]+)\\>",
            list_outre="(?P<job_id>[0-9]+)[^(RUN|PEND)]*" "(?P<job_status>RUN|PEND)",
        )

        test_id = qInt.submit("test_job 1 .")
        # Is the id correct?
        jobs = qInt.list()

        self.assertEqual(len(jobs), 1)
        self.assertEqual(list(jobs.keys())[0], test_id)

        # Now delete it
        qInt.kill(test_id)
        # And check
        jobs = qInt.list()

        self.assertEqual(len(jobs), 0)


if __name__ == "__main__":
    # For this to work we need to be sure that all files in _TESTCMD_DIR
    # are executable
    mockfiles = ["mocksub.py", "mocklist.py", "mockkill.py"]
    for mf in mockfiles:
        mfpath = os.path.join(_TESTCMD_DIR, mf)
        try:
            st = os.stat(mfpath)
        except OSError:
            sys.exit("Mock queue system for test not found")
        os.chmod(mfpath, st.st_mode | stat.S_IEXEC)

    # Remove the pipe if left over
    try:
        os.remove(".test_sub.fifo")
    except OSError:
        pass
    # Then add the folder to the system's PATH temporarily and we're good to go
    os.environ["PATH"] += ":" + _TESTCMD_DIR

    unittest.main()
