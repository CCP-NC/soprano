"""Run all tests in this folder"""

import os, sys
import glob
import subprocess as sp

tests_py = glob.glob(os.path.join(os.path.dirname(__file__), '*tests.py'))
failed = []
for t in tests_py:
    if sp.Popen(['python', t]).wait() != 0:
        failed += [t]

if len(failed) > 0:
    sys.stdout.write("The following tests failed:\n\t")
    print("\n\t".join(failed))