"""Run all tests in this folder"""

import time
import os
import sys
import glob
import subprocess as sp

tests_py = glob.glob(os.path.join(os.path.dirname(__file__), "*tests.py"))
failed = []
t0 = time.time()
for t in tests_py:
    if sp.Popen(["python", t]).wait() != 0:
        failed += [t]

print("All tests completed")
print("Time spent: {0: .3f} s".format(time.time() - t0))
if len(failed) > 0:
    sys.stdout.write("The following tests failed:\n\t")
    print("\n\t".join(failed))

    sys.exit(1)  # Remark that an error has happened
