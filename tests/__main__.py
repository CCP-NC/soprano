"""Run all tests in this folder"""

import glob
import os
import subprocess as sp
import sys
import time

tests_py = glob.glob(os.path.join(os.path.dirname(__file__), "*tests.py"))
failed = []
t0 = time.time()
for t in tests_py:
    if sp.Popen(["python", t]).wait() != 0:
        failed += [t]

print("All tests completed")
print(f"Time spent: {time.time() - t0: .3f} s")
if len(failed) > 0:
    sys.stdout.write("The following tests failed:\n\t")
    print("\n\t".join(failed))

    sys.exit(1)  # Remark that an error has happened
