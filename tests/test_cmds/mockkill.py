#!/usr/bin/env python


import os
import pickle
import sys

kill_id = sys.argv[1]
mydir = os.path.dirname(os.path.realpath(__file__))
try:
    qfile = open(os.path.join(mydir, "queue.pkl"), "rb")
    joblist = pickle.load(qfile)
    qfile.close()
    if kill_id not in joblist:
        raise OSError
except OSError:
    sys.exit(f"Job <{kill_id}> not found")

# If it's in joblist, delete it
del joblist[kill_id]
print(f"Job <{kill_id}> has been terminated")

if len(joblist) > 0:
    pickle.dump(joblist, open(os.path.join(mydir, "queue.pkl"), "wb"))
else:
    os.remove(os.path.join(mydir, "queue.pkl"))
