#!/usr/bin/env python

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import pickle

kill_id = sys.argv[1]
mydir = os.path.dirname(os.path.realpath(__file__))
try:
    qfile = open(os.path.join(mydir, "queue.pkl"), "rb")
    joblist = pickle.load(qfile)
    qfile.close()
    if kill_id not in joblist:
        raise IOError()
except IOError:
    sys.exit("Job <{0}> not found".format(kill_id))

# If it's in joblist, delete it
del joblist[kill_id]
print("Job <{0}> has been terminated".format(kill_id))

if len(joblist) > 0:
    pickle.dump(joblist, open(os.path.join(mydir, "queue.pkl"), "wb"))
else:
    os.remove(os.path.join(mydir, "queue.pkl"))
