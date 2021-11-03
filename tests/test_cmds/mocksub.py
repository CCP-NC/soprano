#!/usr/bin/env python

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import pickle
import numpy as np

# A "mock" queue submission system to test how well QueueInterface works
# Submission command

mydir = os.path.dirname(os.path.realpath(__file__))

# For Python3 support
try:
    get_input = raw_input
except NameError:
    get_input = input

try:
    script_first_line = get_input()
except EOFError:
    script_first_line = ""

try:
    joblist = pickle.load(open(os.path.join(mydir, "queue.pkl"), "rb"))
except IOError:
    joblist = {}

# Well, add a job with the name of the script's first word
fline_spl = script_first_line.split()
jobname = fline_spl[0]
joblength = float(fline_spl[1])

rnd_id = np.random.randint(100000, 999999)
while rnd_id in joblist:
    rnd_id = np.random.randint(100000, 999999)
rnd_id = str(rnd_id)
joblist[rnd_id] = {"name": jobname, "end": time.time() + joblength}
if len(fline_spl) > 2:
    joblist[rnd_id]["path"] = fline_spl[2]

print("Job <{0}> submitted".format(rnd_id))

pickle.dump(joblist, open(os.path.join(mydir, "queue.pkl"), "wb"))
