#!/usr/bin/env python

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import pickle
import numpy as np

# A "mock" queue submission system to test how well QueueInterface works
# Submission command

mydir = os.path.dirname(os.path.realpath(__file__))

try:
	script_first_line = raw_input()
except EOFError:
	script_first_line = ""

try:
	joblist = pickle.load(open(os.path.join(mydir, 'queue.pkl')))
except IOError:
	joblist = {}

# Well, add a job with the name of the script's first word
jobname = script_first_line.split()[0]
rnd_id = np.random.randint(100000, 999999)
while rnd_id in joblist:
	rnd_id = np.random.randint(100000, 999999)
joblist[str(rnd_id)] = jobname
print("Job <{0}> submitted".format(rnd_id))

pickle.dump(joblist, open(os.path.join(mydir, 'queue.pkl'), 'w'))
