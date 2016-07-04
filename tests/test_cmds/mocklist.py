#!/usr/bin/env python

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import pickle

# A "mock" queue submission system to test how well QueueInterface works
# Listing command

mydir = os.path.dirname(os.path.realpath(__file__))
try:
	joblist = pickle.load(open(os.path.join(mydir, 'queue.pkl')))
	if len(joblist) == 0:
		raise IOError()
except IOError:
	sys.exit("No unfinished jobs found")

for job in joblist:
	print(("{0}\tusername\tRUN\tqueuename\tnodeURL\t{1}"
		   "MM DD HH:MM").format(job, joblist[job]))
