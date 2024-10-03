#!/usr/bin/env python


import os
import pickle
import sys
import time

# A "mock" queue submission system to test how well QueueInterface works
# Listing command

mydir = os.path.dirname(os.path.realpath(__file__))
try:
    joblist = pickle.load(open(os.path.join(mydir, "queue.pkl"), "rb"))
    if len(joblist) == 0:
        raise OSError
except OSError:
    sys.exit("No unfinished jobs found")

# Check if any jobs are finished and in case remove them
joblist_updated = {}
for job in joblist:
    print("{0} => {1}".format(joblist[job]["end"], time.time()))
    if joblist[job]["end"] > time.time():
        joblist_updated[job] = joblist[job]
    elif "path" in joblist[job]:
        open(
            os.path.join(joblist[job]["path"], joblist[job]["name"] + ".castep"),
            "w",
        ).write("Fake CASTEP")

joblist = joblist_updated

for job in joblist:
    print(
        ("{0}\tusername\tRUN\tqueuename\tnodeURL\t{1}" "\tMM DD HH:MM").format(
            job, joblist[job]["name"]
        )
    )

pickle.dump(joblist, open(os.path.join(mydir, "queue.pkl"), "wb"))
