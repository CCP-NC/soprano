"""
Definition of Submitter class

Base class for all Submitters to inherit from.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import pickle
import shutil
import signal
import tempfile
import numpy as np
import threading as thr
import subprocess as sp

from soprano.utils import is_string
from soprano.hpc.submitter import QueueInterface

class Submitter(object):

    """Submitter object

    Template to derive all specialised Submitters. These are meant to generate,
    submit and post-process any number of jobs on a queueing system in the form
    of a background process running on a head node. It implements
    methods that should be mostly overridden by the child classes.
    Six methods define its core behaviour:

    1) next_job is the function that outputs the specification for each new job
       to submit. The specification should be a dict with two members, 'name'
       (a string) and 'args' (ideally a dict). If no more jobs are available
       it should return None;
    2) setup_job takes as arguments name, args and folder (a temporary one
       created independently) and is supposed to generate the input files
       for the job before submission. It returns nothing;
    3) check_job takes as arguments job ID, name, args and folder and should
       return a bool confirmation of whether the job has finished or not. By
       default it simply checks whether the job is still listed in the queue,
       however other checks can be implemented in its place;
    4) finish_job takes as arguments name, args and folder and takes care of
       the post processing once a job is complete. Here meaningful data should
       be extracted and useful files copied to permament locations, as the
       temporary folder will be deleted immediately afterwards. It returns
       nothing;
    5) start_run takes no arguments, executes at the beginning of a run,
       before the pipe is opened;
    6) finish_run takes no arguments, executes at the end of a run, after the
       pipe is closed.

    In addition, the Submitter takes a template launching script which can
    be tagged with keywords, mainly <name> for the job name or any other
    arguments present in args. These will be replaced with the appropriate
    values when the script is submitted.
    """

    def __init__(self, name, queue, submit_script, max_jobs=4, check_time=10,
                 max_time=3600):
        """Initialize the Submitter object

        | Args:
        |   name (str): name to be used for this Submitter (two Submitters
        |               with the same name can't be launched in the same
        |               working directory)
        |   queue (QueueInterface): object describing the properties of the 
        |                           interface to the queue system in use
        |   submit_script (str): text of the script to use when submitting a
        |                        job to the queue. All tags of the form <name>
        |                        will be replaced with the job's name, and all
        |                        similar tags of the form <[arg]> will be
        |                        replaced if the argument name is present in
        |                        the job's args dictionary
        |   max_jobs (Optional[int]): maximum number of jobs to submit at a 
        |                             given time. Default is 4
        |   check_time (Optional[float]): time in seconds between consecutive
        |                                 checks for the queue status and 
        |                                 attempts to submit new jobs. Default
        |                                 is 10
        |   max_time (Optional[float]): time in seconds the Submitter thread
        |                               will run for before shutting down. If
        |                               set to zero the thread won't stop
        |                               until killed with Submitter.stop.

        """

        # Check type
        if not isinstance(queue, QueueInterface):
            raise TypeError('A QueueInterface must be passed to the '
                            'Submitter')

        if not is_string(submit_script):
            raise TypeError('submit_script must be a string')

        self.name = name
        self.queue = queue
        self.submit_script = submit_script
        self.max_jobs = max_jobs
        self.check_time = check_time
        self.max_time = max_time if max_time > 0 else np.inf

    def set_parameters(self):
        """Set additional parameters. In this generic example class it has
        no arguments, but in specific implementations it will be used to
        add more variables without overriding __init__."""

        pass        

    def start(self):

        self._jobs = {}

        self._running = True
        self._t0 = time.time()  # Starting time. Second precision is fine

        # Initialise signal catching
        signal.signal(signal.SIGINT, self._catch_signal)
        signal.signal(signal.SIGTERM, self._catch_signal)

        self.start_run()
        self._main_loop()
        self.finish_run()

    def _catch_signal(self, signum, frame):
        # This catches the signal when termination is asked
        self._running = False
        # Also, kill all jobs still running
        for job_id in self._jobs:
            self.queue.kill(job_id)
            self.finish_job(**self._jobs[job_id])
            shutil.rmtree(self._jobs[job_id]['folder'])

    def _main_loop(self):
        """Main loop run as separate thread. Should not be edited when
        inheriting from the class"""

        while self._running and (time.time()-self._t0) < self.max_time:

            loop_t0 = time.time()

            # Submit jobs
            while len(self._jobs) < self.max_jobs:

                njob = self.next_job()
                if njob is None:
                    break
                # Create the temporary folder
                njob['folder'] = tempfile.mkdtemp()
                # Perform setup
                self.setup_job(**njob)
                # Create custom script
                job_script = self.submit_script.replace('<name>',
                                                        njob['name'])
                # Replace the rest of the tags
                for tag in njob['args']:
                    job_script = job_script.replace('<{0}>'.format(tag),
                                                    str(njob['args'][tag]))
                job_script = job_script.replace('<folder>',
                                                njob['folder'])

                # And submit!
                job_id = self.queue.submit(job_script, cwd=njob['folder'])
                self._jobs[job_id] = njob

            # Now check for finished jobs
            completed = [job_id for job_id in self._jobs
                         if self.check_job(job_id, **self._jobs[job_id])]
            for job_id in completed:
                self.finish_job(**self._jobs[job_id])
                # Remove the temporary directory
                shutil.rmtree(self._jobs[job_id]['folder'])
                # Finally delete it from our list
                del(self._jobs[job_id])

            sleep_time = self.check_time - (time.time()-loop_t0)
            sleep_time = sleep_time if sleep_time > 0 else 0
            time.sleep(sleep_time)

    def next_job(self):
        """Return a dictionary definition of the next job in line"""
        return {'name': 'default_job', 'args': {}}

    def setup_job(self, name, args, folder):
        """Perform preparatory operations on the job"""
        pass

    def check_job(self, job_id, name, args, folder):
        """Checks if given job is complete or not"""
        return job_id not in self.queue.list()

    def finish_job(self, name, args, folder):
        """Performs completiion operations on the job. At this point any 
        relevant output files should be copied from 'folder' to their final
        destination as the temporary folder itself will be deleted immediately
        after"""
        pass

    def start_run(self):
        """Operations to perform when the daemon thread starts running"""
        pass

    def finish_run(self):
        """Operations to perform after the daemon thread stops running"""
        pass

    @staticmethod
    def stop(name):
        """Stop daemon of given name"""
        fname = '.{0}.submitter'.format(name)
        sp.Popen(['pkill', '-f', fname])