"""
Definition of Submitter class and script to start/stop it.

Base class for all Submitters to inherit from.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import shutil
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
    Four methods define its core behaviour:

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
       nothing.

    In addition, the Submitter takes a template launching script which can
    be tagged with keywords, mainly <name> for the job name or any other
    arguments present in args. These will be replaced with the appropriate
    values when the script is submitted.
    """

    def __init__(self, name, queue, submit_script, max_jobs=4, check_time=10,
                 max_time=3600):

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

        # Message pipe. Used to stop the thread from the outside
        self._fifo_path = ".{0}.fifo".format(name)
        # And create the FIFO PIPE
        try:
            os.mkfifo(self._fifo_path, 0777)
        except OSError:
            raise RuntimeError('A Submitter with the given name already '
                               'exists')

    def start(self):

        self._jobs = {}

        self._running = True
        self._t0 = time.time()  # Starting time. Second precision is fine

        mthr = thr.Thread(target=self.main_loop)
        mthr.daemon = True
        mthr.start()

    def main_loop(self):

        self._pipe = os.open(self._fifo_path, os.O_RDONLY|os.O_NONBLOCK)

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
                                                    njob['args'][tag])
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

            # Finally, grab messages from the PIPE
            msg = os.read(self._pipe, 16)
            if 'STOP' in msg:
                self._running = False

            sleep_time = self.check_time - (time.time()-loop_t0)
            sleep_time = sleep_time if sleep_time > 0 else 0
            time.sleep(sleep_time)

        os.close(self._pipe)

    def next_job(self):
        return {'name': 'default_job', 'args': {}}

    def setup_job(self, name, args, folder):
        pass

    def check_job(self, job_id, name, args, folder):
        return job_id in self.queue.list()

    def finish_job(self, name, args, folder):
        pass

    @staticmethod
    def stop(name):
        """Stop a Submitter thread, given its name"""

        pipename = ".{0}.fifo".format(name)

        try:
            pipe = os.open(pipename,
                           os.O_WRONLY|os.O_NONBLOCK)
            os.write(pipe, 'STOP')
            os.close(pipe)
        except OSError:
            raise RuntimeError('No Submitter with the given name could be '
                               'found')
        os.remove(pipename)