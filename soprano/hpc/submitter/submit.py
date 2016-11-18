# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
from datetime import datetime

from soprano.utils import is_string, safe_communicate
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
       for the job before submission. It returns a boolean, confirming that
       the setup went well; if False, the job will be skipped;
    3) check_job takes as arguments job ID, name, args and folder and should
       return a bool confirmation of whether the job has finished or not. By
       default it simply checks whether the job is still listed in the queue,
       however other checks can be implemented in its place;
    4) finish_job takes as arguments name, args and folder and takes care of
       the post processing once a job is complete. Here meaningful data should
       be extracted and useful files copied to permament locations, as the
       temporary folder will be deleted immediately afterwards. It returns
       nothing;
    5) start_run takes no arguments, executes at the beginning of a run;
    6) finish_run takes no arguments, executes at the end of a run.

    In addition, the Submitter takes a template launching script which can
    be tagged with keywords, mainly <name> for the job name or any other
    arguments present in args. These will be replaced with the appropriate
    values when the script is submitted.
    """

    def __init__(self, name, queue, submit_script, max_jobs=4, check_time=10,
                 max_time=3600, temp_folder=None):
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
        |   max_time (Optional[float]): time in seconds the Submitter will run
        |                               for before shutting down. If set to
        |                               zero the thread won't stop until
        |                               killed with Submitter.stop.
        |   temp_folder (Optional[str]): where to store the temporary folders
        |                                for the calculations. By default it's
        |                                the system's tmp/ folder, but might
        |                                be changed if there's a need because
        |                                of writing permissions.

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
        self.tmp_dir = (os.path.abspath(temp_folder)
                        if temp_folder is not None else '')

        # User defined signals
        self._free_signals = [signal.__dict__[s] for s in ('SIGUSR1',
                                                           'SIGUSR2')
                              if s in signal.__dict__]
        self._user_signals = {}

        self._log = None  # Will keep track of failed jobs etc.

    def set_parameters(self):
        """Set additional parameters. In this generic example class it has
        no arguments, but in specific implementations it will be used to
        add more variables without overriding __init__."""

        pass

    def add_signal(self, command, callback):
        """Add a signal listener to this submitter. Unix systems only allow
        for up to TWO user-defined signals to be specified.

        | Args:
        |   command (str): command that should be used to call this signal.
        |                  This would be used as:
        |                  python -m soprano.hpc.submitter <command> <file>
        |                  and will trigger the callback's execution
        |   callback (function<self> => None): method of the user defined
        |                      Submitter class to use as a callback when the
        |                      given signal is sent. Should accept and return
        |                      nothing.

        """

        # Is the command a reserved word?
        if command in ('start', 'stop', 'list'):
            raise ValueError('The commands start, stop and list are reserved'
                             ' and can not be used for custom signals.')

        # Are there any free signals left?
        try:
            signum = self._free_signals.pop()
        except IndexError:
            raise RuntimeError('Impossible to assign more signals - maximum '
                               'number supported by the OS has been reached.')

        self._user_signals[command] = [signum, callback]

    def remove_signal(self, command):
        """Remove a previously defined custom signal by its assigned command.

        | Args:
        |   command (str): command assigned to the signal handler to remove.        

        """

        try:
            signum, _ = self._user_signals[command]
            del(self._user_signals[command])
            self._free_signals.append(signum)
        except KeyError:
            raise ValueError('Command does not correspon to any assigned'
                             ' signal.')

    def start(self):

        self._jobs = {}

        self._running = True
        self._t0 = time.time()  # Starting time. Second precision is fine

        # Initialise signal catching
        signal.signal(signal.SIGINT, self._catch_signal)
        signal.signal(signal.SIGTERM, self._catch_signal)
        # Now assign the user-defined ones
        for cmd in self._user_signals:
            signum, cback = self._user_signals[cmd]
            signal.signal(signum, self._catch_signal)

        self._log = open(self.name + '.log', 'w')
        self.log('Starting run on {0}\n'.format(datetime.now()))

        # Just a sanity check
        has_name = False
        for l in self.submit_script.split('\n'):
            if '<name>' in l.split('#')[0]:
                has_name = True
                break

        if not has_name:
            self.log('WARNING: the submission script does not contain the '
                     '<name> tag in any non-commented lines. '
                     'This is most likely an error - check your '
                     'input files\n')

        self.start_run()
        self._main_loop()
        self.finish_run()
        self.log('Run finished on {0}\n'.format(datetime.now()))
        self._log.close()

    def _catch_signal(self, signum, frame):
        if signum in (signal.SIGINT, signal.SIGTERM):
            # This catches the signal when termination is asked
            self.log('SIGTERM received - '
                     'Starting termination of this run...\n')
            self._running = False
            # Also, kill all jobs still running
            for job_id in self._jobs.keys():
                self.queue.kill(job_id)
                self.finish_job(**self._jobs[job_id])
                shutil.rmtree(self._jobs[job_id]['folder'])
            self._jobs = {}
        else:
            for cmd in self._user_signals:
                if signum == self._user_signals[cmd][0]:
                    self._user_signals[cmd][1]()

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
                njob['folder'] = tempfile.mkdtemp(dir=self.tmp_dir)
                # Perform setup
                if not self.setup_job(**njob):
                    self.log('Job {0} did not pass setup check,'
                             'skipping\n').format(njob['name'])
                    continue
                # Create custom script
                job_script = self.submit_script.replace('<name>',
                                                        njob['name'])
                # Replace the rest of the tags
                for tag in njob['args']:
                    job_script = job_script.replace('<{0}>'.format(tag),
                                                    str(njob['args'][tag]))
                job_script = job_script.replace('<folder>',
                                                njob['folder'])

                # And submit! [Only if still running]
                if not self._running:
                    break
                else:
                    self.log('Submitting job '
                             '{0} to queue\n'.format(njob['name']))
                    job_id = self.queue.submit(job_script,
                                               cwd=njob['folder'])
                    self._jobs[job_id] = njob

            # Now check for finished jobs
            completed = [job_id for job_id in self._jobs
                         if self.check_job(job_id, **self._jobs[job_id])]
            for job_id in completed:
                cjob = self._jobs[job_id]
                self.log('Job {0} completed\n'.format((cjob['name'])))
                self.finish_job(**cjob)
                # Remove the temporary directory
                shutil.rmtree(cjob['folder'])
                self.log('Folder {0} deleted\n'.format(cjob['folder']))
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
        return True

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

    def log(self, logtxt):
        self._log.write(logtxt)
        self._log.flush()

    @staticmethod
    def stop(fname, subname):
        """Stop Submitter process from filename and name,
        return False if failed"""
        return sp.Popen(['pkill', '-f', fname + ' ' + subname]).wait() == 0

    @staticmethod
    def list():
        list_proc = sp.Popen(['ps', 'aux'],
                             stdin=sp.PIPE,
                             stdout=sp.PIPE,
                             stderr=sp.PIPE)
        stdout, stderr = safe_communicate(list_proc)
        # Parse stdout
        all_subms = []
        for l in stdout.split('\n'):
            if 'soprano.hpc.submitter._spawn' in l:
                lspl = l.split()
                # File, name, time, PID
                subm = lspl[-3], lspl[-2], lspl[-7], int(lspl[1])
                all_subms.append(subm)

        return all_subms
