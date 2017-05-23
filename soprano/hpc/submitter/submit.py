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
from socket import timeout as TimeoutError

from soprano.utils import is_string, safe_communicate
from soprano.hpc.submitter import QueueInterface


class Submitter(object):

    """Submitter object

    Template to derive all specialised Submitters. These are meant to generate,
    submit and post-process any number of jobs on a queueing system in the form
    of a background process running on a head node. It implements
    methods that should be mostly overridden by the child classes.
    The following methods define its core behaviour:

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
    7) save_state takes no arguments, returns a dict. It is executed when
       continuation=True is used and a run terminates. It will allow the user
       to add class-specific data to the dictionary that is stored in the
       pickle file (in addition to the default, namely the list and info on
       currently running jobs). This should be used for example to store state
       information that is necessary for job generation. It should be composed
       of serialisable objects.
    8) load_state takes as arguments the loaded data in dictionary form. It
       should perform the reverse operation of save_state, grabbing the info
       and restoring the Submitter's state to its previous condition.

    In addition, the Submitter takes a template launching script which can
    be tagged with keywords, mainly <name> for the job name or any other
    arguments present in args. These will be replaced with the appropriate
    values when the script is submitted.
    """

    def __init__(self, name, queue, submit_script, max_jobs=4, check_time=10,
                 max_time=3600, temp_folder=None, remote_workdir=None,
                 remote_getfiles=['*.*'], ssh_timeout=1.0,
                 continuation=False):
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
        |                                the current folder.
        |   remote_workdir (Optional[str]): if present, uses a directory on a
        |                                   remote machine by logging in via
        |                                   SSH. Must be in the format
        |                                   <host>:<path/to/directory>.
        |                                   Host must be defined in the user's
        |                                   ~/.ssh/config file - check the
        |                                   docs for RemoteTarget for more
        |                                   information. It is possible to
        |                                   omit the colon and directory, that
        |                                   will use the home directory of the
        |                                   given folder; that is HEAVILY
        |                                   DISCOURAGED though. Best practice
        |                                   would be to create an empty
        |                                   directory on the remote machine
        |                                   and use that, to avoid accidental
        |                                   overwriting/deleting of important
        |                                   files.
        |   remote_getfiles (Optional[list(str)]): list of files to be
        |                                          downloaded from the remote
        |                                          copy of the job's temporary
        |                                          directory. By default, all
        |                                          of them. Can be a list
        |                                          using specific names,
        |                                          wildcards etc. Filenames
        |                                          can also use the
        |                                          placeholder {name} to
        |                                          signify the job name, as
        |                                          well as any other element
        |                                          from the arguments.
        |   ssh_timeout (Optional[float]): connection timeout in seconds
        |                                  (default is 1 second)
        |   continuation (Optional[bool]): if True, when the Submitter is
        |                                  stopped it will not terminate the
        |                                  current jobs; rather, it will store
        |                                  the list in a pickle file.
        |                                  If the submitter is ran from the
        |                                  same folder then it will "pick up
        |                                  from where it left" and try
        |                                  recovering those jobs, then
        |                                  restart. If one wishes for
        |                                  additional values to be saved and
        |                                  restored, the save_state and
        |                                  load_state methods need to be
        |                                  defined.

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

        # Remote directory?
        if remote_workdir is None:
            self.host = None
        else:
            if ':' in remote_workdir:
                self.host, self.hostdir = remote_workdir.split(':', 1)
            else:
                self.host = remote_workdir
                self.hostdir = ''

        self.remote_getfiles = remote_getfiles

        self.queue.set_remote_host(self.host, ssh_timeout)

        self.continuation = continuation

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

        self._log = open(self.name + '.log', 'w')
        self.log('Starting run on {0}\n'.format(datetime.now()))

        self._jobs = {}
        self._waiting_jobs = []  # Jobs created but not yet submitted
        self._completed_jobs = []  # Jobs completed but not yet finalised

        if self.continuation and os.path.exists(self._pklname):
            self._load()

        self._running = True
        self._t0 = time.time()  # Starting time. Second precision is fine

        # Initialise signal catching
        signal.signal(signal.SIGINT, self._catch_signal)
        signal.signal(signal.SIGTERM, self._catch_signal)
        # Now assign the user-defined ones
        for cmd in self._user_signals:
            signum, cback = self._user_signals[cmd]
            signal.signal(signum, self._catch_signal)

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
            self._terminate()
        else:
            for cmd in self._user_signals:
                if signum == self._user_signals[cmd][0]:
                    self._user_signals[cmd][1]()

    def _terminate(self):
        self._running = False
        # Also, kill all jobs still running
        if not self.continuation:
            for job_id in self._jobs.keys():
                self.queue.kill(job_id)
                # If needed, get the files from remote host
                if self.host is not None:
                    self._getjob_remote(self._jobs[job_id])
                self.finish_job(**self._jobs[job_id])
                shutil.rmtree(self._jobs[job_id]['folder'])
        else:
            self._save()

        self._jobs = {}

    def _main_loop(self):
        """Main loop run as separate thread. Should not be edited when
        inheriting from the class"""

        while self._running and (time.time()-self._t0) < self.max_time:

            loop_t0 = time.time()

            # Submit jobs
            while len(self._jobs) < self.max_jobs:

                if len(self._waiting_jobs) == 0:
                    njob = self.next_job()
                    if njob is None:
                        break
                    # Create the temporary folder
                    njob['folder'] = tempfile.mkdtemp(dir=self.tmp_dir)
                    # Perform setup
                    if not self.setup_job(**njob):
                        self.log('Job {0} did not pass setup check, '
                                 'skipping\n'.format(njob['name']))
                        # Remove the temporary directory
                        shutil.rmtree(njob['folder'])
                        self.log(('Folder {0} '
                                  'deleted\n').format(njob['folder']))
                        continue
                else:
                    njob = self._waiting_jobs.pop(0)

                # If we're working with a remote host, we need to copy the
                # input folder!
                if self.host is not None:
                    try:
                        self._putjob_remote(njob)
                    except TimeoutError:
                        self.log('Timeout when trying to push input files to'
                                 ' host. If it happens too frequently, try '
                                 'increasing the ssh_timeout argument.\n')
                        self._waiting_jobs.append(njob)
                        break

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

                    if self.host is not None:
                        cwd = os.path.join(self.hostdir, njob['name'])
                    else:
                        cwd = njob['folder']

                    job_id = self.queue.submit(job_script,
                                               cwd=cwd)
                    self._jobs[job_id] = njob

            # Now check for finished jobs
            self._completed_jobs += [job_id for job_id in self._jobs
                                     if self.check_job(job_id,
                                                       **self._jobs[job_id])]

            for job_id in self._completed_jobs:
                cjob = self._jobs[job_id]
                # If needed, get the files from remote host
                if self.host is not None:
                    try:
                        self._getjob_remote(cjob)
                    except TimeoutError:
                        self.log('Timeout when trying to fetch results from'
                                 ' host. If it happens too frequently, try '
                                 'increasing the ssh_timeout argument.\n')
                        continue
                self.log('Job {0} completed\n'.format((cjob['name'])))
                self.finish_job(**cjob)
                # Remove the temporary directory
                shutil.rmtree(cjob['folder'])
                self.log('Folder {0} deleted\n'.format(cjob['folder']))
                # Finally delete it from our list
                del(self._jobs[job_id])

            # Only keep the ones that haven't been copied yet
            self._completed_jobs = list(set(self._completed_jobs) &
                                        set(self._jobs.keys()))

            sleep_time = self.check_time - (time.time()-loop_t0)
            sleep_time = sleep_time if sleep_time > 0 else 0
            time.sleep(sleep_time)

    def _putjob_remote(self, njob):
        """Copy the files generated for a job to a remote work directory"""
        with self.queue.remote_target.context as rtarg:
            rtarg.run_cmd('mkdir {0}'.format(njob['name']),
                          cwd=self.hostdir)
            rtarg.put_files(os.path.join(njob['folder'], '*'),
                            os.path.join(self.hostdir,
                                         njob['name']))

    def _getjob_remote(self, cjob):
        with self.queue.remote_target.context as rtarg:
            rpaths = [os.path.join(self.hostdir, cjob['name'],
                                   f.format(name=cjob['name'],
                                            **cjob['args']))
                      for f in self.remote_getfiles]
            rtarg.get_files(rpaths, cjob['folder'])
            rtarg.run_cmd('rm -rf {0}'.format(cjob['name']), cwd=self.hostdir)

    def _save(self):

        savedata = {}

        try:
            savedata.update(self.save_state())
        except TypeError:
            raise TypeError("Method save_state must return a dict")

        to_save = ['_jobs', '_waiting_jobs', '_completed_jobs']

        for k in to_save:
            savedata[k] = getattr(self, k)

        self.log('Saving data for future runs...\n')
        try:
            pickle.dump(savedata, open(self._pklname, 'w'))
        except IOError:
            self.log('Saving failed\n')

    def _load(self):

        to_load = ['_jobs', '_waiting_jobs', '_completed_jobs']

        self.log('Loading data from past runs...\n')
        try:
            loaddata = pickle.load(open(self._pklname))
        except IOError:
            self.log('Loading failed\n')
            return

        for k in to_load:
            setattr(self, k, loaddata[k])
            del loaddata[k]

        self.load_state(loaddata)

    def save_state(self):
        """Return a dictionary containing serialisable data to be saved from
        one run to the next"""
        return {}

    def load_state(self, loaded):
        """Replace attributes from loaded data in dictionary form"""
        return

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
        """Performs completion operations on the job. At this point any
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

    @property
    def _pklname(self):
        return self.name + '_save.pkl'
