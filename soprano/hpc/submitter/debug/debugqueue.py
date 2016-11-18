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
Definition of a fake QueueInterface class, useful for debugging Submitters.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import numpy as np

import subprocess as sp
from threading import Thread
from soprano.utils import safe_communicate
from soprano.hpc.submitter.queues import QueueInterface


class DebugQueueInterface(QueueInterface):

    """DebugQueueInterface object

    A class meant to emulate a QueueInterface while doing absolutely nothing
    of what it does. Jobs are simply stored locally, there's a fixed waiting
    time, and are then executed. Ideally they should be simple, quick stuff
    (like an echo command). No guarantees for actually long jobs.

    In the submitted script a syntax for additional variables is allowed,
    similar to real queue systems. These follow the convention of many engines
    of having to start with #$. For example

    #$ WAIT 10

    means the job will be put in a "wait" state for 10 seconds. The currently
    available variables are:

    WAIT - specify how long the job has to stay in a "wait" state. If two
    values are provided, these are considered bounds for a random number
    RUN - same as above, but for the running state. This has no bearing on the
    *actual* running time (it's suggested that it's something very quick)

    """

    def __init__(self, dt=0.1):
        """Initialize the DebugQueueInterface.

        | Args:
        |   dt (float): frequency with which the queue status is updated

        """

        self._job_list = {}
        self._dt = dt

        self._runthr = None

    def _main_loop(self):

        while(len(self._job_list) > 0):
            t = time.time()
            completed_jobs = []
            for j_id in list(self._job_list.keys()):
                job = self._job_list[j_id]
                if job['status'] == 'w' and (t-job['t0']) > job['WAIT']:
                    # Switch to run
                    job['status'] = 'r'
                    # And actually run it!
                    proc = sp.Popen(['bash'], stdin=sp.PIPE,
                                    stdout=sp.PIPE,
                                    stderr=sp.PIPE,
                                    cwd=job['cwd'])
                    stdout, stderr = safe_communicate(proc, job['script'])
                elif job['status'] == 'r' \
                        and (t-job['t0']-job['WAIT']) > job['RUN']:
                    # Just eliminate it
                    completed_jobs.append(j_id)

            for cj in completed_jobs:
                del(self._job_list[cj])

            time.sleep(self._dt)

    def submit(self, script, cwd=None):
        """Submit a job to the queue.

        | Args:
        |   script (str): content of the submission script
        |   cwd (Optional[str]): path to the desired working directory
        |
        | Returns:
        |   job_id (str): the job ID assigned by the queue system and parsed
        |                 with sub_outre
        """

        # Parse the script for parameters
        job = {
            'WAIT': 0.0,
            'RUN': 0.0
        }

        for l in script.split('\n'):
            if l[:2] == '#$':
                # Parse!
                keyw, vals = l[2:].split(None, 1)
                if keyw in job:
                    if type(job[keyw]) is float:
                        # Ok, what do we have?
                        vals = vals.split()
                        if len(vals) == 1:
                            job[keyw] = float(vals[0])
                        elif len(vals) == 2:
                            job[keyw] = np.random.uniform(*map(float, vals))

        # Python 2-to-3 compatibility
        try:
            script = bytes(script, 'utf-8')
        except TypeError:
            pass

        # Ok, now add the actual script
        job['script'] = script
        job['cwd'] = cwd
        job['t0'] = time.time()
        job['status'] = 'w'
        # Generate a suitable id
        new_id = '1'
        while new_id in self._job_list:
            new_id = str(np.random.randint(10**(len(new_id)),
                                           10**(1+len(new_id))))

        self._job_list[new_id] = job
        if self._runthr is None or not(self._runthr.isAlive()):
            self._runthr = Thread(target=self._main_loop)
            self._runthr.daemon = True
            self._runthr.start()

        return new_id

    def list(self):
        """List all jobs found in the queue

        | Returns:
        |   jobs (dict): a dict of jobs classified by ID containing all info
        |                that can be matched through list_outre
        |
        """

        jobs = {}
        for j_id in self._job_list:
            jobs[j_id] = {'job_id': j_id,
                          'job_status': self._job_list[j_id]['status']}

        return jobs

    def kill(self, job_id):
        """Kill the job with the given ID

        | Args:
        |   job_id (str): ID of the job to kill
        |
        """

        try:
            del(self._job_list[job_id])
        except KeyError:
            pass
