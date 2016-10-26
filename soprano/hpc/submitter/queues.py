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
Definition of QueueInterface class.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import subprocess as sp

from soprano.utils import is_string


class QueueInterface(object):

    """QueueInterface object

    A class meant to simplify interfacing in a basic way
    with a Queue system. Contains commands to submit to the queue, list the
    job IDs, and kill them if necessary. Will contain Regexps to parse for IDs
    and additional information as returned upon submission and listing.
    It is important that the regular expressions used employ NAMED GROUPS to
    parse the various fields. In particular, a job_id group must ALWAYS be
    present.
    The class also provides some static variables implementing standard
    interfaces for common queueing systems. These can be retrieved by using
    QueueInterface.<NAME>. The currently implemented names are the following:

    - LSF (IBM's managing system, using the command bsub)
    - GridEngine (Sun's managing system, also available in an open version,
                  using the command qsub)
    """

    def __init__(self, sub_cmd, list_cmd, kill_cmd, sub_outre, list_outre):
        """Initialize the QueueInterface.

        | Args:
        |   sub_cmd (str): command used to submit a script to the queue
        |   list_cmd (str): command used to list all queued jobs for the user
        |   kill_cmd (str): command used to kill a job given its id
        |   sub_outre (str): regular expression used to parse the output of
        |                    sub_cmd. Must contain at least a job_id named
        |                    group
        |   list_outre (str): regular expression used to parse the output of
        |                     list_cmd. Must contain at least a job_id named
        |                     group

        """

        self.sub_cmd = sub_cmd
        self.list_cmd = list_cmd
        self.kill_cmd = kill_cmd

        self.sub_outre = re.compile(sub_outre)
        if 'job_id' not in self.sub_outre.groupindex:
            raise ValueError('sub_outre does not contain job_id group')
        self.list_outre = re.compile(list_outre)
        if 'job_id' not in self.list_outre.groupindex:
            raise ValueError('list_outre does not contain job_id group')

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

        subproc = sp.Popen([self.sub_cmd], stdin=sp.PIPE,
                           stdout=sp.PIPE,
                           stderr=sp.PIPE,
                           cwd=cwd)

        stdout, stderr = subproc.communicate(script)

        # Parse out the job id!
        match = self.sub_outre.search(stdout)
        if match is None:
            raise RuntimeError('Submission of job has failed with output:\n'
                               '\tSTDOUT: {0}\n\tSTDERR: {1}'.format(stdout,
                                                                     stderr))
        else:
            return match.groupdict()['job_id']

    def list(self):
        """List all jobs found in the queue

        | Returns:
        |   jobs (dict): a dict of jobs classified by ID containing all info
        |                that can be matched through list_outre
        |
        """
        subproc = sp.Popen([self.list_cmd], stdout=sp.PIPE,
                           stderr=sp.PIPE)

        stdout, stderr = subproc.communicate()

        # Parse out everything!
        jobs = {}
        for line in stdout.split('\n'):
            match = self.list_outre.search(line)
            if match is None:
                continue
            else:
                jobdict = match.groupdict()
                jobs[jobdict['job_id']] = jobdict

        return jobs

    def kill(self, job_id):
        """Kill the job with the given ID

        | Args:
        |   job_id (str): ID of the job to kill
        |
        """

        subproc = sp.Popen([self.kill_cmd, job_id], stdout=sp.PIPE,
                           stderr=sp.PIPE)
        stdout, stderr = subproc.communicate()

    @classmethod
    def LSF(cls):
        return cls(sub_cmd='bsub',
                   list_cmd='bjobs',
                   kill_cmd='bkill',
                   sub_outre='Job \<(?P<job_id>[0-9]+)\>',
                   list_outre='(?P<job_id>[0-9]+)[^(RUN|PEND)]*'
                              '(?P<job_status>RUN|PEND)')

    @classmethod
    def GridEngine(cls):
        return cls(sub_cmd='qsub',
                   list_cmd='qstat',
                   kill_cmd='qdel',
                   sub_outre='Your job (?P<job_id>[0-9]+)',
                   list_outre='(?P<job_id>[0-9]+)\s.*'
                              '\s(?P<job_status>r|qw)\s')
