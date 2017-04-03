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
Utilities for remote submission, especially using Paramiko for SSH
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    import paramiko as pmk
except ImportError:
    pmk = None
import os


class RemoteTarget(object):

    """RemoteTarget object

    Uses Paramiko to embed into one single object all we need to connect
    and send orders to some remote machine. It will accept a single host name
    as an argument and requires it to be present in the ~/.ssh/config file
    with all relevant parameters. It also must have an ssh key configurated
    for access (passwords are NOT accepted as they can't be stored and passed
    safely) and must have already been added to known_hosts (in other words,
    you must already have connected to it from the shell).
    This class is meant to be used as an environment with the 'with' keyword.

    """

    def __init__(self, host, timeout=1.0):
        """Initialize the RemoteTarget

        | Args:
        |   host (str): the name of the target to connect to, as present
        |               in the ~/.ssh/config file
        |   timeout (Optional[float]): connection timeout in seconds (default
        |                              is 1 second)

        """

        # Check that Paramiko is even present
        if pmk is None:
            raise RuntimeError('Paramiko not installed - RemoteTarget can not'
                               ' be used')

        # Ok, check that the hostname exists
        config = pmk.SSHConfig()
        config.parse(open(os.path.expanduser('~/.ssh/config')))

        if host not in config.get_hostnames():
            raise ValueError('Host '
                             '{0} not found in ~/.ssh/config'.format(host))

        hostdata = config.lookup(host)
        self._connect_args = {
            'hostname': hostdata['hostname'],
            'timeout': timeout
        }

        if 'port' in hostdata:
            self._connect_args['port'] = int(hostdata['port'])
        if 'user' in hostdata:
            self._connect_args['username'] = hostdata['user']
        if 'identityfile' in hostdata:
            self._connect_args['key_filename'] = hostdata['identityfile'][-1]

        # Now create the SSHClient and parse the system's hostkeys
        self._client = pmk.SSHClient()
        self._client.load_system_host_keys()

        # We're all ready now!

    def __enter__(self):
        # Open the connection
        self._client.connect(**self._connect_args)
        return self

    def __exit__(self, type, value, traceback):
        # Make sure the connection is closed
        self._client.close()

    def run_cmd(self, cmd, cwd=None, stdin=None):
        """
        Run a command on the remote machine.

        | Args:
        |   cmd (str): command to run remotely
        |   cwd (Optional[str]): working directory in which to run the command
        |   stdin (Optional[str]): content to communicate to the command's
        |                          stdin

        | Returns:
        |   stdout, stderr (str): string representations of output and error
        |                         messages from the command

        """

        if cwd is not None:
            cmd = 'cd {0}; '.format(cwd) + cmd

        (_stdin,
         _stdout,
         _stderr) = self._client.exec_command(cmd,
                                              timeout=self._connect_args[
                                                  'timeout'])

        if stdin is not None:
            _stdin.write(stdin)
            _stdin.channel.shutdown_write()

        return _stdout.read(), _stderr.read()
