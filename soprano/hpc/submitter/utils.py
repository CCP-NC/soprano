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
import glob
import fnmatch
from soprano.utils import is_string


class RemoteTarget(object):

    """RemoteTarget object

    Uses Paramiko to embed into one single object all we need to connect
    and send orders to some remote machine. It will accept a single host name
    as an argument and requires it to be present in the ~/.ssh/config file
    with all relevant parameters. It also must have an ssh key configurated
    for access (passwords are NOT accepted as they can't be stored and passed
    safely) and must have already been added to known_hosts (in other words,
    you must already have connected to it from the shell).

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
                               ' be initialised')

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

    @property
    def context(self):
        """Returns a context using this RemoteTarget to connect.
        This is done this way so that we only need to instantiate the Target
        once to avoid pointless overhead but still can handle the connection
        securely by making sure it's closed no matter what.
        """
        return RemoteTargetContext(self)


def _ensure_open_sftp(f):
    """Decorator for SFTP related functions in RemoteTargetContext"""

    def sftp_checked(self, *args, **kwargs):
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return f(self, *args, **kwargs)

    return sftp_checked


class RemoteTargetContext(object):

    """RemoteTargetContext object

    Works as a context to be used with the 'with' statement. Should usually
    just be created by a RemoteTarget through the appropriate property.

    """

    def __init__(self, rT):
        self._client = rT._client
        self._connect_args = rT._connect_args
        self._sftp = None

    def __enter__(self):
        # Open the connection
        self._client.connect(**self._connect_args)
        return self

    def __exit__(self, type, value, traceback):
        # Make sure the connection is closed
        if self._sftp is not None:
            self._sftp.close()
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

    @_ensure_open_sftp
    def put_files(self, localpaths, remotedir):
        """
        Copy files to the remote machine via SFTP.

        | Args:
        |   localpaths (str or [str]): path of file(s) to copy. Can include
        |                              wildcards.
        |   remotedir (str): remote directory to copy the file(s) into.

        """

        if is_string(localpaths):
            localpaths = [localpaths]

        for lpath in localpaths:
            files = glob.glob(lpath)

            for f in files:
                _, fname = os.path.split(f)
                self._sftp.put(f, os.path.join(remotedir, fname),
                               confirm=True)

    @_ensure_open_sftp
    def get_files(self, remotepaths, localdir):
        """
        Download files from the remote machine via SFTP.

        | Args:
        |   remotepaths (str or [str]): path of file(s) to copy. Can include
        |                               wildcards.
        |   localdir (str): local directory to copy the file(s) into.

        """

        if is_string(remotepaths):
            remotepaths = [remotepaths]

        for rpath in remotepaths:
            remotedir, remotefiles = os.path.split(rpath)

            all_files = self._sftp.listdir(remotedir)
            files = fnmatch.filter(all_files, remotefiles)

            for f in files:
                _, fname = os.path.split(f)
                self._sftp.get(os.path.join(remotedir, f),
                               os.path.join(localdir, fname))
