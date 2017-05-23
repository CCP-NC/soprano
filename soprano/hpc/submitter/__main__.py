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

"""Convenience util used to start/stop submitter processes"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def submitter_handler():

    import os
    import sys
    import argparse as ap
    import subprocess as sp
    from soprano.utils import import_module
    from soprano.hpc.submitter import Submitter

    # First of all, check that we are on Linux, this only works there
    if sys.platform not in ('linux', 'linux2'):
        sys.exit('Invalid OS: background Submitter launching works only on '
                 'Linux')

    parser = ap.ArgumentParser(description="Use this script to start or stop "
                               "a Submitter object as a background process.")
    # Required arguments
    parser.add_argument('action', type=str, nargs=1,
                        help="Action to perform: "
                             "start -> start the given submitter, "
                             "stop -> stop the given submitter if running, "
                             "list -> list the currently running submitters"
                             " from the given file,"
                             " or any custom action defined for your"
                             " submitter")
    parser.add_argument('submitter_file', type=str,
                        help="Name of the Python module file containing the"
                             " declaration for the Submitter to use")
    # Optional arguments
    parser.add_argument('-n', type=str, default=None,
                        help="Name of the Submitter instance to use, if "
                        "more than one is present in the given file. "
                        "CAREFUL: this is the name of the variable, not the "
                        "one passed as parameter in the constructor.")
    parser.add_argument('-nohup', action='store_true', default=False,
                        help="If True, nohup is used to make sure that the "
                        "process does not quit in case the user logs out "
                        "(for example on a login node of an HPC machine).")

    try:
        args = parser.parse_args()
    except ValueError:
        sys.exit('Invalid submitter_file argument: file is not a valid Python'
                 ' module')

    # First, check that the required submitter file exists and that it
    # contains what we need
    abs_subm_file = os.path.abspath(args.submitter_file)
    loaded_module = import_module(abs_subm_file)

    # Now load the actual Submitter!
    subms = {v: getattr(loaded_module, v) for v in dir(loaded_module)
             if v[:2] != '__' and
             issubclass(getattr(loaded_module, v).__class__, Submitter)}

    if len(subms) == 0:
        sys.exit('No Submitter instance found in '
                 '{0}'.format(args.submitter_file))
    elif len(subms) > 1 and args.n is None:
        sys.exit('Too many Submitter instances found in '
                 '{0}'.format(args.submitter_file))
    elif args.n is not None and args.n not in subms:
        sys.exit(('Submitter of name {0} '
                  'not found in {1}').format(args.n, args.submitter_file))
    # We're fine!
    if args.n is not None:
        submitter_name = args.n
    else:
        submitter_name = list(subms.keys())[0]

    if args.action[0] == 'start':
        # Ok, first let's check that nothing like that is running already
        subm_l = Submitter.list()
        for s in subm_l:
            if (os.path.abspath(s[0]) == abs_subm_file 
                and submitter_name == s[1]):
                sys.exit('The requested submitter is already'
                         ' running')
        cmd = ['python', '-m', 'soprano.hpc.submitter._spawn',
               abs_subm_file, submitter_name, '&']
        if args.nohup:
            cmd = ['nohup'] + cmd
        sp.Popen(cmd)
        print("Submitter "
              "{0} from file {1} started".format(submitter_name,
                                                 args.submitter_file))
    elif args.action[0] == 'stop':
        # PKILL the process
        succ = Submitter.stop(abs_subm_file,
                              submitter_name)
        if succ:
            print("Submitter "
                  "{0} from file {1} stopped".format(submitter_name,
                                                     args.submitter_file))
        else:
            print("The requested submitter is not running")
    elif args.action[0] == 'list':
        subm_l = Submitter.list()
        tabf = "{1: >10}\t| {2: >10}\t| {0: >10}"
        print(tabf.format('File', 'Name', 'Time'))
        for s in subm_l:
            if os.path.abspath(s[0]) != abs_subm_file:
                continue
            print(tabf.format(*s))
    elif args.action[0] in subms[submitter_name]._user_signals:
        signum, _ = subms[submitter_name]._user_signals[args.action[0]]
        subm_l = Submitter.list()
        for s in subm_l:
            if (os.path.abspath(s[0]) == abs_subm_file
                    and s[1] == submitter_name):
                os.kill(s[3], signum)
                break
    else:
        print("Unknown action {0}".format(args.action[0]))

submitter_handler()
