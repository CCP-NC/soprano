"""Convenience util used to start/stop submitter processes"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def submitter_handler():

    import os, sys
    import argparse as ap
    import subprocess as sp
    from soprano.hpc.submitter import Submitter

    # First of all, check that we are on Linux, this only works there
    if sys.platform not in ('linux', 'linux2'):
        sys.exit('Invalid OS: background Submitter launching works only on '
                 'Linux')

    class IsValidModule(ap.Action):

        def __call__(self, parser, namespace, values, option_string=''):
            # Is this actually a valid Python module name?
            splname = os.path.splitext(values)
            if splname[1] != '.py':
                raise ValueError('Argument {0} is not a valid Python file')
            else:
                setattr(namespace, self.dest, splname[0])

    parser = ap.ArgumentParser()
    # Required arguments
    parser.add_argument('action', type=str, nargs=1,
                        choices=['start', 'stop'],
                        help="Action to perform on the given Submitter -"
                             " whether to start or stop it")
    parser.add_argument('submitter_file', type=str, action=IsValidModule,
                        help="Name of the Python module file containing the"
                             " declaration for the Submitter to use")
    # Optional arguments
    parser.add_argument('-n', type=str, default=None,
                        help="Name of the Submitter instance to use, if "
                        "more than one is present in the given file. "
                        "CAREFUL: this is the name of the variable, not the "
                        "one passed as parameter in the constructor.")

    try:
        args = parser.parse_args()
    except ValueError:
        sys.exit('Invalid submitter_file argument: file is not a valid Python'
                 ' module')

    # First, check that the required submitter file exists and that it
    # contains what we need
    try:
        loaded_module = __import__(args.submitter_file)
    except ImportError:
        sys.exit('Invalid submitter_file argument: file not found')

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
        submitter_name = subms.keys()[0]

    if args.action[0] == 'start':
        sp.Popen(['python', '-m', 'soprano.hpc.submitter._spawn',
                  args.submitter_file, submitter_name, '&'])
    else:
        # PKILL the process
        sp.Popen(['pkill', '-f', args.submitter_file])



submitter_handler()