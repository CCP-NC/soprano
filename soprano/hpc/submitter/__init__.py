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

"""Classes and functions required for processes that automatically submit jobs
to a queueing system working in the background.

These can be launched interactively from the command line. In order to do
that:

1. write your own implementation of a submitter class by inheriting from
   soprano.hpc.submitter.Submitter or use one of the provided ones;
2. write an input file in which you simply create an instance of said class
   and set up its parameters (ideally by calling the set_parameters method);
3. launch that submitter from the command line with the following command:

    ``python -m soprano.hpc.submitter start <filename>``

You can have multiple submitter instances, even of different types, defined
in the same file: in that case you will need to use the -n option to specify
which one you want to launch (the name you need to use is the name of the
*variable* you stored the instance in). If you are working on remote login and
you want to prevent the submitter from being terminated upon exiting your
session use the -nohup option.
To list which submitters from a given file are running, and how long have they
been running for, just use:

    ``python -m soprano.hpc.submitter list <filename>``

Similarly, you can stop a running submitter with:

    ``python -m soprano.hpc.submitter stop <filename>``

Submitters have a 'name' property and will save a <name>.log file in which any
output from their run can be stored.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.hpc.submitter.queues import QueueInterface
from soprano.hpc.submitter.submit import Submitter
from soprano.hpc.submitter.castep import CastepSubmitter
