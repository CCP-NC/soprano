"""HPC Daemons are meant to run one for each node of a cluster machine.
They actually manage the running of multi-core processes at a local level.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.hpc.daemons.daemon import (DaemonHPC, DaemonRunner,)