"""Classes and functions required for processes that automatically submit jobs
to a queueing system working in the background.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.hpc.submitter.queues import QueueInterface
from soprano.hpc.submitter.submit import Submitter
from soprano.hpc.submitter.castep import CastepSubmitter
