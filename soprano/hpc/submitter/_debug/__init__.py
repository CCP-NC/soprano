"""Functions useful for debugging QueueInterface and Submitters. These
provide a 'fake' queue that executes basic jobs with artificial delays
in order to simulate an environment similar to what can be found on an HPC
machine."""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.hpc.submitter._debug.debugqueue import DebugQueueInterface
