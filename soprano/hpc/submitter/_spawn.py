"""
Methods used to spawn or kill multiple daemons on a cluster machine.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import shutil

# Spawn a Submitter instance from a given module
subm_module, subm_name = sys.argv[1:3]
loaded_module = __import__(subm_module, fromlist=[subm_name])
subm = getattr(loaded_module, subm_name)
try:
    subm.start()
except Exception as e:
    # If ANYTHING goes wrong, at least clean up!
    for j in subm._jobs:
        subm.queue.kill(j)
        try:
            shutil.rmtree(subm._jobs[j]['folder'])
        except OSError:
            pass # Whatever, it was deleted already I guess
    subm.log('Submitter crashed following error:'
             '\n{0}:\n\t{1}'.format(type(e).__name__, e))

