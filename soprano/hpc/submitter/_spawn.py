"""
Methods used to spawn or kill multiple daemons on a cluster machine.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

# Spawn a Submitter instance from a given module
subm_module, subm_name = sys.argv[1:3]
loaded_module = __import__(subm_module, fromlist=[subm_name])
getattr(loaded_module, subm_name).start()