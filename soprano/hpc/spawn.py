"""
Methods used to spawn or kill multiple daemons on a cluster machine.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import pickle

subm_name, classmodule, classname = sys.argv[1:4]

loaded_module = __import__(classmodule, fromlist=[classname])
globals()[classname] = getattr(loaded_module, classname)

subm_obj = pickle.load(open(subm_name))
subm_obj._start_execution() 
