"""
Module containing AtomProperties that pertain to symmetry detection.
Depends on having the Python bindings to SPGLIB installed on the system.
"""

try:
	try:
		import spglib
	except ImportError:
		from pyspglib import spglib
except ImportError:
	raise ImportError('SPGLIB was not found on this system -'
					  ' symmetry properties can not be used')

# Python 2-to-3 compatibility code
from __future__ import absolute_import
