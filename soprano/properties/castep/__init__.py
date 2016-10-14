"""
Module containing AtomsProperties related specifically to CASTEP calculations.
Some of these can be looked up only in a CASTEP Calculator; others require
passing the path of the .castep file as a parameter and actually parsing its
contents.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.properties.castep.castep import (CastepEnthalpy,)
