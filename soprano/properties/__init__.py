"""
Contains classes, modules and functions relevant to Properties,
a catch-all term for things we might want to extract or calculate from
Atoms and AtomsCollections. Some will require running an external ASE
calculator first, some will just work on their own, some will require
some calculations and parameters.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.properties.atomsproperty import AtomsProperty