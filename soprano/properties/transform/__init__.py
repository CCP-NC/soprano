"""
Module containing  a special set of AtomsProperties that transform an Atoms
object into another (by translating, rotating or mirroring all or some ions,
and so on). These all accept an Atoms object and some parameters and return
an Atoms object as well. Default behaviour for the .get method in most cases
will be to do nothing at all, these properties are meant to be instantiated.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.properties.transform.transform import (Translate, Rotate, Mirror,
                                                    Regularise)
