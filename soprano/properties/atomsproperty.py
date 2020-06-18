# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Definition of AtomsProperty class.

A generic template class that specific Properties will inherit from.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from soprano.collection import AtomsCollection


class AtomsProperty(object):

    default_name = 'generic_property'
    default_params = {}

    def __init__(self, name=None, **params):
        """Initialize an AtomsProperty and set its parameters.
        The AtomsProperty instance can then be called with a structure as its
        only argument to get the property with the given parameters.

        | Args:
        |   name (str): a name to give to this specific instance of the
        |               property (will be used to store it as array if
        |               requested)
        |   params: named arguments specific to this type of property

        """

        if name is not None:
            self.name = name
        else:
            self.name = self.default_name
        # Validate the passed parameters
        self.params = dict(self.default_params)
        for p in params:
            if p not in self.params:
                raise ValueError('Invalid argument passed to'
                                 ' \'{0}\''.format(self.__class__.__name__))
            else:
                self.params[p] = params[p]

    @classmethod
    def get(self, s, store_array=False, **kwargs):
        """Extract the given property using the default parameters
        on an Atoms object s

        | Args:
        |   s (ase.Atoms or AtomsCollection): the structure or collection
        |                                     from which to extract the
        |                                     property
        |   store_array (bool): if s is a collection, whether to store the
        |                       resulting data as an array in the collection
        |                       using the default name for this property
        |

        | Returns:
        |   property: the value of the property for the given structure or
        |             a list of values if a collection has been passed
        |

        """

        if isinstance(s, AtomsCollection):
            arr = s.all.map(self.get, **kwargs)
            if store_array:
                s.set_array(self.default_name, arr)
            return arr
        else:
            params = dict(self.default_params)
            params.update(kwargs)
            return self.extract(s, **params)

    @staticmethod
    def extract(s, **params):
        """Extract the given property with given parameters from an Atoms
        object.

        | Args:
        |   s (ase.Atoms): the structure from which to extract the property
        |   params: named arguments specific to this type of property
        |

        | Returns:
        |   property: the value of the property for the given structure and
        |             parameters
        |

        """

        # Do something specific to get the property of interest
        # Then return the value

        pass

    def __call__(self, s, store_array=False):
        """Calling the AtomsProperty returns the value of the property as
        extracted with the parameters of this specific instance.

        | Args:
        |   s (ase.Atoms or AtomsCollection): the structure or collection
        |                                     from which to extract the
        |                                     property
        |   store_array (bool): if s is a collection, whether to store the
        |                       resulting data as an array in the collection
        |                       using the given name for this instance
        |

        | Returns:
        |   property: the value of the property for the given structure or
        |             a list of values if a collection has been passed
        |

        """

        if isinstance(s, AtomsCollection):
            arr = s.all.map(self.__call__)
            if store_array:
                s.set_array(self.name, arr)
            return arr
        else:
            return self.extract(s, **self.params)
