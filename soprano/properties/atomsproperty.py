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


from soprano.collection import AtomsCollection
import numpy as np

from soprano.nmr.tensor import NMRTensor, contains_nmr_tensors


class AtomsProperty:

    default_name = "generic_property"
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
                raise ValueError(
                    "Invalid argument passed to"
                    f" '{self.__class__.__name__}'"
                )
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

    def mean(self, s, axis=None, weights=None, **kwargs):
        """
        Compute the mean of the property over a list of structures.

        The default behaviours are:
        - For a list of scalars, compute the mean along the specified axis.
        - For a list of dictionaries, compute the mean for each key across all dictionaries.
        - For a list of NMRTensor objects, compute the mean using the NMRTensor.mean method.
        - For a list of arrays, convert to numpy array and then compute the mean along the specified axis.


        | Args:
        |   s (list of ase.Atoms or an AtomsCollection): the structure or collection
        |                                     from which to extract the
        |                                     property
        |   axis (int or None): axis along which the means are computed. If None, compute the mean of scalars.
        |   weights (array-like, optional): An array of weights associated with the values.
        |                                   If this is specified, the weighted average will be computed.
        |                                   Must have the same shape as the property values.
        |   **kwargs: Additional arguments passed to the property's get method.

        | Returns:
        |   mean_property: the mean value of the property for the given structures
        |
        | Raises:
        |   ValueError: If s is not a collection/list, if property values are None,
        |               or if there's an incompatible shape for computing the mean.
        |   TypeError: If the property values are of a type that cannot be averaged.
        """
        if not (isinstance(s, AtomsCollection) or isinstance(s, list)):
            raise ValueError(
                "Mean can only be computed on a collection/list of structures"
            )
        
        property_values = self.get(s, **kwargs)

        if not property_values:
            raise ValueError(
                "No property values found for the given structures"
            )
        
        # Apply weights if provided
        if weights is not None:
            # Ensure weights match the length of property values
            if len(weights) != len(property_values):
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match "
                    f"length of property values ({len(property_values)})"
                )
        
        # Dict mean
        if isinstance(property_values[0], dict):
            if weights is not None:
                raise NotImplementedError(
                    "Weighted mean for dictionaries is not implemented"
                )
            if axis is not None:
                raise NotImplementedError(
                    "Mean along an axis for dictionaries is not implemented"
                )
            mean_property = {}
            for key in property_values[0].keys():
                # Check if all dictionaries have the same keys
                if not all(key in prop for prop in property_values):
                    raise ValueError(
                        f"Key '{key}' not present in all property dictionaries"
                    )
                
                values = [d.get(key) for d in property_values]
                try:
                    mean_values = np.mean(values, axis=axis)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot compute mean for key '{key}': {str(e)}"
                    )
                mean_property[key] = mean_values
            return mean_property
        else: # List or array mean
            
            # Special handling for lists of NMRTensor objects (including nested)
            if contains_nmr_tensors(property_values):
                # Pass weights to NMRTensor.mean if supported
                return NMRTensor.mean(property_values, weights=weights, axis=axis)

            try:
                if weights is not None:
                    return np.average(property_values, axis=axis, weights=weights)
                return np.mean(property_values, axis=axis)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot compute mean of property values: {str(e)}")
    

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
