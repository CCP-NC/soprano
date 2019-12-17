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
optional.py

Contains imports for libraries considered optional, and decorators to include
checks for them.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import wraps

try:
    import networkx as _networkx
except ImportError:
    _networkx = None

try:
    import spglib as _spglib
except ImportError:
    try:
        from pyspglib import spglib as _spglib
    except ImportError:
        _spglib = None

try:
    import sklearn as _sklearn
except ImportError:
    _sklearn = None


"""
These decorators check if the required module is available, if not print
an error message, if yes pass it as a variable to the function itself. 
They all take the name one desires the library to have within the function
as an argument. The function needs to have a named variable of the same name
in its interface.
"""


def requireNetworkX(import_name='networkx'):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            if _networkx is None:
                raise RuntimeError('This function requires an installation of'
                                   ' NetworkX to work - please install it '
                                   'with:\n\tpip install networkx')
            else:
                kwargs[import_name] = _networkx
                return func(*args, **kwargs)

        return wrapper

    return decorator


def requireSpglib(import_name='spglib'):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            v = list(map(int, _spglib.__version__.split('.')))

            if _spglib is None:
                raise RuntimeError('This function requires an installation of'
                                   ' spglib to work - please install it '
                                   'with:\n\tpip install spglib')
            elif v[0] < 1 or (v[0] == 1 and v[1] <= 8):
                raise RuntimeError('This function requires a version of'
                                   ' spglib superior to 1.8 to work - '
                                   'please install it '
                                   'with:\n\tpip install --upgrade spglib')
            else:
                kwargs[import_name] = _spglib
                return func(*args, **kwargs)

        return wrapper

    return decorator


def requireScikitLearn(import_name='sklearn'):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            if _sklearn is None:
                raise RuntimeError('This function requires an installation of'
                                   ' scikit-learn to work - please install '
                                   'it with:\n\tpip install scikit-learn')
            else:
                kwargs[import_name] = _sklearn
                return func(*args, **kwargs)

        return wrapper

    return decorator
