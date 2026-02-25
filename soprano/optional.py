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


import warnings
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
    import moyopy as _moyopy
except ImportError:
    _moyopy = None

try:
    import sklearn as _sklearn
except ImportError:
    _sklearn = None


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------


def has_spglib() -> bool:
    """Return ``True`` if spglib is importable."""
    return _spglib is not None


def has_moyopy() -> bool:
    """Return ``True`` if moyopy is importable."""
    return _moyopy is not None


def has_symmetry_backend() -> bool:
    """Return ``True`` if at least one symmetry backend is available."""
    return has_spglib() or has_moyopy()


"""
These decorators check if the required module is available, if not print
an error message, if yes pass it as a variable to the function itself.
They all take the name one desires the library to have within the function
as an argument. The function needs to have a named variable of the same name
in its interface.
"""


def requireNetworkX(import_name="networkx"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            if _networkx is None:
                raise RuntimeError(
                    "This function requires an installation of"
                    " NetworkX to work - please install it "
                    "with:\n\tpip install networkx"
                )
            else:
                kwargs[import_name] = _networkx
                return func(*args, **kwargs)

        return wrapper

    return decorator


def requireSpglib(import_name="spglib"):
    """Decorator that injects the spglib module into a function.

    .. deprecated::
        Use :func:`soprano.properties.symmetry.backend.get_symmetry_dataset`
        with ``backend="spglib"`` (or ``"auto"``) instead.  This decorator
        will be removed in a future release.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"@requireSpglib is deprecated.  Use "
                f"soprano.properties.symmetry.backend.get_symmetry_dataset() "
                f"with the desired backend instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if _spglib is None:
                raise RuntimeError(
                    "This function requires an installation of"
                    " spglib to work - please install it "
                    "with:\n\tpip install spglib"
                )
            v = list(map(int, _spglib.__version__.split(".")))
            if v[0] < 1 or (v[0] == 1 and v[1] <= 8):
                raise RuntimeError(
                    "This function requires a version of"
                    " spglib superior to 1.8 to work - "
                    "please install it "
                    "with:\n\tpip install --upgrade spglib"
                )
            kwargs[import_name] = _spglib
            return func(*args, **kwargs)

        return wrapper

    return decorator


def requireScikitLearn(import_name="sklearn"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            if _sklearn is None:
                raise RuntimeError(
                    "This function requires an installation of"
                    " scikit-learn to work - please install "
                    "it with:\n\tpip install scikit-learn"
                )
            else:
                kwargs[import_name] = _sklearn
                return func(*args, **kwargs)

        return wrapper

    return decorator
