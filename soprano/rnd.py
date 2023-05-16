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
random.py

Manages randomness across the board for Soprano functions
"""

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# Define a single Random Generator to use across the board in Soprano functions


class RandomType(type):
    def __getattr__(cls, key):
        return cls._generator.__getattribute__(key)


class Random(object, metaclass=RandomType):

    _generator = RandomState(MT19937(SeedSequence(None)))

    @classmethod
    def reseed(cls, seed=None):
        cls._generator = RandomState(MT19937(SeedSequence(seed)))

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(Random.choice(range(n), r, replace=False))
    return tuple(pool[i] for i in indices)