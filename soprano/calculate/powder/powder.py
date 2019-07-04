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
powder.py

Generic powder averaging scheme class for inheritance
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

class PowderScheme(object):

	def __init__(self, mode):

		if mode not in ('sphere', 'hemisphere', 'octant'):
			raise ValueError('Invalid mode passed to powder averaging scheme')

		self.mode = mode

	def get_orient_angles(self, N):
		pass

	def get_orient_trig(self, N):
		pass

	def get_orient_points(self, N):
		pass