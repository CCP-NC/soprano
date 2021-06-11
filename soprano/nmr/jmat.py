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
Contains the JMat class, used to quickly generate quantum spin operators
"""


import numpy as np


class JMat(object):

    def __init__(self, S=0.5):

        if (S % 0.5 != 0):
            raise ValueError('Invalid spin value S = {0}'.format(S))

        self._S = abs(S)
        self._m = -np.arange(-self._S, self._S+1)
        # Create operators
        self._Jz = np.diag(self._m)+.0j
        self._Jp = 2*np.diag(0.5*(np.cumsum(2*self._m)[:-1]**0.5), k=1)+.0j
        self._Jm = self.Jp.T
        self._Jx = 0.5*(self._Jp+self._Jm)
        self._Jy = 0.5j*(self._Jm-self._Jp)

    @property
    def S(self):
        return self._S
    
    @property
    def m(self):
        return self._m

    @property
    def Jx(self):
        return self._Jx

    @property
    def Jy(self):
        return self._Jy

    @property
    def Jz(self):
        return self._Jz

    @property
    def Jp(self):
        return self._Jp

    @property
    def Jm(self):
        return self._Jm

    @property
    def J2(self):
        return self._S*(self._S+1)*np.eye(len(self._m))

    @property
    def Jvec(self):
        return np.array([self.Jx, self.Jy, self.Jz])
