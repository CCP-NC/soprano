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
Functions for creating probability distributions for large sums of scalars or
vectors.
"""

import numpy as np


def _make_chifun(values=[1, -1], probabilities=[0.5, 0.5]):
    v = np.array(values)
    p = np.array(probabilities)
    p = np.where(p >= 0, p, 0)
    p /= np.sum(p)

    def chifun(t):
        return np.sum(p[None, :]*np.exp(1.0j*t[:, None]*v[None, :]), axis=1)

    return chifun


def _scalar_sum_chifun(t, scalars=[([1, -1], [0.5, 0.5])]):

    funcs = [_make_chifun(*s)(t) for s in scalars]
    return np.prod(funcs, axis=0)


def _scalar_sum_distribution(scalars=[([1, -1], [0.5, 0.5])], width=None,
                             h_steps=100):

    if width is None:
        width = np.sum([np.amax(np.abs(s[0])) for s in scalars])

    dt = h_steps/(2*h_steps+1.0)*2*np.pi/width
    t = np.linspace(-h_steps*dt, h_steps*dt, 2*h_steps+1)

    chifun = _scalar_sum_chifun(t, scalars)
    spec = np.abs(np.fft.ifft(chifun))
    spec = np.fft.fftshift(spec)

    om = np.linspace(-width, width, 2*h_steps+1)
    spec /= np.trapz(spec, om)

    return om, spec
