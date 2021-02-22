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


def _make_t_om(width, h_steps):

    dt = h_steps/(2*h_steps+1.0)*2*np.pi/width
    t = np.linspace(-h_steps*dt, h_steps*dt, 2*h_steps+1)
    om = np.linspace(-width, width, 2*h_steps+1)

    return t, om


def _make_phifun(values=[1, -1], probabilities=[0.5, 0.5]):
    v = np.array(values)
    p = np.array(probabilities)
    p = np.where(p >= 0, p, 0)
    p /= np.sum(p)

    def phifun(t):
        return np.sum(p[None, :]*np.exp(-1.0j*t[:, None]*v[None, :]), axis=1)

    return phifun


def _scalar_sum_phifun(t, scalars=[([1, -1], [0.5, 0.5])]):

    funcs = [_make_phifun(*s)(t) for s in scalars]
    return np.prod(funcs, axis=0)


def _scalar_sum_distribution(scalars=[([1, -1], [0.5, 0.5])], h_steps=100):

    maxs = np.sum([np.amax(s[0]) for s in scalars])
    mins = np.sum([np.amin(s[0]) for s in scalars])

    center = (maxs + mins) / 2.0
    width = np.amax(np.abs([maxs-center, mins-center]))

    t, om = _make_t_om(width, h_steps)
    om += center

    phifun = _scalar_sum_phifun(t, scalars)*np.exp(1.0j*center*t)

    spec = np.abs(np.fft.ifft(phifun))
    spec = np.fft.fftshift(spec)

    spec /= np.trapz(spec, om)

    return om, spec

def _vector_len2_phi(scalars, axes, t):

    n = len(scalars)
    axes = np.array(axes)

    if axes.shape[0] != n:
        raise ValueError('Axes do not match scalars')
    if len(axes.shape) != 2:
        raise ValueError('Axes must be one vector for each scalar')
    d = axes.shape[1]

    # Start by building a matrix of vector dot products
    M = np.sum(axes[:, None, :]*axes[None, :, :], axis=-1)
    w, R = np.linalg.eigh(M)

    w = np.where(np.isclose(w, 0), 0, w)

    # Build the scalar combinations from the eigenvectors
    phi_sp = []
    for ev, evec in list(zip(w, R.T))[-d:]:
        phi = np.ones(len(t))+0.0j
        for j, s in enumerate(scalars):
            phi *= _make_phifun(np.array(s[0])*evec[j]*ev**0.5, s[1])(t)
        phi_sp.append(phi)
    phi_sp = np.array(phi_sp)

    adjt = np.where(t != 0, t, np.inf)

    phi_sp2 = ((-1.0j*np.pi/adjt)**0.5 *
               np.trapz(phi_sp[:, :, None] *
                        np.exp(0.25j*t[:, None]**2/adjt[None, :]
                               )[None, :, :], t, axis=1))

    return phi_sp2

def _vector_len2_distribution(scalars=[([1, -1], [0.5, 0.5])],
                              axes=[[0, 0, 1]], h_steps=100):

    # Bound: treat them as if all axes were aligned
    maxn = np.sum([np.amax(np.abs(s[0])) for s in scalars])
    width = maxn**2

    center = width
    t, om = _make_t_om(width, 2*h_steps)

    om += center

    phi_sp2 = _vector_len2_phi(scalars, axes, t)

    phi_len2 = np.prod(phi_sp2, axis=0)*np.exp(1.0j*center*t)
    P_len2 = np.real(np.fft.ifft(phi_len2)*np.exp(-1.0j*center*t))
    P_len2 = np.fft.fftshift(P_len2)

    # Normalization step necessary due to the anomaly at t = 0
    P_norm = np.average(P_len2[2*h_steps+1:])
    P_len2 -= P_norm
    P_len2 /= np.trapz(P_len2, om)

    om = om[:2*h_steps+1]
    P_len2 = P_len2[:2*h_steps+1]

    return om, P_len2

def _vector_len_distribution(scalars=[([1, -1], [0.5, 0.5])],
                             axes=[[0, 0, 1]], h_steps=100):

    # Bound: treat them as if all axes were aligned
    maxn = np.sum([np.amax(np.abs(s[0])) for s in scalars])

    width = maxn
    center = width

    t, om = _make_t_om(width, 2*h_steps)
    t /= (2*width)
    om += center
    
    phi_sp2 = _vector_len2_phi(scalars, axes, t)

    phi_len2 = np.prod(phi_sp2, axis=0)
    P_len = np.real(np.trapz(phi_len2[:,None]*np.exp(1.0j*t[:,None]*om[None,:]**2), t, axis=0))
    # Normalization step necessary due to the anomaly at t = 0
    P_norm = np.average(P_len[2*h_steps+1:])
    P_len -= P_norm
    P_len *= 2*om
    P_len /= np.trapz(P_len, om)

    om = om[:2*h_steps+1]
    P_len = P_len[:2*h_steps+1]

    return om, P_len