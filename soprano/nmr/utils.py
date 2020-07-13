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

"""Utility functions for NMR-related properties"""

import re
import numpy as np
import scipy.constants as cnst
from ase.quaternions import Quaternion

# Left here for backwards compatibility
from soprano.data.nmr import (_get_isotope_data, _get_nmr_data, _el_iso)


def _evals_sort(evals, convention='c', return_indices=False):
    """Sort a list of eigenvalue triplets by varios conventions"""
    evals = np.array(evals)
    iso = np.average(evals, axis=1)

    if convention in ('i', 'd'):
        to_sort = evals
    elif convention in ('h', 'n'):
        to_sort = np.abs(evals-iso[:, None])

    sort_i = np.argsort(to_sort,
                        axis=1)
    if convention == 'd':
        sort_i = sort_i[:, ::-1]
    elif convention == 'h':
        sort_i[:, 0], sort_i[:, 1] = sort_i[:, 1], sort_i[:, 0].copy()
    sorted_evals = evals[np.arange(evals.shape[0])[:, None],
                         sort_i]
    if not return_indices:
        return sorted_evals
    else:
        return sorted_evals, sort_i


def _haeb_sort(evals, return_indices=False):
    return _evals_sort(evals, 'h', return_indices)


def _anisotropy(haeb_evals, reduced=False):
    """Calculate anisotropy given eigenvalues sorted with Haeberlen
    convention"""

    f = 2.0/3.0 if reduced else 1.0

    return (haeb_evals[:, 2]-(haeb_evals[:, 0]+haeb_evals[:, 1])/2.0)*f


def _asymmetry(haeb_evals):
    """Calculate asymmetry"""

    aniso = _anisotropy(haeb_evals, reduced=True)
    # Fix the anisotropy zero values
    aniso = np.where(aniso == 0, np.inf, aniso)

    return (haeb_evals[:, 1]-haeb_evals[:, 0])/aniso


def _span(evals):
    """Calculate span"""

    return np.amax(evals, axis=-1)-np.amin(evals, axis=-1)


def _skew(evals):
    """Calculate skew"""

    span = _span(evals)
    span = np.where(span == 0, np.inf, span)
    return 3*(np.median(evals,
                        axis=1) -
              np.average(evals,
                         axis=1))/span


def _evecs_2_quat(evecs):
    """Convert a set of eigenvectors to a Quaternion expressing the
    rotation of the tensor's PAS with respect to the Cartesian axes"""

    # First, guarantee that the eigenvectors express *proper* rotations
    evecs = np.array(evecs)*np.linalg.det(evecs)[:, None, None]

    # Then get the quaternions
    return [Quaternion.from_matrix(evs.T) for evs in evecs]


def _dip_constant(Rij, gi, gj):
    """Dipolar constants for pairs ij, with distances Rij and gyromagnetic
    ratios gi and gj"""

    return - (cnst.mu_0*cnst.hbar*gi*gj / (8*np.pi**2*Rij**3))


def _dip_tensor(d, r, rotation_axis=None):
    """Full dipolar tensor given a constant and a connecting vector"""

    r = np.array(r)
    r /= np.linalg.norm(r)

    if rotation_axis is None:
        D = d*(3*r[:, None]*r[None, :]-np.eye(3))/2.0
    else:
        a = np.array(rotation_axis)
        a /= np.linalg.norm(a)
        vp2 = np.dot(r, a)**2
        D = 0.5*d*(3*vp2-1)*(3*a[:, None]*a[None, :]-np.eye(3))/2.0

    return D


def _J_constant(Kij, gi, gj):
    """J coupling constants for pairs ij, with reduced constant Kij and
    gyromagnetic ratios gi and gj"""

    return cnst.h*gi*gj*Kij/(4*np.pi**2)*1e19
