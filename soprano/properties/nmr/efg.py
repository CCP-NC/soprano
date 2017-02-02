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

"""Implementation of AtomsProperties that relate to NMR electric field
gradients"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import pkgutil
import numpy as np
from soprano.properties import AtomsProperty
from soprano.properties.nmr.utils import (_haeb_sort, _anisotropy, _asymmetry,
                                          _span, _skew)


try:
    _nmr_data = pkgutil.get_data('soprano',
                                 'data/nmrdata.json').decode('utf-8')
    _nmr_data = json.loads(_nmr_data)
except IOError:
    _nmr_data = None


def _has_efg_check(f):
    # Decorator to add a check for the electric field gradient array
    def decorated_f(s, *args, **kwargs):
        if not (s.has('efg')):
            raise RuntimeError('No electric field gradient data found for'
                               ' this system')
        return f(s, *args, **kwargs)

    return decorated_f


class EFGDiagonal(AtomsProperty):

    """
    EFGDiagonal

    Produces an array containing eigenvalues and eigenvectors for the
    symmetric part of each EFG tensor in the system. By default
    saves them as part of the Atoms' info as well.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   save_info (bool): if True, save the diagonalised tensors in the
    |                     Atoms object's info. By default True.

    | Returns:
    |   efg_diag (np.ndarray): list of eigenvalues and eigenvectors

    """

    default_name = 'efg_diagonal'
    default_params = {
        'save_info': True
    }

    @staticmethod
    @_has_efg_check
    def extract(s, save_info):

        efg_diag = [np.linalg.eigh((efg+efg.T)/2.0)
                    for efg in s.get_array('efg')]
        efg_evals, efg_evecs = [np.array(a) for a in zip(*efg_diag)]

        if save_info:
            s.info[EFGDiagonal.default_name + '_evals'] = efg_evals
            # Store also the Haeberlen sorted version
            s.info[EFGDiagonal.default_name +
                   '_evals_hsort'] = _haeb_sort(efg_evals)
            s.info[EFGDiagonal.default_name + '_evecs'] = efg_evecs

        return np.array([dict(zip(('evals', 'evecs'), efg))
                         for efg in efg_diag])


class EFGVzz(AtomsProperty):

    """
    EFGVzz

    Produces an array containing the major component of the electric field
    gradient in a system (au).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   efg_list (np.ndarray): list of Vzz values
    """

    default_name = 'efg_vzz'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_efg_check
    def extract(s, force_recalc):

        if ((not EFGDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            EFGDiagonal.get(s)

        efg_evals = s.info[EFGDiagonal.default_name + '_evals_hsort']

        return efg_evals[:, -1]


class EFGAnisotropy(AtomsProperty):

    """
    EFGAnisotropy

    Produces an array containing the electric field gradient anisotropies in a
    system (au).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   efg_list (np.ndarray): list of anisotropies

    """

    default_name = 'efg_anisotropy'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_efg_check
    def extract(s, force_recalc):

        if ((not EFGDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            EFGDiagonal.get(s)

        efg_evals = s.info[EFGDiagonal.default_name + '_evals_hsort']

        return _anisotropy(efg_evals)


class EFGReducedAnisotropy(AtomsProperty):

    """
    EFGReducedAnisotropy

    Produces an array containing the electric field gradient reduced 
    anisotropies in a system (au).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   efg_list (np.ndarray): list of reduced anisotropies

    """

    default_name = 'efg_red_anisotropy'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_efg_check
    def extract(s, force_recalc):

        if ((not EFGDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            EFGDiagonal.get(s)

        efg_evals = s.info[EFGDiagonal.default_name + '_evals_hsort']

        return _anisotropy(efg_evals, reduced=True)


class EFGAsymmetry(AtomsProperty):

    """
    EFGAsymmetry

    Produces an array containing the electric field gradient asymmetries
    in a system (adimensional).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   efg_list (np.ndarray): list of asymmetries

    """

    default_name = 'efg_asymmetry'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_efg_check
    def extract(s, force_recalc):

        if ((not EFGDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            EFGDiagonal.get(s)

        efg_evals = s.info[EFGDiagonal.default_name + '_evals_hsort']

        return _asymmetry(efg_evals)


class EFGSpan(AtomsProperty):

    """
    EFGSpan

    Produces an array containing the electric field gradient tensor span
    in a system (au).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   efg_list (np.ndarray): list of spans

    """

    default_name = 'efg_span'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_efg_check
    def extract(s, force_recalc):

        if ((not EFGDiagonal.default_name + '_evals' in s.info) or
                force_recalc):
            EFGDiagonal.get(s)

        efg_evals = s.info[EFGDiagonal.default_name + '_evals']

        return _span(efg_evals)


class EFGSkew(AtomsProperty):

    """
    EFGSkew

    Produces an array containing the magnetic shielding tensor skew
    in a system.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   efg_list (np.ndarray): list of skews

    """

    default_name = 'efg_skew'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_efg_check
    def extract(s, force_recalc):

        if ((not EFGDiagonal.default_name + '_evals' in s.info) or
                force_recalc):
            EFGDiagonal.get(s)

        efg_evals = s.info[EFGDiagonal.default_name + '_evals']

        return _skew(efg_evals)
