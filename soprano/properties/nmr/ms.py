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

"""Implementation of AtomsProperties that relate to NMR shieldings/shifts"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.properties import AtomsProperty
from soprano.nmr.utils import (_haeb_sort, _anisotropy, _asymmetry,
                               _span, _skew, _evecs_2_quat)


def _has_ms_check(f):
    # Decorator to add a check for the magnetic shieldings array
    def decorated_f(s, *args, **kwargs):
        if not (s.has('ms')):
            raise RuntimeError('No magnetic shielding data found for this'
                               ' system')
        return f(s, *args, **kwargs)

    return decorated_f


class MSDiagonal(AtomsProperty):

    """
    MSDiagonal

    Produces an array containing eigenvalues and eigenvectors for the
    symmetric part of each magnetic shielding tensor in the system. By default
    saves them as part of the Atoms' arrays as well.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   save_array (bool): if True, save the diagonalised tensors in the
    |                      Atoms object as an array. By default True.

    | Returns:
    |   ms_diag (np.ndarray): list of eigenvalues and eigenvectors

    """

    default_name = 'ms_diagonal'
    default_params = {
        'save_array': True
    }

    @staticmethod
    @_has_ms_check
    def extract(s, save_array):

        ms_diag = [np.linalg.eigh((ms+ms.T)/2.0) for ms in s.get_array('ms')]
        ms_evals, ms_evecs = [np.array(a) for a in zip(*ms_diag)]

        if save_array:
            s.set_array(MSDiagonal.default_name + '_evals', ms_evals)
            # Store also the Haeberlen sorted version
            s.set_array(MSDiagonal.default_name +
                        '_evals_hsort', _haeb_sort(ms_evals))
            s.set_array(MSDiagonal.default_name + '_evecs', ms_evecs)

        return np.array([dict(zip(('evals', 'evecs'), ms)) for ms in ms_diag])


class MSIsotropy(AtomsProperty):

    """
    MSIsotropy

    Produces an array containing the magnetic shielding isotropies in a system
    (ppm, with reference if provided).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   ref (float): reference frequency. If provided, the chemical shift
    |                will be returned instead of the magnetic shielding.

    | Returns:
    |   ms_iso (np.ndarray): list of shieldings/shifts

    """

    default_name = 'ms_isotropy'
    default_params = {
        'ref': None
    }

    @staticmethod
    @_has_ms_check
    def extract(s, ref):

        ms_iso = np.trace(s.get_array('ms'), axis1=1, axis2=2)/3.0

        # Referenced?
        if ref is not None:
            ms_iso = ref - ms_iso

        return ms_iso


class MSAnisotropy(AtomsProperty):

    """
    MSAnisotropy

    Produces an array containing the magnetic shielding anisotropies in a
    system (ppm).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of anisotropies

    """

    default_name = 'ms_anisotropy'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if (not s.has(MSDiagonal.default_name + '_evals_hsort') or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + '_evals_hsort')

        return _anisotropy(ms_evals)


class MSReducedAnisotropy(AtomsProperty):

    """
    MSReducedAnisotropy

    Produces an array containing the magnetic shielding reduced anisotropies
    in a system (ppm).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of reduced anisotropies

    """

    default_name = 'ms_red_anisotropy'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if (not s.has(MSDiagonal.default_name + '_evals_hsort') or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + '_evals_hsort')

        return _anisotropy(ms_evals, reduced=True)


class MSAsymmetry(AtomsProperty):

    """
    MSAsymmetry

    Produces an array containing the magnetic shielding asymmetries
    in a system (adimensional).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of asymmetries

    """

    default_name = 'ms_asymmetry'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if (not s.has(MSDiagonal.default_name + '_evals_hsort') or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + '_evals_hsort')

        return _asymmetry(ms_evals)


class MSSpan(AtomsProperty):

    """
    MSSpan

    Produces an array containing the magnetic shielding tensor span
    in a system (ppm).
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of spans

    """

    default_name = 'ms_span'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if (not s.has(MSDiagonal.default_name + '_evals_hsort') or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + '_evals_hsort')

        return _span(ms_evals)


class MSSkew(AtomsProperty):

    """
    MSSkew

    Produces an array containing the magnetic shielding tensor skew
    in a system.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_list (np.ndarray): list of skews

    """

    default_name = 'ms_skew'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if (not s.has(MSDiagonal.default_name + '_evals_hsort') or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.get_array(MSDiagonal.default_name + '_evals_hsort')

        return _skew(ms_evals)


class MSQuaternion(AtomsProperty):

    """
    MSQuaternion

    Produces a list of ase.Quaternion objects expressing the orientation of
    the MS tensors with respect to the cartesian axes.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present.

    | Returns:
    |   ms_quat (np.ndarray): list of quaternions

    """

    default_name = 'ms_quats'
    default_params = {
        'force_recalc': False
    }

    @staticmethod
    @_has_ms_check
    def extract(s, force_recalc):

        if (not s.has(MSDiagonal.default_name + '_evecs') or
                force_recalc):
            MSDiagonal.get(s)

        ms_evecs = s.get_array(MSDiagonal.default_name + '_evecs')

        return _evecs_2_quat(ms_evecs)
