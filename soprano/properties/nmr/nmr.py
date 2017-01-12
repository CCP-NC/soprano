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

"""Implementation of AtomsProperties that relate to NMR parameters"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.properties import AtomsProperty


class MSDiagonal(AtomsProperty):

    """
    MSIsotropy

    Produces an array containing eigenvalues and eigenvectors for the
    symmetric part of each magnetic shielding tensor in the system. By default
    saves them as two new arrays as well.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   save_info (bool): if True, save the diagonalised tensors in the
    |                     Atoms object's info. By default True.

    | Returns:
    |   ms_diag (np.ndarray): list of eigenvalues and eigenvectors

    """

    default_name = 'ms_diagonal'
    default_params = {
        'save_info': True
    }

    @staticmethod
    def extract(s, save_info):

        # Check if the array even exists
        if not (s.has('ms')):
            raise RuntimeError('No magnetic shielding data found for this'
                               ' system')

        ms_diag = [np.linalg.eigh((ms+ms.T)/2.0) for ms in s.get_array('ms')]
        ms_evals, ms_evecs = [np.array(a) for a in zip(*ms_diag)]

        if save_info:
            s.info[MSDiagonal.default_name + '_evals'] = ms_evals
            # Store also the Haeberlen sorted version
            ms_iso = np.average(ms_evals, axis=1)
            sort_i = np.argsort(np.abs(ms_evals-ms_iso[:, None]),
                                axis=1)[:, [1, 0, 2]]
            ms_evals_haeb = ms_evals[np.arange(ms_evals.shape[0])[:, None],
                                     sort_i]
            s.info[MSDiagonal.default_name + '_evals_hsort'] = ms_evals_haeb
            s.info[MSDiagonal.default_name + '_evecs'] = ms_evecs

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
    def extract(s, ref):

        # Check if the array even exists
        if not (s.has('ms')):
            raise RuntimeError('No magnetic shielding data found for this'
                               ' system')

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
    def extract(s, force_recalc):

        # Check if the array even exists
        if not (s.has('ms')):
            raise RuntimeError('No magnetic shielding data found for this'
                               ' system')

        if ((not MSDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.info[MSDiagonal.default_name + '_evals_hsort']
        ms_aniso = ms_evals[:, 2]-(ms_evals[:, 0]+ms_evals[:, 1])/2.0

        return ms_aniso


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
    def extract(s, force_recalc):

        # Check if the array even exists
        if not (s.has('ms')):
            raise RuntimeError('No magnetic shielding data found for this'
                               ' system')

        if ((not MSDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            MSDiagonal.get(s)

        return MSAnisotropy.get(s)*2.0/3.0


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
    def extract(s, force_recalc):

        # Check if the array even exists
        if not (s.has('ms')):
            raise RuntimeError('No magnetic shielding data found for this'
                               ' system')

        if ((not MSDiagonal.default_name + '_evals_hsort' in s.info) or
                force_recalc):
            MSDiagonal.get(s)

        ms_evals = s.info[MSDiagonal.default_name + '_evals_hsort']
        ms_red_aniso = MSReducedAnisotropy.get(s)

        return (ms_evals[:,1]-ms_evals[:,0])/ms_red_aniso
