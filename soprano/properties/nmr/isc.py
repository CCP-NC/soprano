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

"""Implementation of AtomsProperties that relate to NMR J
couplings"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection
from soprano.nmr.utils import (_haeb_sort, _anisotropy, _asymmetry,
                               _span, _skew, _evecs_2_quat,
                               _J_constant)
from soprano.data.nmr import _get_nmr_data, _get_isotope_data


def _has_isc_check(f):
    # Decorator to add a check for the J coupling array
    def decorated_f(s, *args, **kwargs):
        tag = kwargs.get('tag', 'isc')
        if not (s.has(tag)):
            raise RuntimeError('No J coupling data for tag {0}'.format(tag) +
                               ' found in this system')
        return f(s, *args, **kwargs)

    return decorated_f


class ISCDiagonal(AtomsProperty):

    """
    ISCDiagonal

    Produces an array containing eigenvalues and eigenvectors for the
    symmetric part of each J coupling tensor in the system. By default
    saves them as part of the Atoms' arrays as well.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   save_info (bool): if True, save the diagonalised tensors in the
    |                     Atoms object as info. By default True.

    | Returns:
    |   isc_diag (dict): dictionary of eigenvalues and eigenvectors by atom
    |                    index pairs (lower index first)

    """

    default_name = 'isc_diagonal'
    default_params = {
        'tag': 'isc',
        'save_info': True
    }

    @staticmethod
    @_has_isc_check
    def extract(s, tag, save_info):

        isc_dict = {(i, j): np.array(t) for j, r in enumerate(s.get_array(tag))
                    for i, t in enumerate(r) if t is not None}

        isc_diag = {ij: dict(zip(['evals', 'evecs'],
                                 np.linalg.eigh((t+t.T)/2.0)))
                    for ij, t in isc_dict.items()}
        isc_pairs = sorted(isc_diag.keys())
        isc_evals = np.array([isc_diag[ij]['evals'] for ij in isc_pairs])
        isc_evecs = np.array([isc_diag[ij]['evecs'] for ij in isc_pairs])

        if save_info:
            s.info[ISCDiagonal.default_name + '_' +
                   tag + '_pairs'] = isc_pairs
            s.info[ISCDiagonal.default_name + '_' +
                   tag + '_evals'] = isc_evals
            s.info[ISCDiagonal.default_name + '_' +
                   tag + '_evals_hsort'] = _haeb_sort(isc_evals)
            s.info[ISCDiagonal.default_name + '_' +
                   tag + '_evecs'] = isc_evecs

        return isc_diag


class JCDiagonal(AtomsProperty):

    """
    JCDiagonal

    Produces a dictionary of diagonalised J coupling tensors for atom pairs
    in the system. The J coupling for a pair of nuclei i and j is defined as:

    .. math::

        J_{ij} = 10^19\\frac{h\\gamma_i\\gamma_j}{4\\pi^2}K

    where the gammas represent the gyromagnetic ratios of the nuclei and K is
    the J coupling reduced tensor found in a .magres file, in 10^19.T^2.J^-1.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair.

    """

    default_name = 'jc_diagonal'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        # Compute the diagonalised eigenvectors if necessary
        iname_pairs = ISCDiagonal.default_name + '_' + tag + '_pairs'
        iname_evals = ISCDiagonal.default_name + '_' + tag + '_evals'
        iname_evecs = ISCDiagonal.default_name + '_' + tag + '_evecs'
        if iname_pairs not in s.info or force_recalc:
            iprop = ISCDiagonal(tag=tag)
            iprop(s)
        all_pairs = list(s.info[iname_pairs])
        all_evals = s.info[iname_evals]
        all_evecs = s.info[iname_evecs]

        # Selections
        if sel_i is None:
            sel_i = AtomSelection.all(s)
        elif not isinstance(sel_i, AtomSelection):
            sel_i = AtomSelection(s, sel_i)

        if sel_j is None:
            sel_j = sel_i
        elif not isinstance(sel_j, AtomSelection):
            sel_j = AtomSelection(s, sel_j)

        # Find gammas
        elems = s.get_chemical_symbols()
        _nmr_data = _get_nmr_data()

        gammas = _get_isotope_data(elems, 'gamma', isotopes, isotope_list)

        sel_pairs = [(i, j) for i in sel_i.indices
                     for j in sel_j.indices]
        if not self_coupling:
            sel_pairs = [p for p in sel_pairs if p[0] != p[1]]

        jc_dict = {}
        for sp in sel_pairs:
            try:
                i = all_pairs.index(sp)
            except ValueError:
                continue
            evals = _J_constant(all_evals[i], gammas[sp[0]], gammas[sp[1]])
            evecs = all_evecs[i]
            jc_dict[sp] = {'evals': evals, 'evecs': evecs}

        return jc_dict


class JCIsotropy(AtomsProperty):

    """
    JCIsotropy

    Produces a dictionary of J coupling isotropies for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_isotropy'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]['evals'] for ij in jc_keys]
        jc_iso = np.average(jc_evals, axis=1)

        return dict(zip(jc_dict.keys(), jc_iso))


class JCAnisotropy(AtomsProperty):

    """
    JCAnisotropy

    Produces a dictionary of J coupling anisotropies for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_anisotropy'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]['evals'] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_aniso = _anisotropy(jc_evals)

        return dict(zip(jc_dict.keys(), jc_aniso))


class JCReducedAnisotropy(AtomsProperty):

    """
    JCReducedAnisotropy

    Produces a dictionary of J coupling reduced anisotropies for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_red_anisotropy'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]['evals'] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_aniso = _anisotropy(jc_evals, reduced=True)

        return dict(zip(jc_dict.keys(), jc_aniso))


class JCAsymmetry(AtomsProperty):

    """
    JCAsymmetry

    Produces a dictionary of J coupling asymmetries for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_asymmetry'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]['evals'] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_asymm = _asymmetry(jc_evals)

        return dict(zip(jc_dict.keys(), jc_asymm))


class JCSpan(AtomsProperty):

    """
    JCSpan

    Produces a dictionary of J coupling spans for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_span'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]['evals'] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_span = _span(jc_evals)

        return dict(zip(jc_dict.keys(), jc_span))


class JCSkew(AtomsProperty):

    """
    JCSkew

    Produces a dictionary of J coupling skews for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_skew'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]['evals'] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_skew = _asymmetry(jc_evals)

        return dict(zip(jc_dict.keys(), jc_skew))


class JCQuaternion(AtomsProperty):

    """
    JCQuaternion

    Produces a dictionary of J ase.Quaternion objects expressing the
    orientation of the J coupling tensors with respect to the cartesian axes
    for atom pairs in the system.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    | Parameters:
    |   sel_i (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling. By default is None
    |                                   (= all of them).
    |   sel_j (AtomSelection or [int]): Selection or list of indices of atoms
    |                                   for which to return the J
    |                                   coupling with the ones in sel_i. By
    |                                   default is None (= same as sel_i).
    |   tag (str): name of the J coupling component to return. Magres files
    |              usually contain isc, isc_spin, isc_fc, isc_orbital_p and
    |              isc_orbital_d. Default is isc.
    |   isotopes (dict): dictionary of specific isotopes to use, by element
    |                    symbol. If the isotope doesn't exist an error will
    |                    be raised.
    |   isotope_list (list): list of isotopes, atom-by-atom. To be used if
    |                        different atoms of the same element are supposed
    |                        to be of different isotopes. Where a 'None' is
    |                        present will fall back on the previous
    |                        definitions. Where an isotope is present it
    |                        overrides everything else.
    |   self_coupling (bool): if True, include coupling of a nucleus with
    |                         itself. Otherwise excluded. Default is False.
    |   force_recalc (bool): if True, always diagonalise the tensors even if
    |                        already present. Default is False.

    | Returns: 
    |   dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = 'jc_skew'
    default_params = {
        'sel_i': None,
        'sel_j': None,
        'tag': 'isc',
        'isotopes': {},
        'isotope_list': None,
        'self_coupling': False,
        'force_recalc': False,
    }

    @staticmethod
    @_has_isc_check
    def extract(s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling,
                force_recalc):

        jDiagProp = JCDiagonal(sel_i=sel_i, sel_j=sel_j, isotopes=isotopes,
                               tag=tag, isotope_list=isotope_list,
                               self_coupling=self_coupling,
                               force_recalc=force_recalc)
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evecs = [jc_dict[ij]['evecs'] for ij in jc_keys]
        jc_quat = _evecs_2_quat(jc_evecs)

        return dict(zip(jc_dict.keys(), jc_quat))
