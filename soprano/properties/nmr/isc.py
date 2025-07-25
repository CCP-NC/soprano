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


import numpy as np

from soprano.data.nmr import _get_isotope_data, _get_isotope_list
from soprano.nmr.tensor import NMRTensor, TensorConvention
from soprano.nmr.utils import (
    _anisotropy,
    _asymmetry,
    _evecs_2_quat,
    _haeb_sort,
    _J_constant,
    _skew,
    _span,
)
from soprano.nmr.coupling import JCoupling
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection

from typing import Literal, Optional, Union

JC_TENSOR_ORDER = TensorConvention.Haeberlen

def _has_isc_check(f):
    # Decorator to add a check for the J coupling array
    def decorated_f(s, *args, **kwargs):
        tag = kwargs.get("tag", "isc")
        if not (s.has(tag)):
            raise RuntimeError(
                f"No J coupling data for tag {tag}" + " found in this system"
            )
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

    Parameters:
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      save_info (bool): if True, save the diagonalised tensors in the
                        Atoms object as info. By default True.

    Returns:
      isc_diag (dict): dictionary of eigenvalues and eigenvectors by atom
                       index pairs (lower index first)

    """

    default_name = "isc_diagonal"
    default_params = {"tag": "isc", "save_info": True}

    @staticmethod
    @_has_isc_check
    def extract(s, tag, save_info):

        isc_dict = {
            (i, j): np.array(t)
            for j, r in enumerate(s.get_array(tag))
            for i, t in enumerate(r)
            if t is not None
        }

        isc_diag = {
            ij: dict(zip(["evals", "evecs"], np.linalg.eigh((t + t.T) / 2.0)))
            for ij, t in isc_dict.items()
        }
        isc_pairs = sorted(isc_diag.keys())
        isc_evals = np.array([isc_diag[ij]["evals"] for ij in isc_pairs])
        isc_evecs = np.array([isc_diag[ij]["evecs"] for ij in isc_pairs])

        if save_info:
            s.info[ISCDiagonal.default_name + "_" + tag + "_pairs"] = isc_pairs
            s.info[ISCDiagonal.default_name + "_" + tag + "_evals"] = isc_evals
            s.info[ISCDiagonal.default_name + "_" + tag + "_evals_hsort"] = _haeb_sort(
                isc_evals
            )
            s.info[ISCDiagonal.default_name + "_" + tag + "_evecs"] = isc_evecs

        return isc_diag


class JCDiagonal(AtomsProperty):

    """
    JCDiagonal

    Produces a dictionary of diagonalised J coupling tensors for atom pairs
    in the system. The J coupling for a pair of nuclei i and j is defined as:

    .. math::

        J_{ij} = 10^{19}\\frac{h\\gamma_i\\gamma_j}{4\\pi^2}K

    where the gammas represent the gyromagnetic ratios of the nuclei and K is
    the J coupling reduced tensor found in a .magres file, in :math:`10^{19} T^2 J^{-1}`.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair.

    """

    default_name = "jc_diagonal"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ) -> dict[tuple[int, int], dict[str, Union[np.ndarray, list]]]:

        # Compute the diagonalised eigenvectors if necessary
        iname_pairs = ISCDiagonal.default_name + "_" + tag + "_pairs"
        iname_evals = ISCDiagonal.default_name + "_" + tag + "_evals"
        iname_evecs = ISCDiagonal.default_name + "_" + tag + "_evecs"
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

        gammas = _get_isotope_data(elems, "gamma", isotopes, isotope_list)

        sel_pairs = [(i, j) for i in sel_i.indices for j in sel_j.indices]
        if not self_coupling:
            sel_pairs = [p for p in sel_pairs if p[0] != p[1]]

        jc_dict = {}
        for sp in sel_pairs:
            # sort sp if necessary
            sp = tuple(sorted(sp))
            try:
                i = all_pairs.index(sp)
            except ValueError:
                continue
            evals = _J_constant(all_evals[i], gammas[sp[0]], gammas[sp[1]])
            evecs = all_evecs[i]
            jc_dict[sp] = {"evals": evals, "evecs": evecs}

        return jc_dict

class KTensor(AtomsProperty):
    """
    KTensor

    Produces a dictionary of reduced J-coupling NMRTensor objects (i.e. KTensor).

    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
    
    Returns:
      ktensor_dict (dict[tuple[int, int], NMRTensor]): Dictionary of J coupling tensors by atomic index pair,
                      in Hz. These are NMRTensor objects.
    
    Raises:
        ValueError: If the tag is not one of the supported values.
    """

    default_name = "k_tensor"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "self_coupling": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag: str, self_coupling: bool
    ) -> dict[tuple[int, int], NMRTensor]:

        isc_dict: dict[tuple[int, int], np.ndarray] = {
            (i, j): np.array(t)
            for j, r in enumerate(s.get_array(tag))
            for i, t in enumerate(r)
            if t is not None
        }

        # Selections
        if sel_i is None:
            sel_i = AtomSelection.all(s)
        elif not isinstance(sel_i, AtomSelection):
            sel_i = AtomSelection(s, sel_i)

        if sel_j is None:
            sel_j = sel_i
        elif not isinstance(sel_j, AtomSelection):
            sel_j = AtomSelection(s, sel_j)

        sel_pairs = [(i, j) for i in sel_i.indices for j in sel_j.indices]
        if not self_coupling:
            sel_pairs = [p for p in sel_pairs if p[0] != p[1]]

        # Sort sel_pairs?
        sel_pairs = [tuple(sorted(sp)) for sp in sel_pairs if sp in isc_dict]
        ktensor_dict: dict[tuple[int, int], NMRTensor] = {}
        for sp in sel_pairs:
            ktensor_dict[sp] = NMRTensor(isc_dict[sp], order=JC_TENSOR_ORDER)
        return ktensor_dict


class JCTensor(AtomsProperty):
    """
    JCTensor

    Produces a dictionary of J-coupling NMRTensor objects (i.e. JCTensor).

    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
    
    Returns:
      jc_dict (dict[tuple[int, int], NMRTensor]): Dictionary of J coupling tensors by atomic index pair,
                      in Hz. These are NMRTensor objects.
    """

    default_name = "jc_tensor"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s,
        sel_i: Union[AtomSelection, list[int]],
        sel_j: Union[AtomSelection, list[int]],
        tag: str,
        isotopes: dict,
        isotope_list: Optional[list],
        self_coupling: bool
    ) -> dict[tuple[int, int], NMRTensor]:

        kTensorProp = KTensor(
            sel_i=sel_i,
            sel_j=sel_j,
            tag=tag,
            self_coupling=self_coupling,
        )
        ktensor_dict = kTensorProp(s)
        
        # Get gamma values
        gamma_array = _get_isotope_data(
            s.get_chemical_symbols(),
            "gamma",
            isotopes,
            isotope_list,
        )
        jtensor_dict: dict[tuple[int, int], NMRTensor] = {}
        for key, ktensor in ktensor_dict.items():
            # Gamma values for the two nuclei
            gi, gj = gamma_array[key[0]], gamma_array[key[1]]
            # Scale the K tensor to J coupling tensor in Hz
            j_data = _J_constant(ktensor.data, gi, gj)
            # Create the NMRTensor object with the J coupling data
            jtensor_dict[key] = NMRTensor(j_data, order=JC_TENSOR_ORDER)

        return jtensor_dict
        
class JCIsotropy(AtomsProperty):

    """
    JCIsotropy

    Produces a dictionary of J coupling isotropies for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = "jc_isotropy"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]["evals"] for ij in jc_keys]
        jc_iso = np.average(jc_evals, axis=1)

        return dict(zip(jc_dict.keys(), jc_iso))


class JCAnisotropy(AtomsProperty):

    """
    JCAnisotropy

    Produces a dictionary of J coupling anisotropies for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = "jc_anisotropy"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]["evals"] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_aniso = _anisotropy(jc_evals)

        return dict(zip(jc_dict.keys(), jc_aniso))


class JCReducedAnisotropy(AtomsProperty):

    """
    JCReducedAnisotropy

    Produces a dictionary of J coupling reduced anisotropies for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = "jc_red_anisotropy"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]["evals"] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_aniso = _anisotropy(jc_evals, reduced=True)

        return dict(zip(jc_dict.keys(), jc_aniso))


class JCAsymmetry(AtomsProperty):

    """
    JCAsymmetry

    Produces a dictionary of J coupling asymmetries for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = "jc_asymmetry"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]["evals"] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_asymm = _asymmetry(jc_evals)

        return dict(zip(jc_dict.keys(), jc_asymm))


class JCSpan(AtomsProperty):

    """
    JCSpan

    Produces a dictionary of J coupling spans for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = "jc_span"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]["evals"] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_span = _span(jc_evals)

        return dict(zip(jc_dict.keys(), jc_span))


class JCSkew(AtomsProperty):

    """
    JCSkew

    Produces a dictionary of J coupling skews for atom pairs
    in the system. See JCDiagonal for how reduced couplings are transformed
    into couplings.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      dip_dict (dict): Dictionary of couplings by atomic index pair, in Hz.

    """

    default_name = "jc_skew"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evals = [jc_dict[ij]["evals"] for ij in jc_keys]
        jc_evals = _haeb_sort(jc_evals)
        jc_skew = _skew(jc_evals)

        return dict(zip(jc_dict.keys(), jc_skew))

class JCEuler(AtomsProperty):
    """
    JCEuler

    Produces a dictionary of J coupling Euler angles for atom pairs
    in the system. See JCTensor for how the J coupling tensors are
    instantiated.


    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      convention (str): 'zyz' or 'zxz' accepted - the ordering of the Euler
                        angle rotation axes. Default is zyz 
      passive (bool):  active or passive rotations. Default is active (passive=False)
      degrees (bool):  return the angles in degrees. Default is False

    Returns:
      euler_dict (dict): Dictionary of coupling Euler angles by atomic index pair, in Hz.

    """

    default_name = "jc_euler"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "convention": "zyz",
        "passive": False,
        "degrees": False
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s,
        sel_i: Union[AtomSelection, list[int]],
        sel_j: Union[AtomSelection, list[int]],
        tag: str,
        isotopes: dict,
        isotope_list: list,
        self_coupling: bool,
        convention: Literal["zyz", "zxz"],
        passive: bool,
        degrees: bool
    ):

        jc_tensors = JCTensor(
            sel_i=sel_i,
            sel_j=sel_j,
            tag=tag,
            isotopes=isotopes,
            isotope_list=isotope_list,
            self_coupling=self_coupling
        ).get(s)
        euler_dict = {}
        for ij, jc_tensor in jc_tensors.items():
            euler_angles = jc_tensor.euler(
                convention=convention,
                passive=passive,
                degrees=degrees
            )
            euler_dict[ij] = euler_angles
        return euler_dict

class JCQuaternion(AtomsProperty):

    """
    JCQuaternion

    Produces a dictionary of J ase.Quaternion objects expressing the
    orientation of the J coupling tensors with respect to the cartesian axes
    for atom pairs in the system.
    Requires the Atoms object to have been loaded from a .magres file
    containing the relevant information.

    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.
      force_recalc (bool): if True, always diagonalise the tensors even if
                           already present. Default is False.

    Returns:
      quat_dict (dict): Dictionary of coupling quaternions by atomic index pair, in Hz.

    """

    default_name = "jc_quats"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
        "force_recalc": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s, sel_i, sel_j, tag, isotopes, isotope_list, self_coupling, force_recalc
    ):

        jDiagProp = JCDiagonal(
            sel_i=sel_i,
            sel_j=sel_j,
            isotopes=isotopes,
            tag=tag,
            isotope_list=isotope_list,
            self_coupling=self_coupling,
            force_recalc=force_recalc,
        )
        jc_dict = jDiagProp(s)
        jc_keys = jc_dict.keys()
        jc_evecs = [jc_dict[ij]["evecs"] for ij in jc_keys]
        jc_quat = _evecs_2_quat(jc_evecs)

        return dict(zip(jc_dict.keys(), jc_quat))

class JCouplingList(AtomsProperty):
    """
    JCouplingList

    Produces a list of JCoupling objects for all pairs of atoms
    in the system. 
    
    Note that the self.tensor attribute of the JCoupling
    objects is a **reduced** J coupling tensor, i.e. the K tensor
    from the .magres file, in :math:`10^{19} T^2 J^{-1}`.

    Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to return the J
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      tag (str): name of the J coupling component to return. Magres files
                 usually contain isc, isc_spin, isc_fc, isc_orbital_p and
                 isc_orbital_d. Default is isc.
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised.
      isotope_list (list): list of isotopes, atom-by-atom. To be used if
                           different atoms of the same element are supposed
                           to be of different isotopes. Where a 'None' is
                           present will fall back on the previous
                           definitions. Where an isotope is present it
                           overrides everything else.
      self_coupling (bool): if True, include coupling of a nucleus with
                            itself. Otherwise excluded. Default is False.

    Returns:
      jc_list (list[JCoupling]): List of J coupling tensors in Hz.

    Examples:
        # Get all J couplings
        jc_list = JCouplingList.get(atoms)

    """

    default_name = "jc_list"
    default_params = {
        "sel_i": None,
        "sel_j": None,
        "tag": "isc",
        "isotopes": {},
        "isotope_list": None,
        "self_coupling": False,
    }

    @staticmethod
    @_has_isc_check
    def extract(
        s,
        sel_i: Union[AtomSelection, list[int], None],
        sel_j: Union[AtomSelection, list[int], None],
        tag: str,
        isotopes: dict[str, str],
        isotope_list: Optional[list[str]],
        self_coupling: bool
    ) -> list[JCoupling]:

        # Get all Tensors
        k_tensors = KTensor.get(s, sel_i=sel_i, sel_j=sel_j, tag=tag, self_coupling=self_coupling)
        if not k_tensors:
            return []
        
        elems = s.get_chemical_symbols()
        isotopes_numbers = _get_isotope_list(elems, isotopes, isotope_list)

        # Sort the keys which are tuples of ints
        sorted_pairs = sorted(k_tensors.keys(), key=lambda x: (x[0], x[1]))
        
        jc_list: list[JCoupling] = []
        for ij in sorted_pairs:
            site_i, site_j = ij
            species1 = f"{isotopes_numbers[site_i]}{elems[site_i]}"
            species2 = f"{isotopes_numbers[site_j]}{elems[site_j]}"
            jc_list.append(
                JCoupling(
                    site_i=site_i,
                    site_j=site_j,
                    species1=species1,
                    species2=species2,
                    tensor=k_tensors[ij],
                    type="J",
                    tag=tag,
                )
            )

        return jc_list