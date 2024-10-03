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
Classes and functions for simulating approximated NMR spectroscopic results
from structures.
"""


import logging
import re
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from ase import Atoms
from matplotlib.axes import Axes
from scipy.special import fresnel

from soprano.calculate.nmr.utils import (
    Peak2D,
    calculate_distances,
    extract_indices,
    filter_pairs_by_distance,
    generate_contour_map,
    generate_peaks,
    get_atom_labels,
    get_pair_dipolar_couplings,
    get_pair_j_couplings,
    merge_peaks,
    nmr_2D_style,
    nmr_base_style,
    prepare_species_labels,
    process_pairs,
    sort_peaks,
    styled_plot,
    validate_elements,
)
from soprano.calculate.powder.triavg import TriAvg
from soprano.data.nmr import EFG_TO_CHI, _el_iso, _get_isotope_data, _get_isotope_list, _get_nmr_data
from soprano.nmr.utils import _dip_constant
from soprano.properties.nmr import MSIsotropy
from soprano.selection import AtomSelection
from soprano.utils import minimum_supcell, supcell_gridgen

DEFAULT_MARKER_SIZE = 50 # Default marker size for 2D plots (max is 100)
ANNOTATION_LINE_WIDTH = 0.15 # Line width for annotation lines
ANNOTATION_FONT_SCALE = 0.5 # Scale factor for annotation font size wrt to plot font size
DEFAULT_MAX_NUM_LEGEND_ELEMENTS = 6 # Default maximum number of elements to show in the legend in 2D plots

_nmr_data = _get_nmr_data()

# Conversion functions to Tesla
# (they take element and isotope as arguments)
_larm_units = {
    "MHz": lambda e, i: 2 * np.pi * 1.0e6 / _nmr_data[e][i]["gamma"],
    "T": lambda e, i: 1.0,
}

# Function used for second-order quadrupolar shift
# Arguments: cos(2*alpha), eta


def _st_A(c2a, eta):
    return -27.0 / 8.0 - 9.0 / 4.0 * eta * c2a - 3.0 / 8.0 * eta ** 2 * c2a ** 2


def _st_B(c2a, eta):
    return 15.0 / 4.0 - 0.5 * eta ** 2 + 2 * eta * c2a + 0.75 * eta ** 2 * c2a ** 2


def _st_C(c2a, eta):
    return (
        -23.0 / 40.0
        + 14.0 / 15.0 * eta ** 2
        + 0.25 * eta * c2a
        - 3.0 / 8.0 * eta ** 2 * c2a ** 2
    )


def _mas_A(c2a, eta):
    return 21.0 / 16.0 - 7.0 / 8.0 * eta * c2a + 7.0 / 48.0 * eta ** 2 * c2a ** 2


def _mas_B(c2a, eta):
    return (
        -9.0 / 8.0
        + 1.0 / 12.0 * eta ** 2
        + eta * c2a
        - 7.0 / 24.0 * eta ** 2 * c2a ** 2
    )


def _mas_C(c2a, eta):
    return (
        9.0 / 80.0
        - 1.0 / 15.0 * eta ** 2
        - 0.125 * eta * c2a
        + 7.0 / 48.0 * eta ** 2 * c2a ** 2
    )


def _gfunc(ca, cb, eta, A, B, C):
    c2a = 2 * ca ** 2 - 1
    return A(c2a, eta) * cb ** 4 + B(c2a, eta) * cb ** 2 + C(c2a, eta)


# Flags for what to include in spectra
NMRFlags = namedtuple(
    "NMRFlags",
    """CS_ISO
                      CS_ORIENT
                      CS
                      Q_1_ORIENT
                      Q_2_SHIFT
                      Q_2_ORIENT_STATIC
                      Q_2_ORIENT_MAS
                      Q_2_STATIC
                      Q_2_MAS
                      Q_STATIC
                      Q_MAS
                      STATIC
                      MAS""",
)
NMRFlags = NMRFlags(
    CS_ISO=1,
    CS_ORIENT=2,
    CS=1 + 2,
    Q_1_ORIENT=4,
    Q_2_SHIFT=8,
    Q_2_ORIENT_STATIC=16,
    Q_2_ORIENT_MAS=32,
    Q_2_STATIC=8 + 16,
    Q_2_MAS=8 + 32,
    Q_STATIC=4 + 8 + 16,
    Q_MAS=8 + 32,
    STATIC=1 + 2 + 4 + 8 + 16,
    MAS=1 + 8 + 32,
)


MARKER_INFO = {
    'distance': {
        'label': 'Distance',
        'unit': 'Å',
        'fmt': '{x:.1f}'
    },
    'inversedistance': {
        'label': '1/Distance',
        'unit': r'Å$^{{-1}}$',
        'fmt': '{x:.3f}'
    },
    'dipolar': {
        'label': 'Dipolar Coupling',
        'unit': 'kHz',
        'fmt': '{x:.1f}'
    },
    'jcoupling': {
        'label': 'J Coupling',
        'unit': 'Hz',
        'fmt': '{x:.1f}'
    },
    'fixed': {
        'label': 'Fixed',
        'unit': '',
        'fmt': '{x:.1f}'
    },
    'custom': {
        'label': 'Correlation strength',
        'unit': '',
        'fmt': '{x:.1f}'
    }
    }

class NMRCalculator:

    """NMRCalculator

    An object providing an interface to produce basic simulated NMR spectra
    from .magres files. It should be kept in mind that this is *not* a proper
    spin simulation tool, but merely provides a 'guide for the eye' kind of
    spectrum to compare to experimental results. What it can simulate:

    - chemical shift of NMR peaks
    - quadrupolar shifts of NMR peaks up to second order corrections
    - effects of crystal orientation (single crystal)
    - powder average (policrystalline/powder)
    - ultrafast MAS limit spectra

    What it can NOT simulate:

    - finite speed MAS spectra
    - J couplings
    - dipolar interactions
    - complex NMR experiments

    A list of the currently available NMRFlags to be used in conjunction with
    methods that require a list of effects of interest: ::

        NMRFlags.CS_ISO     => chemical shielding, isotropic effect
                .CS_ORIENT  => chemical shielding, orientation dependent
                               effects
                .CS         => chemical shielding, everything
                .Q_1_ORIENT => quadrupolar, 1st order, orientation dependent
                               effects
                .Q_2_SHIFT  => quadrupolar, 2nd order, isotropic shift
                .Q_2_ORIENT_STATIC => quadrupolar, 2nd order, orientation
                                      dependent effects; static limit
                .Q_2_ORIENT_MAS => quadrupolar, 2nd order, orientation
                                      dependent effects; ultrafast MAS limit
                .Q_2_STATIC => quadrupolar, 2nd order, all static effects
                .Q_2_MAS    => quadrupolar, 2nd order, all ultrafast MAS
                               effects
                .Q_STATIC   => quadrupolar, all static effects
                .Q_MAS      => quadrupolar, all ultrafast MAS effects
                .STATIC     => all static effects
                .MAS        => all ultrafast MAS effects


    | Args:
    |   sample (ase.Atoms): an Atoms object describing the system to simulate
    |                       on. Should be loaded with ASE from a .magres file
    |                       if data on shieldings and EFGs is necessary. It
    |                       can also have an optional 'isotopes' array. If it
    |                       does, it will be used in the set_isotopes method
    |                       and interpreted as described in its documentation.
    |   larmor_frequency (float): larmor frequency of the virtual spectrometer
    |                             (referenced to Hydrogen). Default is 400.
    |   larmor_units (str): units in which the larmor frequency is expressed.
    |                       Default are MHz.

    """

    def __init__(self, sample, larmor_frequency=400, larmor_units="MHz"):

        if not isinstance(sample, Atoms):
            raise TypeError("sample must be an ase.Atoms object")

        self._sample = sample

        # Define isotope array
        self._elems = np.array(self._sample.get_chemical_symbols())
        if self._sample.has("isotopes"):
            isos = self._sample.get_array("isotopes")
        else:
            isos = [None] * len(self._sample)
        self.set_isotopes(isos)

        self.set_larmor_frequency(larmor_frequency, larmor_units)

        self._references = {}

    def set_larmor_frequency(
        self, larmor_frequency=400, larmor_units="MHz", element="1H"
    ):
        """
        Set the Larmor frequency of the virtual spectrometer with the desired
        units and reference element.

        | Args:
        |   larmor_frequency (float): larmor frequency of the virtual
        |                             spectrometer. Default is 400.
        |   larmor_units (str): units in which the larmor frequency is
        |                       expressed. Can be MHz or T. Default are MHz.
        |   element (str): element and isotope to reference the frequency to.
        |                  Should be in the form <isotope><element>. Isotope
        |                  is optional, if absent the most abundant NMR active
        |                  one will be used. Default is 1H.

        """

        if larmor_units not in _larm_units:
            raise ValueError("Invalid units for Larmor frequency")

        # Split isotope and element
        el, iso = _el_iso(element)

        self._B = larmor_frequency * _larm_units[larmor_units](el, iso)

    def get_larmor_frequency(self, element):
        """
        Get the Larmor frequency of the virtual spectrometer for the desired
        element in MHz.

        | Args:
        |   element (str): element and isotope whose frequency we require.
        |                  Should be in the form <isotope><element>. Isotope
        |                  is optional, if absent the most abundant NMR active
        |                  one will be used. Default is 1H.

        | Returns:
        |   larmor (float): Larmor frequency in MHz

        """

        el, iso = _el_iso(element)
        return self._B / _larm_units["MHz"](el, iso)

    def set_reference(self, ref, element):
        """
        Set the chemical shift reference (in ppm) for a given element. If not
        provided it will be assumed to be zero.

        | Args:
        |   ref (float): reference shielding value in ppm. Chemical shift will
        |                be calculated as this minus the atom's ms.
        |   element (str): element and isotope whose reference is set.
        |                  Should be in the form <isotope><element>. Isotope
        |                  is optional, if absent the most abundant NMR active
        |                  one will be used.

        """

        el, iso = _el_iso(element)

        if el not in self._references:
            self._references[el] = {}
        self._references[el][iso] = float(ref)

    def set_isotopes(self, isotopes):
        """
        Set the isotopes for each atom in sample.

        | Args:
        |   isotopes (list): list of isotopes for each atom in sample.
        |                    Isotopes can be given as an array of integers or
        |                    of symbols in the form <isotope><element>.
        |                    Their order must match the one of the atoms in
        |                    the original sample ase.Atoms object.
        |                    If an element of the list is None, the most
        |                    common NMR-active isotope is used. If an element
        |                    is the string 'Q', the most common quadrupolar
        |                    active isotope for that nucleus (if known) will
        |                    be used.

        """

        # First: correct length?
        if len(isotopes) != len(self._sample):
            raise ValueError(
                "isotopes array should be as long as the atoms" " in sample"
            )

        # Clean up the list, make sure it's all right
        iso_clean = []
        for i, iso in enumerate(isotopes):
            # Is it an integer?
            iso_name = ""
            if re.match("[0-9]+", str(iso)) is not None:  # numpy-proof test
                iso_name = str(iso)
                # Does it exist?
                if iso_name not in _nmr_data[self._elems[i]]:
                    raise ValueError(
                        "Invalid isotope "
                        f"{iso_name} for element {self._elems[i]}"
                    )
            elif iso is None:
                iso_name = str(_nmr_data[self._elems[i]]["iso"])
            elif iso == "Q":
                iso_name = str(_nmr_data[self._elems[i]]["Q_iso"])
            else:
                el, iso_name = _el_iso(iso)
                # Additional test
                if el != self._elems[i]:
                    raise ValueError(
                        "Invalid element in isotope array - "
                        f"{el} in place of {self._elems[i]}"
                    )

            iso_clean.append(iso_name)

        self._isos = np.array(iso_clean)

    def set_element_isotope(self, element, isotope):
        """
        Set the isotope for all occurrences of a given element.

        | Args:
        |   element (str): chemical symbol of the element for which to set the
        |                  isotope.
        |   isotope (int or str): isotope to set for the given element. The
        |                         same conventions as described for the array
        |                         passed to set_isotopes apply.

        """

        new_isos = [int(x) for x in self._isos]
        new_isos = np.where(self._elems == element, isotope, new_isos)

        self.set_isotopes(new_isos)

    def set_single_crystal(self, theta, phi):
        """
        Set the orientation of the sample as a single crystallite.

        | Args:
        |   theta (float): zenithal angle for the crystallite
        |   phi (float): azimuthal angle for the crystallite

        """

        p = np.array(
            [[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]]
        )
        w = np.array([1.0])
        t = np.array([])

        self._orients = [p, w, t]

    def set_powder(self, N=8, mode="hemisphere"):
        """
        Set the orientation of the sample as a powder average.

        | Args:
        |   N (int): the number of subdivisions used to generate orientations
        |            within the POWDER algorithm. Higher values make for
        |            better but more expensive averages.
        |   mode (str): which part of the solid angle to cover with the
        |               orientations. Can be 'octant', 'hemisphere' or
        |               'sphere'. The latter should not be necessary for any
        |               NMR interaction. Default is 'hemisphere'.

        """

        self._pwdscheme = TriAvg(mode)
        self._orients = self._pwdscheme.get_orient_points(N)

    def spectrum_1d(
        self,
        element,
        min_freq=-50,
        max_freq=50,
        bins=100,
        freq_broad=None,
        freq_units="ppm",
        effects=NMRFlags.CS_ISO,
        use_central=False,
        use_reference: Optional[bool]=None,
    ):
        """
        Return a simulated spectrum for the given sample and element.

        | Args:
        |   element (str): element and isotope to get the spectrum of.
        |                  Should be in the form <isotope><element>. Isotope
        |                  is optional, if absent the most abundant NMR active
        |                  one will be used.
        |   min_freq (float): lower bound of the frequency range
        |                    (default is -50)
        |   min_freq (float): upper bound of the frequency range
        |                    (default is 50)
        |   bins (int): number of bins in which to separate the frequency range
        |              (default is 500)
        |   freq_broad (float): Gaussian broadening width to apply to the
        |                       final spectrum (default is None)
        |   freq_units (str): units used for frequency, can be ppm or MHz
        |                     (default is ppm).
        |   effects (NMRFlags): a flag, or bitwise-joined set of flags, from
        |                       this module's NMRFlags tuple, describing which
        |                       effects should be included and accounted for
        |                       in the calculation. For a list of available
        |                       flags check the docstring for NMRCalculator
        |                       (default is NMRFlags.CS_ISO).
        |   use_central (bool): if True, for half-integer spin nuclei, only
        |                       show the central transition. Ignored for
        |                       integer spin nuclei (default is False).
        |   use_reference (bool): if True, return frequencies as referenced to
        |                         the appropriate nucleus, in chemical shift
        |                         form. If no reference has been provided for
        |                         this nucleus, a value of 0 ppm is used and
        |                         the frequencies are simply flipped in sign
        |                         (default is None, in which case use_reference is True if
        |                          references have been set).

        | Returns:
        |   spec (np.ndarray): array of length 'bins' containing the spectral
        |                      intensities
        |   freq (np.ndarray): array of length 'bins' containing the frequency
        |                      axis

        """

        # First, define the frequency range
        el, iso = _el_iso(element)
        larm = self._B * _nmr_data[el][iso]["gamma"] / (2.0 * np.pi * 1e6)
        I = _nmr_data[el][iso]["I"]
        # Units? We want this to be in ppm
        u = {
            "ppm": 1,
            "MHz": 1e6 / larm,
        }
        try:
            freq_axis = np.linspace(min_freq, max_freq, bins) * u[freq_units]
        except KeyError:
            raise ValueError("Invalid freq_units passed to spectrum_1d")

        # If it's not a quadrupolar nucleus, no reason to keep those effects
        # around...
        if abs(I) < 1:
            effects &= ~NMRFlags.Q_STATIC
            effects &= ~NMRFlags.Q_MAS

        # Ok, so get the relevant atoms and their properties
        a_inds = np.where((self._elems == el) & (self._isos == iso))[0]

        # Are there even any such atoms?
        if len(a_inds) == 0:
            raise RuntimeError(
                "No atoms of the desired isotopes found in the" " system"
            )

        # Sanity check
        if effects & NMRFlags.Q_2_ORIENT_STATIC and effects & NMRFlags.Q_2_ORIENT_MAS:
            # Makes no sense...
            raise ValueError(
                "The flags Q_2_ORIENT_STATIC and Q_2_ORIENT_MAS"
                " can not be set at the same time"
            )

        if effects & NMRFlags.CS:
            try:
                ms_tens = self._sample.get_array("ms")[a_inds]
                ms_tens = (ms_tens + np.swapaxes(ms_tens, 1, 2)) / 2.0
            except KeyError:
                raise RuntimeError(
                    "Impossible to compute chemical shift - "
                    "sample has no shielding data"
                )
            ms_evals, ms_evecs = zip(*[np.linalg.eigh(t) for t in ms_tens])

        if effects & (NMRFlags.Q_STATIC | NMRFlags.Q_MAS):
            try:
                efg_tens = self._sample.get_array("efg")[a_inds]
            except KeyError:
                raise RuntimeError(
                    "Impossible to compute quadrupolar effects"
                    " - sample has no EFG data"
                )
            efg_evals, efg_evecs = zip(*[np.linalg.eigh(t) for t in efg_tens])
            efg_i = (
                np.arange(len(efg_evals))[:, None],
                np.argsort(np.abs(efg_evals), axis=1),
            )
            efg_evals = np.array(efg_evals)[efg_i]
            efg_evecs = np.array(efg_evecs)[efg_i[0], :, efg_i[1]]
            Vzz = efg_evals[:, -1]
            eta_q = (efg_evals[:, 0] - efg_evals[:, 1]) / Vzz
            Q = _nmr_data[el][iso]["Q"]
            chi = Vzz * Q * EFG_TO_CHI

        # Reference (zero if not given)
        try:
            ref = self._references[el][iso]
        except KeyError:
            ref = 0.0

        # Default to using reference if it's not zero
        if use_reference is None:
            use_reference = ref != 0.0

        # Let's start with peak positions - quantities non dependent on
        # orientation

        # Shape: atoms*1Q transitions
        peaks = np.zeros((len(a_inds), int(2 * I)))

        # Magnetic quantum number values
        if I % 1 == 0.5 and use_central:
            m = np.array([-0.5, 0.5])[None, :]
        else:
            m = np.arange(-I, I + 1).astype(float)[None, :]

        if effects & NMRFlags.CS_ISO:
            peaks += np.average(ms_evals, axis=1)[:, None]

        # Quadrupole second order
        if effects & NMRFlags.Q_2_SHIFT:
            nu_l = larm * 1e6
            # NOTE: the last factor of two in this formula was inserted
            # despite not being present in M. J. Duer (5.9) as apparently
            # it's a mistake in the book. Other sources (like the quadrupolar
            # NMR online book by D. Freude and J. Haase, Dec. 2016) report
            # this formulation, with the factor of two, instead.
            q_shifts = np.diff(
                (chi[:, None] / (4 * I * (2 * I - 1))) ** 2
                * m
                / nu_l
                * (-0.2 * (I * (I + 1) - 3 * m ** 2) * (3 + eta_q[:, None] ** 2))
                * 2
            )
            q_shifts /= larm

            peaks += q_shifts

        # Any orientational effects at all?
        has_orient = effects & (
            NMRFlags.CS_ORIENT
            | NMRFlags.Q_1_ORIENT
            | NMRFlags.Q_2_ORIENT_STATIC
            | NMRFlags.Q_2_ORIENT_MAS
        )
        # Are we using a POWDER average?
        use_pwd = len(self._orients[2]) > 0

        if has_orient:
            # Further expand the peaks!
            peaks = np.repeat(peaks[:, :, None], len(self._orients[0]), axis=-1)

        # Now compute the orientational quantities
        if effects & NMRFlags.CS_ORIENT:

            # Compute the traceless ms tensors
            ms_traceless = ms_tens - [
                np.identity(3) * np.average(ev) for ev in ms_evals
            ]
            # Now get the shift contributions for each orientation
            dirs = self._orients[0]

            peaks += np.sum(
                dirs.T[None, :, :] * np.tensordot(ms_traceless, dirs, axes=((2), (1))),
                axis=1,
            )[:, None, :]

        if effects & NMRFlags.Q_1_ORIENT:

            # First order quadrupolar anisotropic effects
            # We consider the field aligned along Z
            cosb2 = self._orients[0][:, 2] ** 2
            sinb2 = 1.0 - cosb2
            cosa2 = (self._orients[0][:, 0] ** 2) / np.where(sinb2 > 0, sinb2, np.inf)

            dir_fac = 0.5 * (
                (3 * cosb2[None, :] - 1)
                + eta_q[:, None] * sinb2[None, :] * (2 * cosa2[None, :] - 1.0)
            )
            m_fac = m[:, :-1] + 0.5
            nu_q = chi * 1.5 / (I * (2 * I - 1.0))

            qfreqs = nu_q[:, None, None] * m_fac[:, :, None] * dir_fac[:, None, :]

            peaks += qfreqs / larm  # Already ppm being Hz/MHz

        if effects & (NMRFlags.Q_2_ORIENT_STATIC | NMRFlags.Q_2_ORIENT_MAS):
            # Which one?
            if effects & NMRFlags.Q_2_ORIENT_STATIC:
                ABC = [_st_A, _st_B, _st_C]
            else:
                ABC = [_mas_A, _mas_B, _mas_C]

            cosa = self._orients[0][:, 0]
            cosb = self._orients[0][:, 1]

            dir_fac = _gfunc(cosa[None, :], cosb[None, :], eta_q[:, None], *ABC)

            m_fac = (
                I * (I + 1.0) - 17.0 / 3.0 * m[:, :-1] * (m[:, :-1] + 1) - 13.0 / 6.0
            )
            nu_q = chi * 1.5 / (I * (2 * I - 1.0))

            qfreqs = -(
                (nu_q ** 2 / (6.0 * larm * 1e6))[:, None, None]
                * m_fac[:, :, None]
                * dir_fac[:, None, :]
            )

            peaks += qfreqs / larm

        # Finally, the overall spectrum
        spec = np.zeros(freq_axis.shape)

        for p_nuc in peaks:
            for p_trans in p_nuc:
                if has_orient and use_pwd:
                    spec += self._pwdscheme.average(
                        freq_axis, p_trans, self._orients[1], self._orients[2]
                    )

        if freq_broad is None and (not has_orient or not use_pwd):
            print(
                "WARNING: no artificial broadening detected in a calculation"
                " without line-broadening contributions. The spectrum could "
                "appear distorted or empty"
            )

        if freq_broad is not None:
            if has_orient and use_pwd:
                fc = (max_freq + min_freq) / 2.0
                bk = np.exp(-(((freq_axis - fc) / freq_broad) ** 2.0))
                bk /= np.sum(bk)
                spec = np.convolve(spec, bk, mode="same")
            else:
                bpeaks = np.exp(
                    -(((freq_axis - peaks[:, :, None]) / freq_broad) ** 2)
                )  # Broadened peaks
                # Normalise them BY PEAK MAXIMUM
                norm_max = np.amax(bpeaks, axis=-1, keepdims=True)
                norm_max = np.where(np.isclose(norm_max, 0), np.inf, norm_max)
                bpeaks /= norm_max
                spec = np.sum(bpeaks, axis=(0, 1) if not has_orient else (0, 1, 2))

        # Normalize the spectrum to the number of nuclei
        normsum = np.sum(spec)
        if np.isclose(normsum, 0):
            print(
                "WARNING: no peaks found in the given frequency range. "
                "The spectrum will be empty"
            )
        else:
            spec *= len(a_inds) * len(spec) / normsum

        if use_reference:
            freq_axis = ref - freq_axis

        freqs = freq_axis / u[freq_units]
        return spec, freqs

    def dq_buildup(
        self,
        sel_i,
        sel_j=None,
        t_max=1e-3,
        t_steps=1000,
        R_cut=3,
        kdq=0.155,
        A=1,
        tau=np.inf,
    ):
        """
        Return a dictionary of double quantum buildup curves for given pairs
        of atoms, built according to the theory given in:

        G. Pileio et al., "Analytical theory of gamma-encoded double-quantum
        recoupling sequences in solid-state nuclear magnetic resonance"
        Journal of Magnetic Resonance 186 (2007) 65-74

        | Args:
        |   sel_i (AtomSelection or [int]): Selection or list of indices of
        |                                   atoms for which to compute the
        |                                   curves. By default is None
        |                                   (= all of them).
        |   sel_i (AtomSelection or [int]): Selection or list of indices of
        |                                   atoms for which to compute the
        |                                   curves with sel_i. By default is
        |                                   None (= same as sel_i).
        |   t_max (float): maximum DQ buildup time, in seconds. Default
        |                  is 1e-3.
        |   t_steps (int): number of DQ buildup time steps. Default is 1000.
        |   R_cut (float): cutoff radius for which periodic copies to consider
        |                  in each pair, in Angstrom. Default is 3.
        |   kdq (float): same as the k constant in eq. 35 of the reference. A
        |                parameter depending on the specific sequence used.
        |                Default is 0.155.
        |   A (float): overall scaling factor for the curve. Default is 1.
        |   tau (float): exponential decay factor for the curve. Default
        |                is np.inf.

        | Returns:
        |   curves (dict): a dictionary of all buildup curves indexed by pair,
        |                  plus the time axis in seconds as member 't'.
        """

        tdq = np.linspace(0, t_max, t_steps)

        # Selections
        if sel_i is None:
            sel_i = AtomSelection.all(self._sample)
        elif not isinstance(sel_i, AtomSelection):
            sel_i = AtomSelection(self._sample, sel_i)

        if sel_j is None:
            sel_j = sel_i
        elif not isinstance(sel_j, AtomSelection):
            sel_j = AtomSelection(self._sample, sel_j)

        # Find gammas
        elems = self._sample.get_chemical_symbols()
        gammas = _get_isotope_data(elems, "gamma", {}, self._isos)

        # Need to sort them and remove any duplicates, also take i < j as
        # convention
        pairs = [
            tuple(sorted((i, j)))
            for i in sorted(sel_i.indices)
            for j in sorted(sel_j.indices)
        ]

        scell_shape = minimum_supcell(R_cut, latt_cart=self._sample.get_cell())
        nfg, ng = supcell_gridgen(self._sample.get_cell(), scell_shape)

        pos = self._sample.get_positions()

        curves = {"t": tdq}

        for ij in pairs:

            r = pos[ij[1]] - pos[ij[0]]

            all_r = r[None, :] + ng

            all_R = np.linalg.norm(all_r, axis=1)

            # Apply cutoff
            all_R = all_R[np.where((all_R <= R_cut) * (all_R > 0))]

            bij = _dip_constant(all_R * 1e-10, gammas[ij[0]], gammas[ij[1]])

            th = 1.5 * kdq * abs(bij[:, None]) * tdq[None, :] * 2 * np.pi
            x = (2 * th / np.pi) ** 0.5

            Fs, Fc = fresnel(x * 2 ** 0.5)

            x[:, 0] = np.inf

            bdup = 0.5 - (1.0 / (x * 8 ** 0.5)) * (
                Fc * np.cos(2 * th) + Fs * np.sin(2 * th)
            )
            bdup[:, 0] = 0

            curves[ij] = A * np.sum(bdup, axis=0) * np.exp(-tdq / tau)

        return curves

    @property
    def B(self):
        """Static magnetic field, in Tesla"""
        return self._B


class NMRData2D:
    '''
    Class to hold and extract 2D NMR data.
    '''
    def __init__(self,
                atoms: Optional[Atoms] = None,
                xelement: Optional[str] = None,
                yelement: Optional[str] = None,
                references: Optional[dict[str, float]] = None,
                gradients: Optional[dict[str, float]] = None,
                peaks: Optional[List[Peak2D]] = None,
                pairs: Optional[List[Tuple[int, int]]] = None,
                correlation_strengths: Optional[List[float]] = None,
                correlation_strength_metric: Optional[str] = None, # 'fixed','distance', 'dipolar', 'jcoupling', 'inversedistance', 'custom'
                rcut: Optional[float] = None,
                isotopes: Optional[dict[str, int]] = None,
                is_shift: Optional[bool] = None,
                include_quadrupolar: bool = False,
                yaxis_order: str = '1Q',
                x_axis_label: Optional[str] = None,
                y_axis_label: Optional[str] = None,
                ):

        self.atoms = atoms
        self.xelement = xelement
        # if yelement is not provided, set it to xelement
        self.yelement = yelement if yelement is not None else xelement
        self.references = references
        self.gradients = gradients
        self.peaks = peaks
        self.pairs = pairs

        # if neither atoms nor peaks are provided, raise an error
        if self.atoms is None and self.peaks is None:
            raise ValueError("Either atoms or peaks must be given.")


        # Either provide correlation strengths or calculate them based on the metric
        # If both are provided, the provided values will be used
        if correlation_strengths:
            if correlation_strength_metric:
                self.logger.warning("Both correlation_strengths and correlation_strength_metric are provided. Using correlation_strengths.")
            correlation_strength_metric = 'custom'


        # If neither are provided, set the metric to 'fixed'
        if correlation_strengths is None and correlation_strength_metric is None:
            correlation_strength_metric = 'fixed'

        self.correlation_strengths = correlation_strengths
        self.correlation_strength_metric = correlation_strength_metric

        self.rcut = rcut
        self.isotopes = isotopes if isotopes is not None else {}
        # is_shift is a boolean.  If undefined, it will be set to True if references are provided, False otherwise
        #  If defined, it will be used as is
        self.is_shift = is_shift if is_shift is not None else (self.references is not None)
        self.include_quadrupolar = include_quadrupolar
        self.yaxis_order = yaxis_order
        self.logger = logging.getLogger(__name__)

        self.correlation_unit = MARKER_INFO[self.correlation_strength_metric]['unit']
        self.correlation_label = MARKER_INFO[self.correlation_strength_metric]['label']
        self.correlation_fmt = MARKER_INFO[self.correlation_strength_metric]['fmt']

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

        # run the main method to extract the data
        self.get_peaks()

    def get_peaks(self, merge_identical=True, should_sort_peaks=False):
        '''
        Get the correlation peaks.

        If self.peaks already exists, then we return them as is.
        If they don't exist, we make sure the required data is available 
        and then generate the peaks and merge if desired.
        '''

        if self.peaks is not None:
            self.logger.debug("Custom peaks provided. ")
            return self.peaks

        if self.atoms is None:
            raise ValueError("Either atoms or peaks must be given.")

        # make sure all the data is there
        self.extract_data()
        labels = get_atom_labels(self.atoms, self.logger)

        self.peaks = generate_peaks(self.data, self.pairs, labels, self.correlation_strengths, self.yaxis_order, self.xelement, self.yelement)

        if merge_identical:
            self.peaks = merge_peaks(self.peaks)

        if should_sort_peaks:
            self.peaks = sort_peaks(self.peaks)

        return self.peaks

    def extract_data(self):
        validate_elements(self.atoms, self.xelement, self.yelement)
        self.idx_x, self.idx_y = extract_indices(self.atoms, self.xelement, self.yelement)
        isotopes = _get_isotope_list(self.atoms.get_chemical_symbols(), isotopes=self.isotopes, use_q_isotopes=False)
        self.xisotope = isotopes[self.idx_x[0]]
        self.yisotope = isotopes[self.idx_y[0]]
        self.xspecies = prepare_species_labels(self.xisotope, self.xelement)
        self.yspecies = prepare_species_labels(self.yisotope, self.yelement)

        self.get_axis_labels()
        self.get_pairs()
        if self.correlation_strength_metric is None:
            raise ValueError("correlation_strength_metric is not defined. Please provide a value for this parameter.")


        self.correlation_strengths = self.get_correlation_strengths()

        self.data = MSIsotropy.get(self.atoms, ref=self.references, grad=self.gradients)
        self.logger.debug(f'Indices of xelement in the atoms object: {self.idx_x}')
        self.logger.debug(f'Indices of yelement in the atoms object: {self.idx_y}')
        self.logger.debug(f'X species: {self.xspecies}')
        self.logger.debug(f'Y species: {self.yspecies}')
        self.logger.debug(f'X values: {self.data[self.idx_x]}')
        self.logger.debug(f'Y values: {self.data[self.idx_y]}')


    def get_correlation_strengths(self):
        if self.pairs is None:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        if self.correlation_strength_metric == 'custom':
            self.logger.info("Using custom correlation strengths.")

            # make sure correlation_strengths is a list or an array
            if not isinstance(self.correlation_strengths, (list, np.ndarray)):
                raise TypeError("correlation_strengths must be a list or an array.")

            # if user provides a list of these, use it!
            # just check that it's the right length
            if len(self.correlation_strengths) != len(self.pairs_el_idx):
                raise ValueError(f"Length of correlation_strengths ({len(self.correlation_strengths)}) does not match the number of pairs ({len(self.pairs_el_idx)}).")
            correlation_strengths = self.correlation_strengths

        elif self.correlation_strength_metric == 'fixed':
            self.logger.info("Using fixed correlation strength.")
            # get all unique pairs of x and y indices
            # set the correlation strength to be the same for all pairs
            correlation_strengths = np.ones(len(self.pairs_el_idx))

        elif self.correlation_strength_metric == 'dipolar':
            self.logger.info("Using dipolar coupling as correlation strength.")
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")
            correlation_strengths = get_pair_dipolar_couplings(self.atoms, self.pairs, self.isotopes)
        elif self.correlation_strength_metric == 'distance' or self.correlation_strength_metric == 'inversedistance':
            log_message = "Using minimum image convention {isinverse}distance as correlation strength."
            isinverse = ''

            # now we can use ASE get_distance to get the distances for each pair
            correlation_strengths = self.pair_distances
            if self.correlation_strength_metric == 'inversedistance':
                correlation_strengths = 1 / correlation_strengths
                isinverse = 'inverse '
            self.logger.info(log_message.format(isinverse=isinverse))


        elif self.correlation_strength_metric == 'jcoupling':
            self.logger.info("Using J-coupling as correlation strength.")
            correlation_strengths = get_pair_j_couplings(self.atoms, self.pairs, self.isotopes)
        elif self.correlation_strength_metric == 'custom':
            self.logger.info("Using custom correlation strengths.")
            correlation_strengths = self.correlation_strengths
        else:
            raise ValueError(f"Unknown correlation_strength_metric option: {self.correlation_strength_metric}")

        # Make sure correlation_strengths is an array
        correlation_strengths = np.array(correlation_strengths)

        self.logger.debug(f"correlation_strengths : {correlation_strengths}")
        # Log pair with smallest and largest correlation strength
        min_idx = np.argmin(np.abs(correlation_strengths))
        max_idx = np.argmax(np.abs(correlation_strengths))
        smallest_pair = self.pairs[min_idx]
        largest_pair = self.pairs[max_idx]

        # Labels for the smallest and largest pairs
        labels = get_atom_labels(self.atoms, self.logger)
        smallest_pair_labels = [labels[smallest_pair[0]], labels[smallest_pair[1]]]
        largest_pair_labels = [labels[largest_pair[0]], labels[largest_pair[1]]]

        self.logger.info(
            f"Pair with smallest (abs) {self.correlation_label}: "
            f"{smallest_pair_labels} ({correlation_strengths[min_idx]:.2f})"
        )
        self.logger.info(
            f"Pair with largest (abs) {self.correlation_label}: "
            f"{largest_pair_labels} ({correlation_strengths[max_idx]:.2f})"
        )

        return correlation_strengths

    def get_axis_labels(self):
        if self.is_shift:
            axis_label = r"$\delta$"
        else:
            axis_label = r"$\sigma$"

        if self.x_axis_label is None:
            self.x_axis_label = f'{self.xspecies} ' + axis_label + ' /ppm'
        if self.y_axis_label is None:
            if self.yaxis_order == '2Q':
                self.y_axis_label = f'{self.yspecies} ' + axis_label + r'$_{\mathrm{%s}}$' % self.yaxis_order + ' /ppm'
            else:
                self.y_axis_label = f'{self.yspecies} ' + axis_label + ' /ppm'


    def get_pairs(self):
        '''
        Get the pairs of x and y indices to plot

        self.idx_x and self.idx_y are the indices of the x and y elements in the atoms object

        This method will set the following attributes:
        self.pairs_el_idx: a list of tuples of the form (xindex, yindex)
        self.pairs: a list of tuples of the form (xindex, yindex)
        '''
        # Process pairs
        self.pairs, self.pairs_el_idx, self.idx_x, self.idx_y = process_pairs(self.idx_x, self.idx_y, self.pairs)

        # Check for invalid pairs if correlation_strength_metric is not fixed
        if self.correlation_strength_metric != 'fixed':
            for pair in self.pairs:
                if len(set(pair)) != 2:
                    raise ValueError("""
                    Two indices in a pair are the same but
                    the correlation_strength_metric is based on distance between sites.
                    It's unclear """)

        if len(self.pairs) == 0:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        # Calculate distances
        # if pair_distances is not already calculated, calculate it
        if not hasattr(self, 'pair_distances'):
            self.pair_distances = calculate_distances(self.pairs, self.atoms)


        # Filter pairs by distance if rcut is specified
        if self.rcut:
            self.logger.info(f"Filtering out pairs that are further than {self.rcut} Å apart.")
            self.logger.info(f"Number of pairs before filtering: {len(self.pairs_el_idx)}")

            self.pairs, self.pairs_el_idx, self.pair_distances, self.idx_x, self.idx_y = filter_pairs_by_distance(
                self.pairs, self.pairs_el_idx, self.pair_distances, self.rcut)

            self.logger.info(f"Number of pairs remaining: {len(self.pairs_el_idx)}")
            self.logger.debug(f"Pairs remaining: {self.pairs}")
            self.logger.debug(f"Pairs el indices remaining: {self.pairs_el_idx}")




@dataclass
class PlotSettings:
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    plot_filename: Optional[str] = None
    show_markers: bool = True
    marker: str = '+'
    marker_linewidth: float = 0.5
    max_marker_size: int = 10
    show_labels: bool = True
    auto_adjust_labels: bool = True
    label_fontsize: Optional[int] = None
    show_lines: bool = True
    show_diagonal: bool = True
    show_connectors: bool = True
    marker_color: Optional[str] = None
    show_legend: bool = False
    num_legend_elements: Optional[int] = None
    show_heatmap: bool = False
    heatmap_levels: Union[int, Iterable[float]] = 20
    show_contour: bool = False
    x_broadening: Optional[float] = None
    y_broadening: Optional[float] = None
    broadening_type: str = 'gaussian' # 'gaussian', 'lorentzian'
    heatmap_grid_size: int = 150
    colormap: str = 'bone_r'
    contour_color: str = 'C1'
    contour_linewidth: float = 0.2
    # To specify the contour levels, either provide a list of values or a integer number of contours between min and max
    contour_levels: Union[Iterable[float], int] = 10


class NMRPlot2D:
    '''
    Class to plot 2D NMR data.
    '''
    def __init__(self,
                nmr_data: NMRData2D,
                plot_settings: Optional[PlotSettings] = None):

        self.nmr_data = nmr_data
        # store the data as numpy arrays for plotting
        npeaks = len(self.nmr_data.peaks)
        self.x = np.zeros(npeaks)
        self.y = np.zeros(npeaks)
        self.sizes = np.zeros(npeaks)

        for i, peak in enumerate(self.nmr_data.peaks):
            self.x[i] = peak.x
            self.y[i] = peak.y
            self.sizes[i] = peak.correlation_strength


        # Use default plot settings if none are provided
        if plot_settings is None:
            plot_settings = PlotSettings()
        self.plot_settings = plot_settings

        self._initialize_plot_settings()

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        # If not set, set number of legend elements to
        # minimum of number of peaks and 5
        if self.plot_settings.num_legend_elements is None:
            self.plot_settings.num_legend_elements = min(npeaks, DEFAULT_MAX_NUM_LEGEND_ELEMENTS)

    def _initialize_plot_settings(self):
        for key, value in self.plot_settings.__dict__.items():
            setattr(self, key, value)

    @styled_plot(nmr_base_style, nmr_2D_style)
    def plot(self, ax:Optional[Axes] = None):
        '''
        Plot the 2D NMR data.

        Parameters
        ----------
        ax : Optional[Axes], optional
            The axes to plot the data on. If not provided, a new figure and axis will be created.

        Returns
        -------
        fig : Figure
            The figure object
        ax : Axes
            The axis object
        '''

        #  Create a new figure and axis if not provided
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, Axes):
            fig = ax.get_figure()
        else:
            raise TypeError("ax must be an Axes object or None.")

        self.ax = ax
        self.fig = fig


        # --- plot lines at peak locations ---
        if self.plot_settings.show_lines:
            self._plot_axlines()

        # --- scatter plot of peaks ---
        if self.plot_settings.show_markers:
            self._plot_markers()


        # --- connectors if required ---
        if self.plot_settings.show_connectors:
            self._plot_connectors()

        # --- plot the axis labels ---
        # Use the x and y axis labels from the plot settings if provided, otherwise use the labels from the NMR data
        x_axis_label = self.plot_settings.x_axis_label if self.plot_settings.x_axis_label else self.nmr_data.x_axis_label
        y_axis_label = self.plot_settings.y_axis_label if self.plot_settings.y_axis_label else self.nmr_data.y_axis_label
        self.ax.set_xlabel(x_axis_label)
        self.ax.set_ylabel(y_axis_label)


        # other plot options
        if self.plot_settings.xlim:
            xlim = self.plot_settings.xlim
            # we handle the inverting of the axes later
            # so we just need to make sure the limits are in the right order here
            self.plot_settings.xlim = (min(xlim), max(xlim))
            self.ax.set_xlim(self.plot_settings.xlim)

        if self.plot_settings.ylim:
            ylim = self.plot_settings.ylim
            # we handle the inverting of the axes later
            # so we just need to make sure the limits are in the right order here
            self.plot_settings.ylim = (min(ylim), max(ylim))
            self.ax.set_ylim(self.plot_settings.ylim)

        # if shifts are plotted, invert the axes
        if self.nmr_data.is_shift:
            self.ax.invert_xaxis()
            self.ax.invert_yaxis()

        # --- plot the diagonal line ---
        xelem_same_as_yelem = self.nmr_data.xelement == self.nmr_data.yelement and self.nmr_data.xelement is not None
        if xelem_same_as_yelem and self.plot_settings.show_diagonal:
            # use self.xlim and self.ylim to draw a diagonal line
            ylims = self.ax.get_ylim()
            xlims = self.ax.get_xlim()
            self.ax.plot(xlims, ylims, ls='--', c='k', lw=1, alpha=0.2)

        if self.plot_settings.show_heatmap or self.plot_settings.show_contour:
            X, Y, Z = self._get_contour_data()

        # --- heatmap of peaks ---
        if self.plot_settings.show_heatmap:
            self._plot_heatmap(X, Y, Z)

        if self.plot_settings.show_contour:
            self._plot_contour(X, Y, Z)

        # --- plot the site annotations ---
        if self.plot_settings.show_labels:
            self._plot_annotations(optimise=self.plot_settings.auto_adjust_labels)


        # Display or save the plot
        self.fig.tight_layout()

        if self.plot_settings.plot_filename:
            self.fig.savefig(self.plot_settings.plot_filename, dpi=300, bbox_inches='tight')

        return self.fig, ax
    def _plot_axlines(self):
        #  we don't want to plot identical lines multiple times
        xticks = np.unique(np.round(self.x, 6))
        yticks = np.unique(np.round(self.y, 6))
        # Plot lines at the peak locations
        for x in xticks:
            self.ax.axvline(x, zorder=0)
        for y in yticks:
            self.ax.axhline(y, zorder=0)

    def _get_contour_data(self):
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        if self.plot_settings.x_broadening is None:
            # set the broadening to 5% of the range
            self.plot_settings.x_broadening = 0.05 * abs(xlims[1] - xlims[0])
        if self.plot_settings.y_broadening is None:
            # set the broadening to 5% of the range
            self.plot_settings.y_broadening = 0.05 * abs(ylims[1] - ylims[0])

        X, Y, Z = generate_contour_map(
                    self.nmr_data.peaks,
                    grid_size = self.plot_settings.heatmap_grid_size,
                    broadening = self.plot_settings.broadening_type,
                    x_broadening=self.plot_settings.x_broadening,
                    y_broadening=self.plot_settings.y_broadening,
                    xlims = self.plot_settings.xlim,
                    ylims = self.plot_settings.ylim)
        return X, Y, Z

    def _plot_heatmap(self, X, Y, Z):
        # Plot the heatmap
        if isinstance(self.plot_settings.heatmap_levels, int):
            self.plot_settings.heatmap_levels = np.linspace(Z.min(), Z.max(), self.plot_settings.heatmap_levels)
        cs = self.ax.contourf(X, Y, Z, cmap=self.plot_settings.colormap, zorder=-1, levels=self.plot_settings.heatmap_levels)

    def _plot_contour(self, X, Y, Z):
        # fig.colorbar(cs, cax=cbar_ax, orientation='vertical', label='Intensity')
        if isinstance(self.plot_settings.contour_levels, (int, float)):
            self.plot_settings.contour_levels = np.linspace(Z.min(), Z.max(), self.plot_settings.contour_levels)

        # Add contour lines
        if self.plot_settings.show_contour:
            self.ax.contour(
                X, Y, Z,
                colors=self.plot_settings.contour_color,
                linewidths=self.plot_settings.contour_linewidth,
                levels=self.plot_settings.contour_levels
                )

    def _plot_connectors(self):
        if self.plot_settings.show_connectors and self.nmr_data.yaxis_order == '2Q' and self.nmr_data.xelement == self.nmr_data.yelement:
            x = self.x
            y = self.y
            y_order = np.argsort(y)
            # loop over peaks and plot lines between peaks with the same y value
            for i, idx in enumerate(y_order):
                if np.isclose(y[idx], y[y_order[i-1]], atol=1e-6):
                    self.ax.plot([x[idx], x[y_order[i-1]]],
                            [y[idx], y[y_order[i-1]]],
                            c='0.25',
                            lw=0.75,
                            ls='-',
                            zorder=1)

    def _plot_markers(self):
        if self.plot_settings.marker_color is None:
            # use peak colors
            color = [peak.color for peak in self.nmr_data.peaks]
            # if all colors are the same, use unique color
            if len(set(color)) == 1:
                color = color[0]

        else:
            color = self.plot_settings.marker_color

        scatter = self.ax.scatter(
                        self.x,
                        self.y,
                        s=self._normalize_marker_sizes(self.sizes),
                        c=color,
                        marker=self.plot_settings.marker,
                        linewidths=self.plot_settings.marker_linewidth,
                        zorder=10)
        if self.plot_settings.show_legend:
            # produce a legend with a cross-section of sizes from the scatter
            kw = dict(prop="sizes", num=self.plot_settings.num_legend_elements, color=color,
                      fmt=self.nmr_data.correlation_fmt + f" {self.nmr_data.correlation_unit}",
                      func=lambda s: s*np.abs(self.sizes).max() / self.plot_settings.max_marker_size)
            handles, labels = scatter.legend_elements(**kw) # type: ignore
            self.ax.legend(handles, labels,
                      title=self.nmr_data.correlation_label,
                      fancybox=True,
                      framealpha=0.8).set_zorder(12)

    def _normalize_marker_sizes(self, sizes):
        # Normalize the marker sizes, making sure all sizes are positive
        sizes = np.abs(sizes)
        marker_size_range = np.max(sizes) - np.min(sizes)
        self.logger.info(f"Marker size range: {marker_size_range} {self.nmr_data.correlation_unit}")
        max_abs_marker = np.max(sizes)
        # Normalize the marker sizes such that the maximum marker size is self.plot_settings.max_marker_size
        return sizes / max_abs_marker * self.plot_settings.max_marker_size

    def _plot_annotations(self, unique=True, optimise = True, labels_offset = 0.10):
        '''
        Get the annotations for the plot
        '''
        self.annotations = []

        # scale general font size by ANNOTATION_FONT_SCALE unless plot settings are provided
        font_size = self.plot_settings.label_fontsize
        if font_size is None:
            font_size = self.ax.xaxis.label.get_fontsize() * ANNOTATION_FONT_SCALE # type: ignore

        xpos, ypos = self.x, self.y
        xpos_label = xpos.copy()
        ypos_label = ypos.copy()

        xlabels = [peak.xlabel for peak in self.nmr_data.peaks] # type: ignore
        ylabels = [peak.ylabel for peak in self.nmr_data.peaks] # type: ignore

        # get the unique labels, keeping the indices
        if unique:
            xlabels, xidx = np.unique(xlabels, return_index=True)
            ylabels, yidx = np.unique(ylabels, return_index=True)
            # update the positions
            xpos = xpos[xidx]
            ypos = ypos[yidx]
            xpos_label = xpos_label[xidx]
            ypos_label = ypos_label[yidx]

        # we might still have some ylabels that are not the same but are at exactly the same position
        # so we need to check for this
        if len(ylabels) != len(np.unique(np.round(ypos,6))):
            # randomly perturb those y positions that are the same
            unique_ypos, idx, counts = np.unique(np.round(ypos,6), return_index=True, return_counts=True)
            for i, count in enumerate(counts):
                if count > 1:
                    ypos_label[idx[i]:idx[i]+count] += np.random.uniform(-0.5, 0.5, count)


        self.logger.debug(f'X labels: {xlabels}')
        self.logger.debug(f'X positions: {xpos}')
        self.logger.debug(f'Y labels: {ylabels}')
        self.logger.debug(f'Y positions: {ypos}')

        # TODO make a dynamical way to set the armA and armB value
        # based on the plot size
        if self.plot_settings.plot_filename is None:
            armA = 15
            armB = 15
        elif self.plot_settings.plot_filename.endswith('.pdf'):
            armA = 3
            armB = 5
        else:
            armA = 20
            armB = 30

        ######## Create x labels at the top axis ##########
        texts = []
        for i, xlabel in enumerate(xlabels):
            an = self.ax.annotate(
                xlabel,
                xy=(xpos[i], 1.0),  # Position of the annotation
                xycoords=('data', 'axes fraction'),  # Coordinate system for the annotation # type: ignore
                xytext=(xpos_label[i], 1+labels_offset),  # Position of the text
                textcoords=('data', 'axes fraction'),  # Coordinate system for the text # type: ignore
                fontsize=font_size,  # Font size of the text
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                rotation=90,  # Rotate the text 90 degrees
                arrowprops=dict(
                    arrowstyle="-",  # Style of the arrow
                    connectionstyle=f"arc,angleA=-90,armA={armA},angleB=90,armB={armB},rad=0",  # Connection style of the arrow
                    relpos=(0.5, 0.0),  # Relative position of the arrow
                    lw=ANNOTATION_LINE_WIDTH,  # Line width of the arrow
                    shrinkA=0.0,  # Shrink factor at the start of the arrow
                    shrinkB=0.0,  # Shrink factor at the end of the arrow
                ),
            )
            texts.append(an)  # Add the annotation to the list

        if optimise:
            # Adjust the text annotations to avoid overlap
            adjust_text(
                texts,
                ensure_inside_axes=False,
                avoid_self=False,
                force_pull=(0.0, 0.0),
                force_text=(0.3, 0.0),
                force_explode=(1.5, 0.0),
                expand=(1.3, 1.0),
                max_move=2,
            )
        self.annotations += texts  # Add the adjusted texts to the annotations

        ######## Create y labels at the right axis ##########
        texts = []
        for i, ylabel in enumerate(ylabels):
            an = self.ax.annotate(
                ylabel,
                xy=(1.0, ypos[i]),  # Position of the annotation
                xycoords=('axes fraction', 'data'),  # Coordinate system for the annotation # type: ignore
                xytext=(1+labels_offset, ypos_label[i]),  # Position of the text
                textcoords=('axes fraction', 'data'),  # Coordinate system for the text # type: ignore
                fontsize=font_size,  # Font size of the text
                ha='left',  # Horizontal alignment
                va='center',  # Vertical alignment
                arrowprops=dict(
                    arrowstyle="-",  # Style of the arrow
                    connectionstyle=f"arc,angleA=180,armA={armA},angleB=0,armB={armB},rad=0",  # Connection style of the arrow
                    relpos=(0.0, 0.5),  # Relative position of the arrow
                    lw=ANNOTATION_LINE_WIDTH,  # Line width of the arrow
                    shrinkA=0.0,  # Shrink factor at the start of the arrow
                    shrinkB=0.0,  # Shrink factor at the end of the arrow
                ),
            )
            texts.append(an)  # Add the annotation to the list

        if optimise:
            # Adjust the text annotations to avoid overlap
            adjust_text(
                texts,
                ensure_inside_axes=False,
                avoid_self=False,
                force_pull=(0.0, 0.0),
                force_text=(0.4, 0.8),
                force_explode=(0.0, 1.2),
                expand=(1.0, 1.8),
                max_move=1,
            )
        self.annotations += texts  # Add the adjusted texts to the annotations
