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
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from ase import Atoms
from matplotlib.axes import Axes
from scipy.special import fresnel

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from soprano.calculate.nmr.utils import (
    ContourData,
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
from soprano.properties.nmr import DipolarRSSByAtom, MSIsotropy
from soprano.selection import AtomSelection
from soprano.utils import minimum_supcell, supcell_gridgen

DEFAULT_MARKER_SIZE = 50 # Default marker size for 2D plots (max is 100)
ANNOTATION_LINE_WIDTH = 0.15 # Line width for annotation lines
ANNOTATION_FONT_SCALE = 0.5 # Scale factor for annotation font size wrt to plot font size
DEFAULT_MAX_NUM_LEGEND_ELEMENTS = 6 # Default maximum number of elements to show in the legend in 2D plots

_nmr_data = _get_nmr_data()

# Mapping of matplotlib colormaps to Plotly colorscales
MPL_TO_PLOTLY_COLORMAP = {
    'bone_r': 'greys_r',
    'bone': 'greys',
    'viridis': 'viridis',
    'plasma': 'plasma',
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'hot': 'hot',
    'cool': 'ice',
    'gray': 'greys',
    'grey': 'greys',
}

# Mapping of matplotlib markers to Plotly symbols (using -open for hollow markers)
MPL_TO_PLOTLY_MARKER = {
    'o': 'circle',
    's': 'square',
    '^': 'triangle-up',
    'v': 'triangle-down',
    'D': 'diamond',
    'p': 'pentagon',
    '*': 'star',
    'x': 'x-thin',
    '+': 'cross-thin',
}

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
    'dipolar2': {
        'label': 'Dipolar Coupling²',
        'unit': 'kHz²',
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
    },
    'dipolar_rss': {
        'label': 'Dipolar RSS',
        'unit': 'kHz',
        'fmt': '{x:.1f}'
    },
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
                correlation_strength_metric: Optional[str] = None, # 'fixed','distance', 'dipolar', 'dipolar2', 'jcoupling', 'inversedistance', 'custom', 'dipolar_rss'
                rcut: Optional[float] = None,
                rss_cutoff: float = 5.0,
                rss_expand_j: str = 'periodic_images',
                reduce: bool = False,
                symprec: float = 1e-4,
                atoms_full: Optional[Atoms] = None,
                isotopes: Optional[dict[str, int]] = None,
                is_shift: Optional[bool] = None,
                include_quadrupolar: bool = False,
                yaxis_order: str = '1Q',
                x_axis_label: Optional[str] = None,
                y_axis_label: Optional[str] = None,
                average_group: str = "",
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

        # Reduce to unique sites if requested — mirrors the CLI --reduce flag.
        # For dipolar_rss the full-cell atoms are needed so the expand_j
        # expansion can find all equivalent neighbours; store them before
        # merging duplicate sites away.
        if reduce and self.atoms is not None:
            from soprano.nmr.extract import label_atoms, nmr_extract_atoms  # lazy import
            self.atoms = label_atoms(self.atoms)
            if atoms_full is None:
                atoms_full = self.atoms   # keep labeled full atoms for RSS
            _reduced = nmr_extract_atoms(self.atoms.copy(), reduce=True, symprec=symprec)
            if _reduced is not None:
                self.atoms = _reduced


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
        self.rss_cutoff = rss_cutoff
        self.rss_expand_j = rss_expand_j
        self.symprec = symprec
        self.atoms_full = atoms_full
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
        self.average_group = average_group

        # run the main method to extract the data
        self.get_peaks()

    def _average_group_peaks(self, peaks: list) -> list:
        """Merge peaks from the same functional group (e.g. CH₃) post-hoc.

        This method operates on
        peaks generated from the *original*, unmodified atoms object so that
        every coupling is evaluated at the true inter-atomic distance.

        Peaks that belong to the same functional group and are paired with the
        same counter-atom are replaced by a single representative peak:

        * **position** (x, y) — mean of the group members' positions,
        * **correlation_strength** — mean of the members' coupling values
          (the multiplicity weight accounts for group size separately),
        * **multiplicity** — sum of the members' multiplicities
          (equals the group size when starting from the default of 1),
        * **label** — comma-separated sorted concatenation of member labels,
          matching the CLI ``--average-group`` convention.

        Args:
            peaks: Peak list from :func:`generate_peaks` (before
                :func:`merge_peaks`).

        Returns:
            New peak list with functional-group members merged.
        """
        import re  # noqa: PLC0415
        from collections import defaultdict  # noqa: PLC0415
        from dataclasses import replace as _replace  # noqa: PLC0415

        from soprano.nmr.extract import find_XHn_groups, label_atoms  # noqa: PLC0415

        # --- build atom → group-id mapping ---------------------------------
        atoms = label_atoms(self.atoms.copy(), logger=self.logger)
        all_groups = find_XHn_groups(atoms, self.average_group)

        atom_to_group: dict = {}
        for ipat, pattern_groups in enumerate(all_groups):
            for igrp, group in enumerate(pattern_groups):
                gid = (ipat, igrp)
                for aidx in group:
                    atom_to_group[int(aidx)] = gid

        if not atom_to_group:
            self.logger.warning(
                f"average_group='{self.average_group}' matched no groups; "
                "returning peaks unchanged."
            )
            return peaks

        def gkey(idx: int):
            """Group-id if atom is in a group, else its own index as sentinel."""
            return atom_to_group.get(int(idx), int(idx))

        # --- bucket peaks by merged-pair key --------------------------------
        peer_map: dict = defaultdict(list)
        for peak in peaks:
            key = (gkey(peak.idx_x), gkey(peak.idx_y))
            peer_map[key].append(peak)

        # --- combine each bucket --------------------------------------------
        def _sort_key(label: str):
            m = re.search(r'\d+', label)
            return int(m.group()) if m else label

        result = []
        for group_peaks in peer_map.values():
            if len(group_peaks) == 1:
                result.append(group_peaks[0])
                continue

            avg_x = float(np.mean([p.x for p in group_peaks]))
            avg_y = float(np.mean([p.y for p in group_peaks]))
            total_mult = sum(p.multiplicity for p in group_peaks)
            # Use multiplicity-weighted mean so that the invariant
            # merged_strength × merged_multiplicity = Σ(sᵢ × mᵢ)
            # is preserved — consistent with merge_peaks.
            total_weight = sum(p.correlation_strength * p.multiplicity for p in group_peaks)
            avg_strength = total_weight / total_mult if total_mult != 0 else 0.0

            xlabels = sorted({p.xlabel for p in group_peaks}, key=_sort_key)
            ylabels = sorted({p.ylabel for p in group_peaks}, key=_sort_key)
            result.append(_replace(
                group_peaks[0],
                x=avg_x,
                y=avg_y,
                correlation_strength=avg_strength,
                xlabel=','.join(xlabels),
                ylabel=','.join(ylabels),
                multiplicity=total_mult,
            ))

        n_before, n_after = len(peaks), len(result)
        if n_before != n_after:
            self.logger.debug(
                f"average_group='{self.average_group}': "
                f"merged {n_before} → {n_after} peaks."
            )
        return result

    def get_peaks(self, merge_identical=True, should_sort_peaks=False, force_recompute=False):
        '''
        Get the correlation peaks.

        If self.peaks already exists, then we return them as is.
        If they don't exist, we make sure the required data is available 
        and then generate the peaks and merge if desired.

        Set force_recompute=True to discard any cached peaks and regenerate
        from the underlying atoms/pairs data.
        '''

        if force_recompute:
            self.peaks = None

        if self.peaks is not None:
            self.logger.debug("Cached peaks found. Returning without recomputing. Use force_recompute=True to regenerate.")
            return self.peaks

        if self.atoms is None:
            raise ValueError("Either atoms or peaks must be given.")

        # make sure all the data is there
        self.extract_data()
        labels = get_atom_labels(self.atoms, self.logger)

        multiplicities = (
            self.atoms.get_array('multiplicity')
            if self.atoms is not None and self.atoms.has('multiplicity')
            else None
        )
        self.peaks = generate_peaks(self.data, self.pairs, labels, self.correlation_strengths, self.yaxis_order, self.xelement, self.yelement, multiplicities=multiplicities)

        if self.average_group:
            self.peaks = self._average_group_peaks(self.peaks)

        if merge_identical:
            self.peaks = merge_peaks(self.peaks, corr_rel_tol=0.05)

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

        # Check that self.atoms is not None unless the metric is 'custom' or 'fixed', which don't require atoms
        if self.atoms is None and self.correlation_strength_metric not in ('custom', 'fixed'):
            raise ValueError(f"atoms must be provided to calculate correlation strengths for metric '{self.correlation_strength_metric}'.")
        
        if self.pairs is None:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        if self.correlation_strength_metric == 'custom':
            self.logger.info("Using custom correlation strengths.")

            # make sure correlation_strengths is a list or an array
            if not isinstance(self.correlation_strengths, (list, np.ndarray)):
                raise TypeError("correlation_strengths must be a list or an array.")

            # if user provides a list of these, use it!
            # just check that it's the right length
            if len(self.correlation_strengths) != len(self.pairs):
                raise ValueError(f"Length of correlation_strengths ({len(self.correlation_strengths)}) does not match the number of pairs ({len(self.pairs)}).")
            correlation_strengths = self.correlation_strengths

        elif self.correlation_strength_metric == 'fixed':
            self.logger.info("Using fixed correlation strength.")
            # set the correlation strength to be the same for all pairs
            correlation_strengths = np.ones(len(self.pairs))

        elif self.correlation_strength_metric in ('dipolar', 'dipolar2'):
            self.logger.info(f"Using {self.correlation_label.lower()} as correlation strength.")
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")
            correlation_strengths = get_pair_dipolar_couplings(self.atoms, self.pairs, self.isotopes)
            if self.correlation_strength_metric == 'dipolar2':
                correlation_strengths = np.square(correlation_strengths)
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
        elif self.correlation_strength_metric == 'dipolar_rss':
            self.logger.info(
                f"Using dipolar RSS (cutoff={self.rss_cutoff} Å, "
                f"expand_j='{self.rss_expand_j}') as correlation strength."
            )
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")

            if self.rss_expand_j != 'periodic_images' and self.atoms_full is not None:
                # When the structure has been reduced to the asymmetric unit
                # (reduce=True), equivalent sites have been merged away, so
                # expand_j='cif_labels'/'symmetry' would find nothing to expand
                # in self.atoms.  Instead, map pair indices to the full (unmerged)
                # atoms via CIF labels, then let DipolarRSSByAtom expand there.
                reduced_labels = get_atom_labels(self.atoms, self.logger)
                full_labels = get_atom_labels(self.atoms_full, self.logger)

                def _first_full_idx(label):
                    matches = np.where(full_labels == label)[0]
                    if len(matches) == 0:
                        raise ValueError(
                            f"Label '{label}' from reduced atoms not found in "
                            "atoms_full. Ensure atoms_full is labeled consistently."
                        )
                    return int(matches[0])

                correlation_strengths = np.array([
                    DipolarRSSByAtom.get(
                        self.atoms_full,
                        sel_i=[_first_full_idx(reduced_labels[i])],
                        sel_j=[_first_full_idx(reduced_labels[j])],
                        cutoff=self.rss_cutoff,
                        isotopes=self.isotopes,
                        expand_j=self.rss_expand_j,
                        symprec=self.symprec,
                    )[0]
                    for i, j in self.pairs
                ]) * 1e-3  # convert Hz → kHz to match MARKER_INFO and dipolar metric
            else:
                if self.rss_expand_j != 'periodic_images':
                    self.logger.warning(
                        f"rss_expand_j='{self.rss_expand_j}' requested but atoms_full "
                        "was not provided. If the structure has been reduced to the "
                        "asymmetric unit, the expansion will find no additional sites. "
                        "Pass atoms_full (the full unit-cell atoms) to NMRData2D, or "
                        "use reduce=True to let NMRData2D handle this automatically."
                    )
                correlation_strengths = np.array([
                    DipolarRSSByAtom.get(
                        self.atoms,
                        sel_i=[i],
                        sel_j=[j],
                        cutoff=self.rss_cutoff,
                        isotopes=self.isotopes,
                        expand_j=self.rss_expand_j,
                        symprec=self.symprec,
                    )[0]
                    for i, j in self.pairs
                ]) * 1e-3  # convert Hz → kHz to match MARKER_INFO and dipolar metric
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
        # Process pairs; idx_x/idx_y (unique element indices) are not overwritten here —
        # process_pairs expands them into per-pair form internally.
        self.pairs, self.pairs_el_idx, _, _ = process_pairs(self.idx_x, self.idx_y, self.pairs)

        # In a DQ/SQ experiment the diagonal peaks arise from two *distinct*
        # atoms of the same chemical shift — not from a spin correlated with
        # itself.  The literal self-pair (i, i) is unphysical: the inter-nuclear
        # distance is zero so the dipolar coupling diverges, and it contributes
        # nothing to the real spectrum.  Remove all self-pairs unconditionally
        # in DQ mode so that diagonal peaks emerge naturally from the physics.
        if self.yaxis_order == '2Q':
            n_before = len(self.pairs)
            valid = [(i, p) for i, p in enumerate(self.pairs) if p[0] != p[1]]
            if not valid:
                raise ValueError(
                    "No valid (non-self) pairs found for DQ/SQ spectrum. "
                    "Diagonal peaks in a DQ experiment come from distinct atoms "
                    "with the same shift — check that the structure contains "
                    "coupled pairs of the requested elements."
                )
            idxs, self.pairs = zip(*valid)
            self.pairs = list(self.pairs)
            self.pairs_el_idx = [self.pairs_el_idx[i] for i in idxs]
            n_removed = n_before - len(self.pairs)
            if n_removed:
                self.logger.debug(
                    f"Removed {n_removed} self-pair(s) (i==i) — unphysical in DQ/SQ."
                )

        # For distance-based metrics, self-pairs would cause division by zero;
        # after the DQ filter above they are already gone for 2Q mode.
        # For SQ/homonuclear mode we still want to flag them rather than silently
        # produce NaN couplings.
        if self.correlation_strength_metric != 'fixed':
            self_pairs = [p for p in self.pairs if p[0] == p[1]]
            if self_pairs:
                self.logger.warning(
                    f"{len(self_pairs)} self-pair(s) remain after filtering with "
                    f"metric='{self.correlation_strength_metric}'. "
                    "Distance/coupling for a self-pair is undefined. "
                    "Consider using --rcut or passing explicit pairs."
                )

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

            self.pairs, self.pairs_el_idx, self.pair_distances, _, _ = filter_pairs_by_distance(
                self.pairs, self.pairs_el_idx, self.pair_distances, self.rcut)

            self.logger.info(f"Number of pairs remaining: {len(self.pairs_el_idx)}")
            self.logger.debug(f"Pairs remaining: {self.pairs}")
            self.logger.debug(f"Pairs el indices remaining: {self.pairs_el_idx}")

    def to_dataframe(self, include_metadata=True):
        """
        Convert the NMR data to a pandas DataFrame.
        
        This method exports all peak data and optionally includes metadata.
        The resulting DataFrame can be saved to CSV or other formats for
        later use or sharing.
        
        Parameters
        ----------
        include_metadata : bool, optional
            If True, includes columns with experimental metadata such as
            elements, isotopes, references, etc. Default is True.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing peak positions (x, y), labels (xlabel, ylabel),
            correlation strengths, and optionally metadata.
        
        Raises
        ------
        ImportError
            If pandas is not installed.
        
        Examples
        --------
        >>> df = nmr_data.to_dataframe()
        >>> df.to_csv('nmr_peaks.csv', index=False)
        
        >>> # Load back from CSV
        >>> import pandas as pd
        >>> df = pd.read_csv('nmr_peaks.csv')
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required to export data to DataFrame. "
                "Install it with: pip install pandas"
            )
        
        # Get peaks (this will generate them if needed)
        peaks = self.get_peaks()
        
        if not peaks:
            self.logger.warning("No peaks to export.")
            return pd.DataFrame()
        
        # Extract data from Peak2D namedtuples
        data = {
            'x': [peak.x for peak in peaks],
            'y': [peak.y for peak in peaks],
            'xlabel': [peak.xlabel for peak in peaks],
            'ylabel': [peak.ylabel for peak in peaks],
            'correlation_strength': [peak.correlation_strength for peak in peaks],
        }
        
        if include_metadata:
            # Add metadata columns (same value for all rows)
            data.update({
                'xelement': self.xelement,
                'yelement': self.yelement,
                'xisotope': self.xisotope if hasattr(self, 'xisotope') else None,
                'yisotope': self.yisotope if hasattr(self, 'yisotope') else None,
                'correlation_metric': self.correlation_strength_metric,
                'correlation_unit': self.correlation_unit,
                'correlation_label': self.correlation_label,
                'yaxis_order': self.yaxis_order,
                'is_shift': self.is_shift,
            })
            
            # Add references if they exist
            if self.references:
                data['xreference'] = self.references.get(self.xelement, None)
                data['yreference'] = self.references.get(self.yelement, None)
            
            # Add gradients if they exist
            if self.gradients:
                data['xgradient'] = self.gradients.get(self.xelement, None)
                data['ygradient'] = self.gradients.get(self.yelement, None)
            
            # Add rcut if it was used
            if self.rcut:
                data['rcut'] = self.rcut
        
        df = pd.DataFrame(data)
        
        self.logger.info(f"Exported {len(df)} peaks to DataFrame.")
        return df

    def get_contour_data(
        self,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        broadening_type: str = 'lorentzian',
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
    ) -> 'ContourData':
        """
        Compute (and cache) the 2D contour grid for this spectrum.

        The result is stored on ``self._contour_data`` and is reused as long
        as the same parameters are requested.  Pass any argument explicitly to
        force recomputation with different settings.

        Parameters
        ----------
        x_broadening : float, optional
            FWHM linewidth in the direct (x) dimension.  Defaults to 5 % of
            the x peak range.  Internally converted to HWHM (Lorentzian) or
            σ (Gaussian) as appropriate.
        y_broadening : float, optional
            FWHM linewidth in the indirect (y) dimension.  Same default logic
            as *x_broadening*.
        broadening_type : str
            ``'gaussian'`` (default) or ``'lorentzian'``.
        grid_size : int
            Number of points along each axis of the grid.  Default 150.
        xlims : tuple of float, optional
            Used **only** to compute the default 5 % broadening when
            *x_broadening* is *None*.  The grid itself always spans the full
            peak range plus ``5 × x_broadening`` padding on each side;
            passing *xlims* does not clip the grid.  Use
            ``PlotSettings.xlim`` to control the display limits.
        ylims : tuple of float, optional
            Same as *xlims* but for the indirect (y) dimension.

        Returns
        -------
        ContourData
            Named-tuple with fields ``X``, ``Y``, ``Z`` (meshgrid arrays),
            ``x_broadening``, ``y_broadening``, ``broadening_type``,
            ``xlims``, ``ylims``.
        """
        peaks = self.get_peaks()
        if not peaks:
            raise ValueError("No peaks available – cannot compute contour data.")

        # Use absolute correlation strengths: the heatmap shows *magnitude*
        # of correlation.  Signed metrics (e.g. negative dipolar constants)
        # would otherwise produce a map with negative intensities.
        from dataclasses import replace as _dc_replace
        peaks_abs = [
            _dc_replace(p, correlation_strength=abs(p.correlation_strength))
            for p in peaks
        ]

        # Resolve default broadening from peak spread or supplied limits
        if xlims is None:
            x_min = min(p.x for p in peaks_abs)
            x_max = max(p.x for p in peaks_abs)
        else:
            x_min, x_max = min(xlims), max(xlims)

        if ylims is None:
            y_min = min(p.y for p in peaks_abs)
            y_max = max(p.y for p in peaks_abs)
        else:
            y_min, y_max = min(ylims), max(ylims)

        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0

        if x_broadening is None:
            x_broadening = 0.05 * x_range
        if y_broadening is None:
            y_broadening = 0.05 * y_range

        # Check cache
        cache_key = (x_broadening, y_broadening, broadening_type, grid_size,
                     xlims, ylims)
        if getattr(self, '_contour_cache_key', None) == cache_key:
            return self._contour_data

        X, Y, Z = generate_contour_map(
            peaks_abs,
            grid_size=grid_size,
            broadening=broadening_type,
            x_broadening=x_broadening,
            y_broadening=y_broadening,
        )

        # Actual grid limits (may be wider than xlims due to broadening padding)
        actual_xlims = (float(X[0, 0]), float(X[0, -1]))
        actual_ylims = (float(Y[0, 0]), float(Y[-1, 0]))

        self._contour_data = ContourData(
            X=X,
            Y=Y,
            Z=Z,
            x_broadening=x_broadening,
            y_broadening=y_broadening,
            broadening_type=broadening_type,
            xlims=actual_xlims,
            ylims=actual_ylims,
        )
        self._contour_cache_key = cache_key
        return self._contour_data

    def export_contour_data(
        self,
        path: str,
        fmt: str = 'simpson',
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        broadening_type: str = 'lorentzian',
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
        x_larmor_freq_mhz: Optional[float] = None,
        y_larmor_freq_mhz: Optional[float] = None,
    ) -> None:
        """
        Export the 2D NMR contour data to a file.

        The contour grid is computed via :meth:`get_contour_data` (and cached
        for subsequent calls).  The peak list is always included alongside the
        grid where the format supports it.

        Parameters
        ----------
        path : str
            Output file path.  For ``'simpson'`` format use a ``.spe``
            extension so that nmrglue / ssNake auto-detect the file type.
        fmt : {'simpson', 'npz', 'csv', 'json', 'ssnake'}
            Export format:

            ``'simpson'``
                SIMPSON TEXT format (``TYPE=SPE``).  Readable by nmrglue
                (``nmrglue.fileio.simpson.read``) and ssNake
                (*Open → SIMPSON*).  The grid is written as a 2D real
                spectrum; the imaginary channel is zero everywhere.  A
                companion ``<path>.peaks.csv`` file is written alongside.

            ``'npz'``
                NumPy compressed archive.  Arrays ``X``, ``Y``, ``Z`` plus
                scalar metadata are stored.  Reload with
                ``np.load(path, allow_pickle=True)``.

            ``'csv'``
                Flat table with columns ``x``, ``y``, ``intensity``.
                Useful for import into Origin, Excel, etc.

            ``'json'``
                ssNake native JSON format.  Stores Larmor frequency
                (``freq``) directly, so **ppm is available immediately**
                on load without any manual axis editing.  Requires
                *x_larmor_freq_mhz* (and *y_larmor_freq_mhz* for
                heteronuclear spectra).

        x_larmor_freq_mhz : float, optional
            Larmor frequency in MHz for the **direct (x)** dimension nucleus.
            Only used by the ``'simpson'`` exporter.

        y_larmor_freq_mhz : float, optional
            Larmor frequency in MHz for the **indirect (y)** dimension nucleus.
            For homonuclear experiments this equals *x_larmor_freq_mhz*.
            For heteronuclear experiments (e.g. 13C on x, 1H on y) the two
            frequencies differ and both must be provided.

            *Why these matter for ssNake:*  The SIMPSON TEXT format has no
            field for the spectrometer frequency.  ssNake therefore sets the
            carrier to 0 MHz on load and cannot offer ppm as a unit.  When
            Larmor frequencies are provided:

            * ``SW`` is written in Hz using *x_larmor_freq_mhz*.
            * ``SW1`` is written in Hz using *y_larmor_freq_mhz* (falls
              back to *x_larmor_freq_mhz* when *y_larmor_freq_mhz* is None).

            After loading in ssNake, go to *Axes → Edit axes* and enter the
            appropriate Larmor frequency for each dimension; ppm will then
            be available.

            When both are *None* sweep widths are written in ppm and a
            warning is emitted.

        x_broadening, y_broadening, broadening_type, grid_size, xlims, ylims
            Forwarded to :meth:`get_contour_data`.  If the grid has already
            been cached with identical parameters the cached result is reused.

        Raises
        ------
        ValueError
            If no peaks are available or an unknown format is requested.
        """
        cd = self.get_contour_data(
            x_broadening=x_broadening,
            y_broadening=y_broadening,
            broadening_type=broadening_type,
            grid_size=grid_size,
            xlims=xlims,
            ylims=ylims,
        )

        fmt = fmt.lower().strip()

        if fmt == 'simpson':
            self._export_simpson(path, cd,
                                 x_larmor_freq_mhz=x_larmor_freq_mhz,
                                 y_larmor_freq_mhz=y_larmor_freq_mhz)
        elif fmt == 'npz':
            self._export_npz(path, cd)
        elif fmt == 'csv':
            self._export_csv_grid(path, cd)
        elif fmt in ('json', 'ssnake'):
            self._export_json_ssnake(path, cd,
                                     x_larmor_freq_mhz=x_larmor_freq_mhz,
                                     y_larmor_freq_mhz=y_larmor_freq_mhz)
        else:
            raise ValueError(
                f"Unknown export format '{fmt}'. "
                "Choose from 'simpson', 'npz', 'csv', 'json'."
            )
        self.logger.info(f"Exported contour data to '{path}' (format={fmt}).")

    # ------------------------------------------------------------------
    # Private export helpers
    # ------------------------------------------------------------------

    def _export_simpson(self, path: str, cd: 'ContourData',
                         x_larmor_freq_mhz: Optional[float] = None,
                         y_larmor_freq_mhz: Optional[float] = None) -> None:
        """Write a SIMPSON TEXT (.spe) file readable by nmrglue and ssNake.

        Sweep widths
        ------------
        SIMPSON/ssNake expect SW in Hz.  We convert: SW_hz = SW_ppm × freq_MHz.
        For heteronuclear spectra the two dimensions can have different Larmor
        frequencies (e.g. 13C direct, 1H indirect).  *y_larmor_freq_mhz*
        falls back to *x_larmor_freq_mhz* when not given (homonuclear case).

        A companion ``<path>.peaks.csv`` file is written with the peak list.
        """
        import csv

        Z = cd.Z                              # shape (NI, NP)
        NI, NP = Z.shape
        SW_ppm  = cd.xlims[1] - cd.xlims[0]  # direct (x) sweep width in ppm
        SW1_ppm = cd.ylims[1] - cd.ylims[0]  # indirect (y) sweep width in ppm

        # y falls back to x for homonuclear case
        y_freq = y_larmor_freq_mhz if y_larmor_freq_mhz is not None else x_larmor_freq_mhz

        if x_larmor_freq_mhz is not None:
            SW  = SW_ppm  * x_larmor_freq_mhz
            SW1 = SW1_ppm * y_freq
            sw_unit = 'Hz'
        else:
            SW  = SW_ppm
            SW1 = SW1_ppm
            sw_unit = 'ppm'
            self.logger.warning(
                "Exporting SIMPSON .spe without Larmor frequencies: SW/SW1 are "
                "written in ppm.  ssNake cannot select ppm as a unit without "
                "spectrometer frequencies.  Pass x_larmor_freq_mhz (and "
                "y_larmor_freq_mhz for heteronuclear spectra) to fix this."
            )

        with open(path, 'w') as f:
            f.write('SIMP\n')
            f.write(f'NP={NP}\n')
            f.write(f'NI={NI}\n')
            f.write(f'SW={SW:.8g}\n')
            f.write(f'SW1={SW1:.8g}\n')
            f.write('TYPE=SPE\n')
            f.write('# Exported by Soprano NMRData2D.export_contour_data\n')
            f.write(f'# SW_unit={sw_unit}\n')
            if x_larmor_freq_mhz is not None:
                f.write(f'# SPECFREQ_x={x_larmor_freq_mhz:.6g} MHz  (direct dim)\n')
                f.write(f'# SPECFREQ_y={y_freq:.6g} MHz  (indirect dim)\n')
                f.write(f'# ssNake: Axes -> Edit axes, set carriers to these values\n')
            f.write(f'# x_broadening={cd.x_broadening:.6g} ppm\n')
            f.write(f'# y_broadening={cd.y_broadening:.6g} ppm\n')
            f.write(f'# broadening_type={cd.broadening_type}\n')
            f.write(f'# xlims_ppm={cd.xlims[0]:.6g} {cd.xlims[1]:.6g}\n')
            f.write(f'# ylims_ppm={cd.ylims[0]:.6g} {cd.ylims[1]:.6g}\n')
            f.write('DATA\n')
            for i in range(NI):
                for j in range(NP):
                    f.write(f'{Z[i, j]:.8g} 0.0\n')
            f.write('END')

        # Write companion peak list
        peaks_path = path + '.peaks.csv'
        peaks = self.get_peaks()
        with open(peaks_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_ppm', 'y_ppm', 'xlabel', 'ylabel', 'correlation_strength'])
            for p in peaks:
                writer.writerow([p.x, p.y, p.xlabel, p.ylabel, p.correlation_strength])
        self.logger.info(f"Peak list written to '{peaks_path}'.")

    def _export_npz(self, path: str, cd: 'ContourData') -> None:
        """Write a NumPy compressed archive with the grid and metadata."""
        peaks = self.get_peaks()
        peak_x = np.array([p.x for p in peaks])
        peak_y = np.array([p.y for p in peaks])
        peak_strength = np.array([p.correlation_strength for p in peaks])
        peak_xlabels = np.array([p.xlabel for p in peaks])
        peak_ylabels = np.array([p.ylabel for p in peaks])

        np.savez_compressed(
            path,
            X=cd.X,
            Y=cd.Y,
            Z=cd.Z,
            peak_x=peak_x,
            peak_y=peak_y,
            peak_strength=peak_strength,
            peak_xlabels=peak_xlabels,
            peak_ylabels=peak_ylabels,
            x_broadening=cd.x_broadening,
            y_broadening=cd.y_broadening,
            broadening_type=np.bytes_(cd.broadening_type),
            xlims=np.array(cd.xlims),
            ylims=np.array(cd.ylims),
        )

    def _export_csv_grid(self, path: str, cd: 'ContourData') -> None:
        """Write a flat CSV with columns x, y, intensity."""
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_ppm', 'y_ppm', 'intensity'])
            NI, NP = cd.Z.shape
            for i in range(NI):
                for j in range(NP):
                    writer.writerow([cd.X[i, j], cd.Y[i, j], cd.Z[i, j]])

    def _export_json_ssnake(
        self,
        path: str,
        cd: 'ContourData',
        x_larmor_freq_mhz: Optional[float],
        y_larmor_freq_mhz: Optional[float],
    ) -> None:
        """Write an ssNake-native JSON file with Larmor frequencies embedded.

        ssNake stores ``freq`` (Larmor frequency per dimension, in Hz) directly
        in its JSON format.  On load this is used to compute the ppm axis
        without any manual entry — ppm is available immediately.

        The grid Z (NI × NP, purely real) is stored as a 1-D ``dataReal``
        array (row-major) with a zero ``dataImag`` partner.

        Parameters
        ----------
        path : str
            Output path.  Use a ``.json`` extension.
        cd : ContourData
            Grid to export.
        x_larmor_freq_mhz : float
            Larmor frequency in MHz for the direct (x) dimension.
        y_larmor_freq_mhz : float
            Larmor frequency in MHz for the indirect (y) dimension.
            Falls back to *x_larmor_freq_mhz* when None (homonuclear).
        """
        import json

        if x_larmor_freq_mhz is None:
            raise ValueError(
                "x_larmor_freq_mhz is required for 'json' export so that "
                "ssNake can display the ppm axis directly."
            )
        y_freq_mhz = y_larmor_freq_mhz if y_larmor_freq_mhz is not None else x_larmor_freq_mhz

        x_freq_hz = x_larmor_freq_mhz * 1e6   # direct   (x) dim
        y_freq_hz = y_freq_mhz         * 1e6   # indirect (y) dim

        # Unit conversion: ppm × MHz = Hz  (1e-6 × 1e6 = 1, exact)
        SW_x_hz = (cd.xlims[1] - cd.xlims[0]) * x_larmor_freq_mhz  # Hz
        SW_y_hz = (cd.ylims[1] - cd.ylims[0]) * y_freq_mhz          # Hz

        # ssNake ppm conversion: ppm = xax_hz * (1e6 / ref)
        # ssNake disables ppm if ref == 0, so ref must be non-zero.
        # With xax_hz = ppm * freq_MHz, setting ref = freq_hz gives:
        #   ppm = xax_hz * 1e6 / freq_hz = xax_hz / freq_MHz ✓
        ref_x = x_freq_hz   # Hz (= x_larmor_freq_mhz * 1e6)
        ref_y = y_freq_hz   # Hz (= y_freq_mhz * 1e6)

        NI, NP = cd.Z.shape
        # ssNake stores data as shape (n_hyper, NI, NP) — for non-hypercomplex
        # 2D data that is (1, NI, NP).  When 'hyper' is present in the JSON
        # ssNake does np.array(dataReal) directly, so the nesting must match.
        data_3d = cd.Z.reshape(1, NI, NP)
        flat_real = data_3d.tolist()          # list[list[list[float]]], shape (1,NI,NP)
        flat_imag = np.zeros((1, NI, NP)).tolist()

        # xaxArray: Hz values where hz = ppm * freq_MHz (no extra factor of 1e3)
        xax_x = (np.linspace(cd.xlims[0], cd.xlims[1], NP) * x_larmor_freq_mhz).tolist()
        xax_y = (np.linspace(cd.ylims[0], cd.ylims[1], NI) * y_freq_mhz).tolist()

        # ssNake dimension ordering follows the data array shape (1, NI, NP):
        #   index 0 → NI → y (indirect) dimension
        #   index 1 → NP → x (direct)   dimension
        # So all per-dimension lists must be [y, x], not [x, y].
        struct = {
            'dataReal':  flat_real,
            'dataImag':  flat_imag,
            'hyper':     [0],
            'freq':      [y_freq_hz, x_freq_hz],
            'sw':        [SW_y_hz, SW_x_hz],
            'spec':      [1, 1],
            'wholeEcho': [0, 0],
            'ref':       [ref_y, ref_x],
            'xaxArray':  [xax_y, xax_x],
            'history':   ['Exported by Soprano NMRData2D.export_contour_data'],
            'metaData':  {
                'x_larmor_MHz': x_larmor_freq_mhz,
                'y_larmor_MHz': y_freq_mhz,
                'x_broadening_ppm': cd.x_broadening,
                'y_broadening_ppm': cd.y_broadening,
                'broadening_type': cd.broadening_type,
            },
        }
        with open(path, 'w') as f:
            json.dump(struct, f)
        self.logger.info(f"ssNake JSON written to '{path}' "
                         f"(x={x_larmor_freq_mhz} MHz, y={y_freq_mhz} MHz).")


# ============================================================================
# Plot Backend Classes
# ============================================================================

def _resolve_levels(
    Z: np.ndarray,
    levels: Union[int, Iterable[float]],
    contour_range: Tuple[float, float],
) -> np.ndarray:
    """Return concrete contour level values from either a count or explicit list.

    Parameters
    ----------
    Z : np.ndarray
        The intensity grid (used only when *levels* is an integer).
    levels : int or iterable of float
        *int* – generate this many evenly-spaced levels inside *contour_range*.
        *iterable* – used directly as absolute intensity values;
        *contour_range* is then ignored.
    contour_range : (float, float)
        ``(lo, hi)`` expressed as **percentages of Z.max()** (0–100 scale),
        applied only when *levels* is an integer.

    Returns
    -------
    np.ndarray
        1-D array of level values.
    """
    if isinstance(levels, (int, float)):
        z_max = float(Z.max())
        lo = contour_range[0] / 100.0 * z_max
        hi = contour_range[1] / 100.0 * z_max
        return np.linspace(lo, hi, int(levels))
    else:
        return np.asarray(levels)


class PlotBackend(ABC):
    """Abstract base class for plot backends"""
    
    @abstractmethod
    def create_figure(self):
        """Create a new figure/chart object"""
        pass
    
    @abstractmethod
    def plot_markers(self, x: np.ndarray, y: np.ndarray, sizes: np.ndarray, 
                    colors: Union[str, list], settings: 'PlotSettings',
                    correlation_info: Optional[dict] = None,
                    xlabels: Optional[list] = None,
                    ylabels: Optional[list] = None,
                    correlation_values: Optional[np.ndarray] = None) -> Any:
        """Plot scatter markers
        
        Args:
            x: x coordinates
            y: y coordinates
            sizes: marker sizes (normalized for plotting)
            colors: marker colors (single color or list)
            settings: PlotSettings object
            correlation_info: Optional dict with correlation metadata for legend
            xlabels: Optional list of x-axis labels for hover text
            ylabels: Optional list of y-axis labels for hover text
            correlation_values: Optional array of actual correlation values (unnormalized)
        """
        pass
    
    @abstractmethod
    def plot_heatmap(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                     settings: 'PlotSettings') -> Any:
        """Plot heatmap contour fill"""
        pass
    
    @abstractmethod
    def plot_contour(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                    settings: 'PlotSettings') -> Any:
        """Plot contour lines"""
        pass
    
    @abstractmethod
    def plot_connectors(self, x: np.ndarray, y: np.ndarray, 
                       settings: 'PlotSettings') -> Any:
        """Plot connecting lines between points"""
        pass
    
    @abstractmethod
    def plot_axlines(self, x: np.ndarray, y: np.ndarray, 
                    settings: 'PlotSettings') -> Any:
        """Plot reference lines at peak positions"""
        pass
    
    @abstractmethod
    def plot_diagonal(self, settings: 'PlotSettings') -> Any:
        """Plot diagonal line"""
        pass
    
    @abstractmethod
    def plot_annotations(self, x: np.ndarray, y: np.ndarray, 
                        xlabels: list, ylabels: list, 
                        settings: 'PlotSettings') -> Any:
        """Plot annotations/labels"""
        pass
    
    @abstractmethod
    def set_axis_properties(self, xlabel: str, ylabel: str, 
                          xlim: Optional[Tuple[float, float]], 
                          ylim: Optional[Tuple[float, float]], 
                          invert_axes: bool) -> None:
        """Set axis labels, limits, and inversions"""
        pass
    
    @abstractmethod
    def finalize(self, filename: Optional[str] = None) -> Any:
        """Finalize and return the plot object"""
        pass


class MatplotlibBackend(PlotBackend):
    """Matplotlib backend implementation (preserves original functionality)"""
    
    def __init__(self, ax: Optional[Axes] = None):
        """Initialize with optional existing axis"""
        if ax is None:
            self.fig, self.ax = plt.subplots()
        elif isinstance(ax, Axes):
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            raise TypeError("ax must be an Axes object or None.")
        
        self.logger = logging.getLogger(__name__)
    
    def create_figure(self):
        """Figure already created in __init__"""
        return self.fig, self.ax
    
    def plot_markers(self, x, y, sizes, colors, settings, correlation_info=None, xlabels=None, ylabels=None, correlation_values=None):
        """Plot scatter markers using matplotlib"""
        scatter = self.ax.scatter(
            x, y, s=sizes, c=colors,
            marker=settings.marker,
            linewidths=settings.marker_linewidth,
            zorder=10
        )
        
        # Add legend if requested
        if settings.show_legend and correlation_info:
            kw = dict(
                prop="sizes", 
                num=correlation_info.get('num_legend_elements', 5),
                color=colors if isinstance(colors, str) else colors[0],
                fmt=correlation_info.get('fmt', '{x:.1f}') + f" {correlation_info.get('unit', '')}",
                func=lambda s: s * correlation_info.get('max_size', 1) / settings.max_marker_size
            )
            handles, labels = scatter.legend_elements(**kw)
            self.ax.legend(
                handles, labels,
                title=correlation_info.get('label', 'Correlation'),
                fancybox=True,
                framealpha=0.8
            ).set_zorder(12)
        
        return scatter
    
    def plot_heatmap(self, X, Y, Z, settings):
        """Plot heatmap using matplotlib contourf"""
        levels = _resolve_levels(Z, settings.heatmap_levels, settings.contour_range)
        return self.ax.contourf(X, Y, Z, cmap=settings.colormap,
                               zorder=-1, levels=levels)

    def plot_contour(self, X, Y, Z, settings):
        """Plot contour lines using matplotlib"""
        levels = _resolve_levels(Z, settings.contour_levels, settings.contour_range)
        return self.ax.contour(
            X, Y, Z,
            colors=settings.contour_color,
            linewidths=settings.contour_linewidth,
            levels=levels
        )
    
    def plot_connectors(self, x, y, settings):
        """Plot connecting lines between points with same y value"""
        y_order = np.argsort(y)
        for i, idx in enumerate(y_order):
            if i > 0 and np.isclose(y[idx], y[y_order[i-1]], atol=1e-6):
                self.ax.plot(
                    [x[idx], x[y_order[i-1]]],
                    [y[idx], y[y_order[i-1]]],
                    c='0.25', lw=0.75, ls='-', zorder=1
                )
    
    def plot_axlines(self, x, y, settings):
        """Plot reference lines at peak positions"""
        xticks = np.unique(np.round(x, 6))
        yticks = np.unique(np.round(y, 6))
        
        for x_val in xticks:
            self.ax.axvline(x_val, zorder=0)
        for y_val in yticks:
            self.ax.axhline(y_val, zorder=0)
    
    def plot_diagonal(self, settings):
        """Plot diagonal line.

        For 2Q (DQ/SQ) mode the diagonal marks the auto-correlation condition
        DQ = 2 × SQ, i.e. y = 2x.  For all other modes the conventional
        y = x identity line is drawn.
        """
        xlims = self.ax.get_xlim()
        if getattr(settings, 'yaxis_order', None) == '2Q':
            # DQ/SQ diagonal: y = 2x
            y_vals = [2 * xlims[0], 2 * xlims[1]]
        else:
            y_vals = list(self.ax.get_ylim())
        self.ax.plot(xlims, y_vals, ls='--', c='k', lw=1, alpha=0.2)
    
    def plot_annotations(self, x, y, xlabels, ylabels, settings):
        """Plot annotations with arrows (matplotlib approach)"""
        font_size = settings.label_fontsize
        if font_size is None:
            font_size = self.ax.xaxis.label.get_fontsize() * ANNOTATION_FONT_SCALE
        
        # Get unique labels and positions
        xlabels_unique, xidx = np.unique(xlabels, return_index=True)
        ylabels_unique, yidx = np.unique(ylabels, return_index=True)
        xpos = x[xidx]
        ypos = y[yidx]
        
        labels_offset = 0.10
        armA = 15 if settings.plot_filename is None else (3 if settings.plot_filename.endswith('.pdf') else 20)
        armB = 15 if settings.plot_filename is None else (5 if settings.plot_filename.endswith('.pdf') else 30)
        
        annotations = []
        
        # X labels at top
        texts = []
        for i, xlabel in enumerate(xlabels_unique):
            an = self.ax.annotate(
                xlabel,
                xy=(xpos[i], 1.0),
                xycoords=('data', 'axes fraction'),
                xytext=(xpos[i], 1+labels_offset),
                textcoords=('data', 'axes fraction'),
                fontsize=font_size,
                ha='center', va='bottom',
                rotation=90,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc,angleA=-90,armA={armA},angleB=90,armB={armB},rad=0",
                    relpos=(0.5, 0.0),
                    lw=ANNOTATION_LINE_WIDTH,
                    shrinkA=0.0, shrinkB=0.0,
                ),
            )
            texts.append(an)
        
        if settings.auto_adjust_labels:
            adjust_text(
                texts, ensure_inside_axes=False, avoid_self=False,
                force_pull=(0.0, 0.0), force_text=(0.3, 0.0),
                force_explode=(1.5, 0.0), expand=(1.3, 1.0), max_move=2,
            )
        annotations.extend(texts)
        
        # Y labels at right
        texts = []
        for i, ylabel in enumerate(ylabels_unique):
            an = self.ax.annotate(
                ylabel,
                xy=(1.0, ypos[i]),
                xycoords=('axes fraction', 'data'),
                xytext=(1+labels_offset, ypos[i]),
                textcoords=('axes fraction', 'data'),
                fontsize=font_size,
                ha='left', va='center',
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc,angleA=180,armA={armA},angleB=0,armB={armB},rad=0",
                    relpos=(0.0, 0.5),
                    lw=ANNOTATION_LINE_WIDTH,
                    shrinkA=0.0, shrinkB=0.0,
                ),
            )
            texts.append(an)
        
        if settings.auto_adjust_labels:
            adjust_text(
                texts, ensure_inside_axes=False, avoid_self=False,
                force_pull=(0.0, 0.0), force_text=(0.4, 0.8),
                force_explode=(0.0, 1.2), expand=(1.0, 1.8), max_move=1,
            )
        annotations.extend(texts)
        
        return annotations
    
    def set_axis_properties(self, xlabel, ylabel, xlim, ylim, invert_axes):
        """Set axis properties"""
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        if xlim:
            self.ax.set_xlim(min(xlim), max(xlim))
        if ylim:
            self.ax.set_ylim(min(ylim), max(ylim))
        
        if invert_axes:
            self.ax.invert_xaxis()
            self.ax.invert_yaxis()
    
    def finalize(self, filename=None):
        """Finalize the plot"""
        self.fig.tight_layout()
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        return self.fig, self.ax


class PlotlyBackend(PlotBackend):
    """Plotly backend implementation for interactive web-based plots with full contour support"""
    
    def __init__(self):
        """Initialize Plotly backend"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyBackend. Install with: pip install plotly")
        
        self.xlabel = ""
        self.ylabel = ""
        self.xlim = None
        self.ylim = None
        self.invert_axes = False
        self.logger = logging.getLogger(__name__)
        # Create the figure immediately
        self.fig = go.Figure()
    
    def create_figure(self):
        """Create a Plotly figure"""
        if self.fig is None:
            self.fig = go.Figure()
        return self.fig
    
    def plot_markers(self, x, y, sizes, colors, settings, correlation_info=None, xlabels=None, ylabels=None, correlation_values=None):
        """Plot scatter markers using Plotly"""
        # Handle colors
        if isinstance(colors, str):
            marker_colors = colors
        else:
            marker_colors = colors
        
        # Map matplotlib marker to Plotly symbol (using -open versions for hollow markers)
        symbol = MPL_TO_PLOTLY_MARKER.get(settings.marker, 'circle-open')
        
        # Normalize sizes for Plotly (scale to reasonable pixel values)
        size_scale = settings.max_marker_size / np.max(sizes) if np.max(sizes) > 0 else 1
        plotly_sizes = sizes * size_scale
        
        # Use actual correlation values if provided, otherwise fall back to sizes
        values_to_display = correlation_values if correlation_values is not None else sizes
        
        # Get format string and unit from correlation_info
        if correlation_info:
            fmt = correlation_info.get('fmt', '{x:.2f}')
            unit = correlation_info.get('unit', '')
            label = correlation_info.get('label', 'Strength')
        else:
            fmt = '{x:.2f}'
            unit = ''
            label = 'Strength'
        
        # Create hover text with labels if available
        if xlabels is not None and ylabels is not None:
            hovertext = [f"{xl}--{yl}<br>x: {xi:.2f}<br>y: {yi:.2f}<br>{label}: {fmt.format(x=vi)} {unit}" 
                         for xl, yl, xi, yi, vi in zip(xlabels, ylabels, x, y, values_to_display)]
        else:
            hovertext = [f"x: {xi:.2f}<br>y: {yi:.2f}<br>{label}: {fmt.format(x=vi)} {unit}" 
                         for xi, yi, vi in zip(x, y, values_to_display)]
        
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=plotly_sizes,
                color='rgba(0,0,0,0)',  # Transparent fill for hollow markers
                symbol=symbol,
                line=dict(width=settings.marker_linewidth, color=marker_colors)
            ),
            hovertext=hovertext,
            hoverinfo='text',
            showlegend=settings.show_legend,
            name=correlation_info.get('label', 'Correlation') if correlation_info else 'Peaks'
        )
        
        self.fig.add_trace(trace)
        return trace
    
    def plot_heatmap(self, X, Y, Z, settings):
        """Plot heatmap using Plotly"""
        colorscale = MPL_TO_PLOTLY_COLORMAP.get(settings.colormap, settings.colormap)
        levels = _resolve_levels(Z, settings.heatmap_levels, settings.contour_range)

        trace = go.Heatmap(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale=colorscale,
            zmin=float(levels[0]),
            zmax=float(levels[-1]),
            showscale=False,
            hoverinfo='skip'
        )

        # Insert as first trace (background)
        self.fig.add_trace(trace)
        # Move to back
        self.fig.data = (self.fig.data[-1],) + self.fig.data[:-1]
        return trace

    def plot_contour(self, X, Y, Z, settings):
        """Plot contour lines using Plotly"""
        colorscale = MPL_TO_PLOTLY_COLORMAP.get(settings.colormap, settings.colormap)
        levels = _resolve_levels(Z, settings.contour_levels, settings.contour_range)
        n = len(levels)
        size = float(levels[-1] - levels[0]) / (n - 1) if n > 1 else 0.0

        trace = go.Contour(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale=colorscale,
            showscale=False,
            contours=dict(
                start=float(levels[0]),
                end=float(levels[-1]),
                size=size,
                coloring='lines',
                showlabels=True,
                labelfont=dict(size=8)
            ),
            line=dict(width=settings.contour_linewidth),
            hoverinfo='x+y+z'
        )

        self.fig.add_trace(trace)
        return trace
    
    def plot_connectors(self, x, y, settings):
        """Plot connecting lines between points with same y value"""
        y_order = np.argsort(y)
        
        for i, idx in enumerate(y_order):
            if i > 0 and np.isclose(y[idx], y[y_order[i-1]], atol=1e-6):
                trace = go.Scatter(
                    x=[x[y_order[i-1]], x[idx]],
                    y=[y[y_order[i-1]], y[idx]],
                    mode='lines',
                    line=dict(color='gray', width=0.75),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                )
                self.fig.add_trace(trace)
    
    def plot_axlines(self, x, y, settings):
        """Plot reference lines at peak positions"""
        xticks = np.unique(np.round(x, 6))
        yticks = np.unique(np.round(y, 6))
        
        # Add vertical lines
        for xt in xticks:
            self.fig.add_vline(
                x=xt,
                line=dict(color='lightgray', width=0.5),
                opacity=0.3
            )
        
        # Add horizontal lines
        for yt in yticks:
            self.fig.add_hline(
                y=yt,
                line=dict(color='lightgray', width=0.5),
                opacity=0.3
            )
    
    def plot_diagonal(self, settings):
        """Plot diagonal line.

        For 2Q (DQ/SQ) mode the diagonal marks the auto-correlation condition
        DQ = 2 × SQ, i.e. y = 2x.  For all other modes the conventional
        y = x identity line is drawn.
        """
        if self.xlim:
            x_vals = [self.xlim[0], self.xlim[1]]
            if getattr(settings, 'yaxis_order', None) == '2Q':
                # DQ/SQ diagonal: y = 2x
                y_vals = [2 * self.xlim[0], 2 * self.xlim[1]]
            elif self.ylim:
                y_vals = [self.ylim[0], self.ylim[1]]
            else:
                y_vals = x_vals
            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                opacity=0.2,
                showlegend=False,
                hoverinfo='skip'
            )
            self.fig.add_trace(trace)
            return trace
        
        return None
    
    def plot_annotations(self, x, y, xlabels, ylabels, settings):
        """Plot text labels as annotations"""
        # Get unique labels
        xlabels_unique, xidx = np.unique(xlabels, return_index=True)
        ylabels_unique, yidx = np.unique(ylabels, return_index=True)
        xpos = x[xidx]
        ypos = y[yidx]
        
        font_size = settings.label_fontsize or 10
        
        # X labels (top)
        y_top = self.ylim[1] if self.ylim else max(y)
        for xp, label in zip(xpos, xlabels_unique):
            self.fig.add_annotation(
                x=xp,
                y=y_top,
                text=label,
                showarrow=False,
                textangle=270,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=font_size)
            )
        
        # Y labels (right)
        x_right = self.xlim[1] if self.xlim else max(x)
        for yp, label in zip(ypos, ylabels_unique):
            self.fig.add_annotation(
                x=x_right,
                y=yp,
                text=label,
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=font_size)
            )
    
    def set_axis_properties(self, xlabel, ylabel, xlim, ylim, invert_axes):
        """Store axis properties for later application"""
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.invert_axes = invert_axes
    
    def finalize(self, filename=None):
        """Apply final layout settings and return figure"""
        if self.fig is None:
            raise ValueError("No figure to finalize")
        
        # Determine axis ranges
        xrange = None
        yrange = None
        x_autorange = True
        y_autorange = True
        
        if self.xlim:
            xrange = [self.xlim[1], self.xlim[0]] if self.invert_axes else list(self.xlim)
            x_autorange = False
        
        if self.ylim:
            yrange = [self.ylim[1], self.ylim[0]] if self.invert_axes else list(self.ylim)
            y_autorange = False
        
        # Set autorange to 'reversed' when invert_axes is True and limits are auto
        x_autorange_setting = 'reversed' if (self.invert_axes and x_autorange) else x_autorange
        y_autorange_setting = 'reversed' if (self.invert_axes and y_autorange) else y_autorange
        
        # Update layout
        self.fig.update_layout(
            xaxis=dict(
                title=self.xlabel,
                range=xrange,
                autorange=x_autorange_setting,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                title=self.ylabel,
                range=yrange,
                autorange=y_autorange_setting,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            width=700,
            height=600,
            hovermode='closest',
            plot_bgcolor='white',
            showlegend=True
        )
        
        # Save if filename provided
        if filename:
            if filename.endswith('.html'):
                self.fig.write_html(filename)
            elif filename.endswith('.json'):
                self.fig.write_json(filename)
            elif filename.endswith('.png'):
                self.fig.write_image(filename)
            elif filename.endswith('.svg'):
                self.fig.write_image(filename)
            elif filename.endswith('.pdf'):
                self.fig.write_image(filename)
            else:
                self.logger.warning(f"Unsupported file format: {filename}")
        
        return self.fig


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
    # Number of filled heatmap levels (int) or explicit absolute intensity values (list).
    # When an int, levels are spaced within the percentage range given by contour_range.
    heatmap_levels: Union[int, Iterable[float]] = 20
    show_contour: bool = False
    x_broadening: Optional[float] = None
    y_broadening: Optional[float] = None
    broadening_type: str = 'lorentzian' # 'gaussian', 'lorentzian'
    heatmap_grid_size: Optional[int] = None # default to finer grid for lorentzian broadening
    colormap: str = 'bone_r'
    contour_color: str = 'C1'
    contour_linewidth: float = 0.2
    # Intensity range for contour/heatmap rendering expressed as (lo, hi) percentages
    # of the maximum grid intensity (0–100 scale).  Default (10, 100) matches ssNake.
    # Ignored when contour_levels / heatmap_levels is an explicit list of absolute values.
    contour_range: Tuple[float, float] = (10.0, 100.0)
    # Number of contour lines (int, spaced within contour_range) or explicit absolute
    # intensity values (list, contour_range is then ignored).
    contour_levels: Union[Iterable[float], int] = 10
    # When False, all markers are drawn at max_marker_size regardless of correlation strength.
    # The correlation strength still affects the heatmap/contour intensity.
    scale_markers: bool = True
    # Mirror of NMRData2D.yaxis_order so backends can adapt the diagonal line.
    yaxis_order: Optional[str] = None

    # Set the default heatmap grid size based on broadening type if not explicitly provided
    def __post_init__(self):
        if self.heatmap_grid_size is None:
            if self.broadening_type == 'lorentzian':
                self.heatmap_grid_size = 600  # Finer grid for sharper Lorentzian peaks
            else:
                self.heatmap_grid_size = 150  # Default grid size for Gaussian or no broadening

class NMRPlot2D:
    '''
    Class to plot 2D NMR data with pluggable backends.
    
    Parameters
    ----------
    nmr_data : NMRData2D
        The NMR data to plot
    plot_settings : Optional[PlotSettings]
        Plot settings to use. If None, defaults are used.
    backend : str
        Backend to use for plotting. Options: 'matplotlib' (default), 'plotly'
    ax : Optional[Axes]
        For matplotlib backend: existing axis to plot on. If None, creates new figure.
    '''
    def __init__(self,
                nmr_data: NMRData2D,
                plot_settings: Optional[PlotSettings] = None,
                backend: str = 'matplotlib',
                ax: Optional[Axes] = None):

        self.nmr_data = nmr_data
        self.backend_name = backend
        
        # store the data as numpy arrays for plotting
        npeaks = len(self.nmr_data.peaks)
        self.x = np.zeros(npeaks)
        self.y = np.zeros(npeaks)
        self.sizes = np.zeros(npeaks)

        for i, peak in enumerate(self.nmr_data.peaks):
            self.x[i] = peak.x
            self.y[i] = peak.y
            self.sizes[i] = peak.correlation_strength * peak.multiplicity


        # Use default plot settings if none are provided
        if plot_settings is None:
            plot_settings = PlotSettings()
        # Let backends know about yaxis_order so they can draw the correct diagonal
        if plot_settings.yaxis_order is None:
            plot_settings.yaxis_order = nmr_data.yaxis_order
        self.plot_settings = plot_settings

        self._initialize_plot_settings()

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        # If not set, set number of legend elements to
        # minimum of number of peaks and 5
        if self.plot_settings.num_legend_elements is None:
            self.plot_settings.num_legend_elements = min(npeaks, DEFAULT_MAX_NUM_LEGEND_ELEMENTS)
        
        # Initialize the appropriate backend
        if backend == 'matplotlib':
            self.backend = MatplotlibBackend(ax=ax)
        elif backend == 'plotly':
            if ax is not None:
                self.logger.warning("ax parameter is ignored for Plotly backend")
            self.backend = PlotlyBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'matplotlib' or 'plotly'.")

    def _initialize_plot_settings(self):
        for key, value in self.plot_settings.__dict__.items():
            setattr(self, key, value)

    def plot(self):
        '''
        Plot the 2D NMR data using the configured backend.

        Returns
        -------
        For matplotlib backend:
            fig : Figure
                The figure object
            ax : Axes
                The axis object
        
        For plotly backend:
            fig : go.Figure
                The Plotly figure object
        '''
        
        # For matplotlib, we need to apply the styled_plot decorator
        if self.backend_name == 'matplotlib':
            return self._plot_matplotlib()
        else:
            return self._plot_generic()
    
    @styled_plot(nmr_base_style, nmr_2D_style)
    def _plot_matplotlib(self):
        """Plot using matplotlib backend with styling"""
        return self._plot_generic()
    
    def _plot_generic(self):
        """Generic plotting logic that works with any backend"""
        
        # Prepare axis labels
        x_axis_label = self.plot_settings.x_axis_label if self.plot_settings.x_axis_label else self.nmr_data.x_axis_label
        y_axis_label = self.plot_settings.y_axis_label if self.plot_settings.y_axis_label else self.nmr_data.y_axis_label
        
        # Normalize xlim and ylim
        if self.plot_settings.xlim:
            xlim = self.plot_settings.xlim
            self.plot_settings.xlim = (min(xlim), max(xlim))
        
        if self.plot_settings.ylim:
            ylim = self.plot_settings.ylim
            self.plot_settings.ylim = (min(ylim), max(ylim))

        # Auto-compute axis limits from peak positions when not user-specified.
        # This MUST happen before drawing the contour/heatmap: the contour grid
        # is padded by up to 50× the broadening beyond the outermost peak, so if
        # we don't set explicit limits first, matplotlib autoscales to the grid
        # extent and the visible area ends up far wider than the peak range.
        # Buffer = 2× the FWHM broadening so the outermost line shape is fully
        # visible with a little breathing room.
        if not self.plot_settings.xlim and len(self.x):
            x_buf = (self.plot_settings.x_broadening or 1.0) * 2
            self.plot_settings.xlim = (
                float(self.x.min()) - x_buf,
                float(self.x.max()) + x_buf,
            )

        if not self.plot_settings.ylim and len(self.y):
            y_buf = (self.plot_settings.y_broadening or 1.0) * 2
            self.plot_settings.ylim = (
                float(self.y.min()) - y_buf,
                float(self.y.max()) + y_buf,
            )

        # Set axis properties first (backends may need this for subsequent operations)
        self.backend.set_axis_properties(
            x_axis_label, y_axis_label,
            self.plot_settings.xlim, self.plot_settings.ylim,
            self.nmr_data.is_shift
        )
        
        # Plot heatmap and contour first (background layers)
        if self.plot_settings.show_heatmap or self.plot_settings.show_contour:
            X, Y, Z = self._get_contour_data_for_backend()
            
            if self.plot_settings.show_heatmap:
                self.backend.plot_heatmap(X, Y, Z, self.plot_settings)
            
            if self.plot_settings.show_contour:
                self.backend.plot_contour(X, Y, Z, self.plot_settings)
        
        # Plot reference lines at peak locations
        if self.plot_settings.show_lines:
            self.backend.plot_axlines(self.x, self.y, self.plot_settings)
        
        # Plot diagonal line for homo-nuclear spectra
        xelem_same_as_yelem = (self.nmr_data.xelement == self.nmr_data.yelement and 
                               self.nmr_data.xelement is not None)
        if xelem_same_as_yelem and self.plot_settings.show_diagonal:
            self.backend.plot_diagonal(self.plot_settings)
        
        # Plot connectors between peaks
        if (self.plot_settings.show_connectors and 
            self.nmr_data.yaxis_order == '2Q' and 
            self.nmr_data.xelement == self.nmr_data.yelement):
            self.backend.plot_connectors(self.x, self.y, self.plot_settings)
        
        # Plot scatter markers
        if self.plot_settings.show_markers:
            colors = self._get_marker_colors()
            if self.plot_settings.scale_markers:
                normalized_sizes = self._normalize_marker_sizes(self.sizes)
            else:
                normalized_sizes = np.full(len(self.sizes), self.plot_settings.max_marker_size)
            
            # Prepare correlation info for legend
            correlation_info = {
                'label': self.nmr_data.correlation_label,
                'unit': self.nmr_data.correlation_unit,
                'fmt': self.nmr_data.correlation_fmt,
                'max_size': np.abs(self.sizes).max(),
                'num_legend_elements': self.plot_settings.num_legend_elements
            }
            
            # Extract labels for hover text
            xlabels = [peak.xlabel for peak in self.nmr_data.peaks]
            ylabels = [peak.ylabel for peak in self.nmr_data.peaks]
            
            self.backend.plot_markers(
                self.x, self.y, normalized_sizes, colors,
                self.plot_settings, correlation_info,
                xlabels, ylabels,
                correlation_values=self.sizes  # Pass actual correlation values
            )
        
        # Plot site annotations/labels
        if self.plot_settings.show_labels:
            xlabels = [peak.xlabel for peak in self.nmr_data.peaks]
            ylabels = [peak.ylabel for peak in self.nmr_data.peaks]
            self.backend.plot_annotations(self.x, self.y, xlabels, ylabels, self.plot_settings)
        
        # Finalize and return
        return self.backend.finalize(self.plot_settings.plot_filename)
    
    def _get_marker_colors(self):
        """Get marker colors from peaks or use settings"""
        if self.plot_settings.marker_color is None:
            colors = [peak.color for peak in self.nmr_data.peaks]
            # If all colors are the same, use single color
            if len(set(colors)) == 1:
                colors = colors[0]
            return colors
        else:
            return self.plot_settings.marker_color
    
    def _get_contour_data_for_backend(self):
        """Delegate contour generation to NMRData2D.get_contour_data().

        Grid limits come from ``PlotSettings.xlim`` / ``PlotSettings.ylim``
        when set explicitly, or are auto-computed from the peak positions when
        those are *None*.

        Note: we intentionally do NOT read the live matplotlib axis limits
        here.  The contour is drawn as the first (background) layer, before
        any markers or other data are plotted, so the axis has not been
        auto-scaled yet.  Reading it would always return matplotlib's default
        (0, 1) initialisation, producing a completely wrong grid range.
        """
        xlims = self.plot_settings.xlim
        ylims = self.plot_settings.ylim

        cd = self.nmr_data.get_contour_data(
            x_broadening=self.plot_settings.x_broadening,
            y_broadening=self.plot_settings.y_broadening,
            broadening_type=self.plot_settings.broadening_type,
            grid_size=self.plot_settings.heatmap_grid_size,
            xlims=xlims,
            ylims=ylims,
        )
        return cd.X, cd.Y, cd.Z


    def _normalize_marker_sizes(self, sizes):
        """Normalize marker sizes for consistent display"""
        sizes = np.abs(sizes)
        marker_size_range = np.max(sizes) - np.min(sizes)
        self.logger.info(f"Marker size range: {marker_size_range} {self.nmr_data.correlation_unit}")
        max_abs_marker = np.max(sizes)
        # Normalize such that max marker size is self.plot_settings.max_marker_size
        return sizes / max_abs_marker * self.plot_settings.max_marker_size
