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

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import numpy as np
from ase import Atoms
from scipy.special import fresnel
from collections import namedtuple
from soprano.utils import minimum_supcell, supcell_gridgen
from soprano.nmr.utils import _dip_constant
from soprano.data.nmr import _get_isotope_data, _get_nmr_data, _el_iso, EFG_TO_CHI, _get_isotope_list
from soprano.calculate.nmr.utils import optimise_annotations, Peak2D, styled_plot, nmr_base_style, nmr_2D_style
from soprano.calculate.powder.triavg import TriAvg
from soprano.selection import AtomSelection
from soprano.properties.nmr import MSIsotropy, DipolarCoupling
from soprano.properties.labeling import MagresViewLabels
from soprano.utils import has_cif_labels
import itertools
import re

import matplotlib.pyplot as plt



import logging

DEFAULT_MARKER_SIZE = 50 # Default marker size for 2D plots (max is 100)
ANNOTATION_LINE_WIDTH = 0.5 # Line width for annotation lines
ANNOTATION_FONT_SCALE = 0.5 # Scale factor for annotation font size wrt to plot font size


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

class NMRCalculator(object):

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
                        "{0} for element {1}".format(iso_name, self._elems[i])
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
                        "{0} in place of {1}".format(el, self._elems[i])
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
        use_reference=False,
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
        |                         (default is False).

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


class Plot2D:
    '''
    Class to handle the 2D plotting of NMR data.
    '''
    def __init__(self, 
                atoms:Atoms=None,
                xelement=None,
                yelement=None,
                references=None,
                gradients=None,
                peaks = None,
                pairs = None,
                markersizes = None,
                rcut = None,
                isotopes={},
                plot_shifts=None,
                include_quadrupolar=False,
                yaxis_order='1Q',
                xlim=None,
                ylim=None,
                x_axis_label=None,
                y_axis_label=None,
                marker='x',
                scale_marker_by = 'fixed',
                max_marker_size=DEFAULT_MARKER_SIZE,
                show_labels=True,
                auto_adjust_labels=True,
                show_lines=True,
                show_diagonal=True,
                show_connectors=True,
                plot_filename=None,
                marker_color = 'C0',
                show_marker_legend=False,
                logger = None,
                ax = None # If None, will create a new figure
                ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


        self.atoms = atoms
        self.xelement = xelement
        self.yelement = yelement if yelement is not None else xelement
        self.references = references
        self.gradients = gradients
        self.peaks = peaks
        self.pairs = pairs
        self.markersizes = markersizes
        self.rcut = rcut
        self.isotopes = isotopes
        self.plot_shifts = plot_shifts
        self.include_quadrupolar = include_quadrupolar
        self.yaxis_order = yaxis_order
        self.xlim = xlim
        self.ylim = ylim
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.marker = marker
        self.scale_marker_by = scale_marker_by
        self.max_marker_size = max_marker_size
        self.show_labels = show_labels
        self.auto_adjust_labels = auto_adjust_labels
        self.show_lines = show_lines
        self.show_diagonal = show_diagonal
        self.show_connectors = show_connectors
        self.plot_filename = plot_filename
        self.ax = ax
        
        
        self.marker_color = marker_color
        self.show_marker_legend = show_marker_legend


        # if the user hasn't specified plot_shifts, then we 
        if plot_shifts is not None:
            # if the user has specified plot_shifts to be True or False, use that
            self.plot_shifts = plot_shifts
        else:
            # otherwise, if references are given, plot shifts
            if self.references:
                self.plot_shifts = True
                self.logger.debug("Plotting chemical shifts since references are given. ")
            else:
                self.plot_shifts = False
                self.logger.debug("Plotting chemical shielding since no references are given. ")
                
    def get_peaks(self, merge_identical=True):
        '''
        Get the correlation peaks.
        '''
        if self.peaks is None:
            if self.atoms is None:
                raise ValueError("Either atoms or peaks must be given.")
            # make sure all the data is there
            self.get_2D_plot_data()
            labels = self.get_labels()
            self.peaks = []
            for ipair, pair in enumerate(self.pairs):
                idx_x, idx_y = pair

                x = self.data[idx_x]
                y = self.data[idx_y]
                strength = self.markersizes[ipair]
                xlabel = labels[idx_x]
                ylabel = labels[idx_y]
                if self.yaxis_order == '2Q':
                    y = x + y
                    if self.xelement == self.yelement:
                        # then we might have both e.g. H1 + H2 and H2 + H1
                        # let's set them both to be H1 + H2 by sorting the labels
                        xlabel, ylabel = sorted([xlabel, ylabel])
                    ylabel  = f'{xlabel} + {ylabel}'
                peak = Peak2D(x = x, y = y, correlation_strength = strength, xlabel = xlabel, ylabel = ylabel, idx_x=idx_x, idx_y=idx_y)
                self.peaks.append(peak)
        else:
            self.logger.debug("Custom peaks provided. ")

        
        if merge_identical:
            self.peaks = self.merge_peaks(self.peaks)

        return self.peaks
    def merge_peaks(self, peaks, xtol=1e-5, ytol=1e-5, corr_tol=1e-5, ignore_correlation_strength=False):
        '''
        Merge peaks that are identical.
        '''
        # first, get the unique peaks
        unique_peaks = []
        unique_xlabels = {}
        unique_ylabels = {}
        for i, peak in enumerate(peaks):
            if i == 0:
                unique_peaks.append(peak)
                unique_xlabels[peak.xlabel] = [peak.xlabel]
                unique_ylabels[peak.ylabel] = [peak.ylabel]
            else:
                # check if it's identical to any of the unique peaks
                is_identical = False
                for unique_peak in unique_peaks:
                    if peak.equivalent_to(
                                    unique_peak,
                                    xtol=xtol,
                                    ytol=ytol,
                                    corr_tol=corr_tol,
                                    ignore_correlation_strength=ignore_correlation_strength):
                        is_identical = True
                        # update labels to include both
                        unique_xlabels[unique_peak.xlabel].append(peak.xlabel)
                        unique_ylabels[unique_peak.ylabel].append(peak.ylabel)
                        
                        break
                if not is_identical:
                    unique_peaks.append(peak)
                    unique_xlabels[peak.xlabel] = [peak.xlabel]
                    unique_ylabels[peak.ylabel] = [peak.ylabel]
        # now, update the labels
        #custom sort function 
        def sort_func(x):
            # if there are any integers, sort by those
            # otherwise, sort by the string
            # use regex to find integers
            int_list = re.findall(r'\d+', x)
            if int_list:
                return int(int_list[0])
            else:
                return x
            

        for i, unique_peak in enumerate(unique_peaks):
            xlabel_list = list(set(unique_xlabels[unique_peak.xlabel]))
            ylabel_list = list(set(unique_ylabels[unique_peak.ylabel]))
            # sort the labels
            xlabel_list.sort(key=sort_func)
            ylabel_list.sort(key=sort_func)
            # join the labels
            xlabel = '/'.join(xlabel_list)
            ylabel = '/'.join(ylabel_list)
            unique_peak.xlabel = xlabel
            unique_peak.ylabel = ylabel
        return unique_peaks

    def get_2D_plot_data(self):
        '''
        Get the data for a 2D NMR plot from 
        a dataframe with columns:
        'MS_shift/ppm' or 'MS_shielding/ppm'
        
        If include_quadrupolar is True, then the quadrupolar
        couplings should also be included in the df.
        

        '''
        if self.xelement is None:
            raise ValueError("xelement must be given.")
        all_elements = self.atoms.get_chemical_symbols()
        isotopes = _get_isotope_list(all_elements, isotopes=self.isotopes, use_q_isotopes=False)
        if self.xelement not in all_elements:
            raise ValueError(f'{self.xelement} not found in the file after the user-specified filters have been applied.')
        if self.yelement not in all_elements:
            raise ValueError(f'{self.yelement} not found in the file after the user-specified filters have been applied.')
        self.idx_x = np.array([atom.index for atom in self.atoms if atom.symbol == self.xelement])
        self.idx_y = np.array([atom.index for atom in self.atoms if atom.symbol == self.yelement])
        self.xisotope = isotopes[self.idx_x[0]]
        self.yisotope = isotopes[self.idx_y[0]]
        self.logger.debug(f'Indices of {self.xelement} in the atoms object: {self.idx_x}')
        self.logger.debug(f'Indices of {self.yelement} in the atoms object: {self.idx_y}')
        species_template =  r'$\mathrm{^{%s}{%s}}$'
        self.xspecies = species_template % (self.xisotope, self.xelement)
        self.yspecies = species_template % (self.yisotope, self.yelement)
        # log species
        self.logger.debug(f'X species: {self.xspecies}')
        self.logger.debug(f'Y species: {self.yspecies}')
        self.get_axis_labels()

        # get pairs
        self.get_plot_pairs()
        # marker sizes
        self.markersizes = self.get_marker_sizes()

        # actual data
        shieldings = MSIsotropy.get(self.atoms)
        shifts     = MSIsotropy.get(self.atoms, ref=self.references, grad=self.gradients)
        if self.plot_shifts:
            if self.xelement not in self.references:
                raise ValueError(f'{self.xelement} not found in the references dictionary. Please specify a reference for both elements.')
            if self.yelement not in self.references:
                raise ValueError(f'{self.yelement} not found in the references dictionary. Please specify a reference for both elements.')
            self.data = shifts
        else:
            self.data = shieldings

        # log the x and y values
        self.logger.debug(f'X values: {self.data[self.idx_x]}')
        self.logger.debug(f'Y values: {self.data[self.idx_y]}')


    def get_axis_labels(self):
        if self.plot_shifts:
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
    
    def get_labels(self):
        if has_cif_labels(self.atoms):
            labels = self.atoms.get_array('labels')
        elif self.atoms.has('MagresView_labels'):
            # we might have already generated the MV style labels
            labels = self.atoms.get_array('MagresView_labels')
        else:
            self.logger.info('No labels found in the atoms object. Generating MagresView-style labels from scratch.')
            labels = MagresViewLabels.get(self.atoms)
            # convert to numpy array
            labels = np.array(labels, dtype='U25')
        return labels

    def add_annotations(self, unique=True, optimise = True):
        '''
        Get the annotations for the plot
        '''
        # annotation font size is 2/3 the general font size
        font_size = self.ax.xaxis.label.get_fontsize() * ANNOTATION_FONT_SCALE
        if self.peaks is None:
            self.get_peaks()

        self.annotations = []
        xpos, ypos = np.array([[peak.x, peak.y] for peak in self.peaks]).T
        xpos_label = xpos.copy()
        ypos_label = ypos.copy()

        xlabels = [peak.xlabel for peak in self.peaks]
        ylabels = [peak.ylabel for peak in self.peaks]

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
        if optimise:
            # optimise the positions of the annotations to prevent overlap
            xpos_label = optimise_annotations(xpos_label, max_iters=5000, C = 5e-7, k = 0.15, ftol=1e-5)
            ypos_label = optimise_annotations(ypos_label, max_iters=5000, C = 5e-7, k = 0.15, ftol=1e-5)
            self.logger.debug(f'Optimised X positions: {xpos_label}')
            self.logger.debug(f'Optimised Y positions: {ypos_label}')

        
        x_left, x_right = self.ax.get_xlim()
        y_bottom, y_top = self.ax.get_ylim()
        # hack to get the arm lengths correct for pdf and interactive rendering
        if self.plot_filename is None:
            armA = 25
            armB = 15
        elif self.plot_filename.endswith('.pdf'):
            armA = 5
            armB = 3
        else:
            armA = 30
            armB = 20
            
        for i, xlabel in enumerate(xlabels):
            an = self.ax.annotate(
                xlabel,
                xy=(xpos[i], y_top),
                xytext=(xpos_label[i], 1.05),
                textcoords=('data', 'axes fraction'),
                fontsize = font_size,
                ha='center',
                va='bottom',
                rotation=90,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc,angleA=-90,armA={armA},angleB=90,armB={armB},rad=0",
                    relpos=(0.5, 0.0),
                    lw=ANNOTATION_LINE_WIDTH,
                    )
                )
            self.annotations.append(an)
        
        for i, ylabel in enumerate(ylabels):
            an = self.ax.annotate(
                ylabel,
                xy=(x_right, ypos[i]),
                xytext=(1.05, ypos_label[i]),
                textcoords=('axes fraction', 'data'),
                fontsize = font_size,
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc,angleA=180,armA={armA},angleB=0,armB={armB},rad=0",
                    relpos=(0.0, 0.5),
                    lw = ANNOTATION_LINE_WIDTH,
                    )
                )
            self.annotations.append(an)

        return self.annotations
    
    def get_plot_pairs(self):
        '''
        Get the pairs of x and y indices to plot

        self.idx_x and self.idx_y are the indices of the x and y elements in the atoms object

        This method will set the following attributes:
        self.pairs_el_idx: a list of tuples of the form (xindex, yindex)
        self.pairs: a list of tuples of the form (xindex, yindex)

        '''
        # number of x and y sites
        nx = len(self.idx_x)
        ny = len(self.idx_y)
        # xelement_indices = np.array([atom.index for atom in self.atoms if atom.symbol == self.xelement])
        # yelement_indices = np.array([atom.index for atom in self.atoms if atom.symbol == self.yelement])
        if self.pairs:
            # if the user has specified pairs, use those
            # get the indices within each element for each pair:
            self.pairs_el_idx = []
            for pair in self.pairs:
                xidx = np.where(self.idx_x == pair[0])[0][0] # there should only be one match
                yidx = np.where(self.idx_y == pair[1])[0][0] # there should only be one match
                self.pairs_el_idx.append((xidx, yidx))
            # update self.idx_x and self.idx_y to be the indices within the self.atoms object
            self.idx_x = np.array(self.pairs)[:,0]
            self.idx_y = np.array(self.pairs)[:,1]
        else:
            
            # self.pairs is a list of tuples of the form (xindex, yindex)
            # these are the indices within the self.x and self.y arrays
            self.pairs_el_idx = list(itertools.product(range(nx), range(ny)))
            # self.pairs_original_idx is a list of tuples of the form (xindex, yindex)
            # these are the indices within the self.atoms object
            self.pairs = list(itertools.product(self.idx_x, self.idx_y))

            
        # remove any pairs where the x and y indices are the same
        # this should only be the case if xelement == yelement
        # do we want to allow this if yaxis_order == '1Q'?
        # self.pairs = [pair for pair in self.pairs if pair[0] != pair[1]]
        
        # check if any two indices in a pair are the same if marker size is not fixed
        if self.scale_marker_by != 'fixed':
            for pair in self.pairs:
                if len(set(pair)) != 2:
                    raise ValueError("""
                    Two indices in a pair are the same but
                    the marker size is based on distance between sites.
                    It's unclear """)
        
        
        if len(self.pairs) == 0:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        self.pair_distances = np.zeros(len(self.pairs_el_idx))
        for i, pair in enumerate(self.pairs):
            if pair[0] == pair[1]:
                # (rather than looking for periodic images of the same atom)
                self.pair_distances[i] = 0.0
            else:
                self.pair_distances[i] = self.atoms.get_distance(*pair, mic=True)
        
        if self.rcut:
            # now filter out those pairs that are too far away
            self.logger.info(f"Filtering out pairs that are further than {self.rcut} Å apart.")
            self.logger.info(f"Number of pairs before filtering: {len(self.pairs_el_idx)}")
            
            dist_mask = np.where(self.pair_distances <= self.rcut)[0]
            self.pairs_el_idx = [self.pairs_el_idx[i] for i in dist_mask]
            self.pairs = [self.pairs[i] for i in dist_mask]
            self.pair_distances = self.pair_distances[dist_mask]
            # update the idx_x and idx_y
            self.idx_x = np.unique([pair[0] for pair in self.pairs])
            self.idx_y = np.unique([pair[1] for pair in self.pairs])
            if len(self.idx_x) == 0 or len(self.idx_y) == 0:
                raise ValueError(f'No pairs found after filtering by distance. Try increasing the cutoff distance (rcut).')
            # sort self.idx_x and self.idx_y
            self.idx_x = np.sort(self.idx_x)
            self.idx_y = np.sort(self.idx_y)
            self.logger.info(f"Number of pairs remaining: {len(self.pairs_el_idx)}")
            self.logger.debug(f"Pairs remaining: {self.pairs}")
            self.logger.debug(f"Pairs el indices remaining: {self.pairs_el_idx}")


    def get_marker_sizes(self):
        if self.markersizes:
            # if user provides a list of these, use it!
            # just check that it's the right length
            if len(self.markersizes) != len(self.pairs_el_idx):
                raise ValueError(f"Length of markersizes ({len(self.markersizes)}) does not match the number of pairs ({len(self.pairs_el_idx)}).")
            
            # set scale_marker_by to 'custom'
            self.scale_marker_by = 'custom'
        
        if self.scale_marker_by == 'fixed':
            self.logger.info("Using fixed marker size.")
            # get all unique pairs of x and y indices
            # set the marker size to be the same for all pairs
            markersizes = np.ones(len(self.pairs_el_idx))
            
        elif self.scale_marker_by == 'dipolar':
            self.logger.info("Using dipolar coupling as marker size.")
            self.logger.debug(f"Using custom isotopes: {self.isotopes}")
            # DipolarCoupling.get returns a dictionary with the dipolar coupling but we oly have one element in the dictionary.
            # We need the first element of the value of this item.
            dip = []
            for i, j in self.pairs:
                if i == j:
                    # set the dipolar coupling to zero for pairs where i == j
                    dip.append(0)
                else:
                    dip.append(list(DipolarCoupling.get(self.atoms, 
                                                        sel_i=[i],
                                                        sel_j=[j],
                                                        isotopes=self.isotopes).values())[0][0])
            # convert to kHz
            dip = np.array(dip) * 1e-3
            markersizes = np.array(dip)
        elif self.scale_marker_by == 'distance' or self.scale_marker_by == 'inversedistance':
            log_message = "Using minimum image convention {isinverse}distance as marker size."
            isinverse = ''

            # now we can use ASE get_distance to get the distances for each pair
            markersizes = self.pair_distances
            if self.scale_marker_by == 'inversedistance':
                markersizes = 1 / markersizes
                isinverse = 'inverse '
            self.logger.info(log_message.format(isinverse=isinverse))
            

        elif self.scale_marker_by == 'J':
            self.logger.info("Using J-coupling as marker size.")
            raise NotImplementedError("J-coupling scaling not implemented yet.")
        elif self.scale_marker_by == 'custom':
            self.logger.info("Using custom marker sizes.")
            markersizes = self.markersizes
        else:
            raise ValueError(f"Unknown scale_marker_by option: {self.scale_marker_by}")
        
        self.logger.debug(f"markersizes: {self.markersizes}")

        self.marker_unit = MARKER_INFO[self.scale_marker_by]['unit']
        self.marker_label = MARKER_INFO[self.scale_marker_by]['label']
        self.marker_fmt = MARKER_INFO[self.scale_marker_by]['fmt']

        #
        # log pair with smallest and largest marker size
        min_idx = np.argmin(np.abs(markersizes))
        max_idx = np.argmax(np.abs(markersizes))
        smallest_pair = self.pairs[min_idx]
        largest_pair  = self.pairs[max_idx]
        # labels for the smallest and largest pairs
        labels = self.get_labels()
        smallest_pair_labels = [labels[smallest_pair[0]], labels[smallest_pair[1]]]
        largest_pair_labels = [labels[largest_pair[0]], labels[largest_pair[1]]]
        self.logger.info(f"Pair with smallest (abs) {self.marker_label}: {smallest_pair_labels} ({markersizes[min_idx]:.2f})")
        self.logger.info(f"Pair with largest (abs) {self.marker_label}: {largest_pair_labels} ({markersizes[max_idx]:.2f})")

        return markersizes
    @styled_plot(nmr_base_style, nmr_2D_style)
    def plot(self):
        '''
        Plot a 2D NMR spectrum from a dataframe with columns 'MS_shift/ppm' or 'MS_shielding/ppm'

        '''
        self.logger.info("Plotting 2D NMR spectrum...")
        self.logger.info(f"Plotting {self.xelement} vs {self.yelement}.")

        # get the data
        peaks = self.get_peaks(merge_identical=True)

        # make the plot!
        if self.ax:
            ax = self.ax
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()
            self.fig = fig
            self.ax = ax


        if self.show_lines:
            xvals = [peak.x for peak in peaks]
            yvals = [peak.y for peak in peaks]
            # unique xvals within a tolerance
            self.xticks = np.unique(np.round(xvals, 6))
            # unique yvals within a tolerance
            self.yticks = np.unique(np.round(yvals, 6))
            # plot the lines
            for i, xval in enumerate(self.xticks):
                ax.axvline(xval, zorder=0)#, ls=linestyle, alpha= linealpha, lw=linewidth, c=linecolor)
            for i, yval in enumerate(self.yticks):
                ax.axhline(yval, zorder=0)#, ls=linestyle, alpha= linealpha, lw=linewidth, c=linecolor)
        
        # --- plot the markers ---
        xvals = [peak.x for peak in peaks]
        yvals = [peak.y for peak in peaks]
        
        if self.show_connectors and self.yaxis_order == '2Q' and self.xelement == self.yelement:
            y_order = np.argsort(yvals)
            # loop over peaks and plot lines between peaks with the same y value
            for i, idx in enumerate(y_order):
                if np.isclose(peaks[idx].y, peaks[y_order[i-1]].y, atol=1e-6):
                    ax.plot([peaks[idx].x, peaks[y_order[i-1]].x],
                            [peaks[idx].y, peaks[y_order[i-1]].y],
                            c='0.25',
                            lw=1,
                            ls='-',
                            zorder=1)
        # marker sizes based on correlation strength
        markersizes = np.array([peak.correlation_strength for peak in peaks])
        # make sure the marker sizes are all positive
        markersizes = np.abs(markersizes)
        marker_size_range = np.max(markersizes) - np.min(markersizes)
        if self.scale_marker_by != 'fixed':
            self.logger.info(f"Marker size range: {marker_size_range} {self.marker_unit}")
        max_abs_marker = np.max(markersizes)
        # normalise the marker sizes such that the maximum marker size is self.max_marker_size
        markersizes = markersizes / max_abs_marker * self.max_marker_size
        # plot the markers
        scatter = ax.scatter(
            xvals,
            yvals,
            s=markersizes,
            marker=self.marker,
            c=[peak.color for peak in peaks],
            lw=1,
            zorder=10)
        

        # --- plot the axis labels ---
        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)

        # if shifts are plotted, invert the axes
        if self.plot_shifts:
            ax.invert_xaxis()
            ax.invert_yaxis()

        # other plot options
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)

        if ((self.xelement is not None) and (self.xelement == self.yelement) and self.show_diagonal):
            # use self.xlim and self.ylim to draw a diagonal line
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
            ax.plot(xlims, ylims, ls='--', c='k', lw=1, alpha=0.2)
        
        # add marker size legend
        if (self.scale_marker_by != 'fixed') and self.show_marker_legend:
            # produce a legend with a cross-section of sizes from the scatter
            kw = dict(prop="sizes", num='auto', color=self.marker_color, 
                      fmt=self.marker_fmt + f" {self.marker_unit}",
                      func=lambda s: s*max_abs_marker / self.max_marker_size)
            handles, labels = scatter.legend_elements(**kw)
            ax.legend(handles, labels,
                      title=self.marker_label,
                      fancybox=True,
                      framealpha=0.8).set_zorder(11)
        

        if self.show_labels:
            # add the annotations to the plot
            annotations = self.add_annotations(optimise=self.auto_adjust_labels) # list of Annotation objects

        if self.plot_filename:
            self.logger.debug(f"Saving to {self.plot_filename}")
            fig.savefig(self.plot_filename)
        return fig, ax