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
import json
import pkgutil
import numpy as np
from ase import Atoms
from scipy import constants as cnst
from collections import namedtuple
from soprano.properties.nmr.utils import _get_nmr_data, _el_iso
from soprano.calculate.nmr.powder import gen_pwd_ang, pwd_avg

_nmr_data = _get_nmr_data()

# Conversion functions to Tesla
# (they take element and isotope as arguments)
_larm_units = {
    'MHz': lambda e, i: 2*np.pi*1.0e6/_nmr_data[e][i]['gamma'],
    'T': lambda e, i: 1.0,
}

# Conversion factor: Vzz*Q to frequency(Hz)
_VzzQ_Hz = 1.0e-31*(cnst.m_e**3*cnst.c**4*cnst.alpha**4)\
    / (cnst.hbar**3*2*np.pi)

# Function used for second-order quadrupolar shift
# Arguments: cos(2*alpha), eta


def _st_A(c2a, eta):
    return -27.0/8.0-9.0/4.0*eta*c2a-3.0/8.0*eta**2*c2a**2


def _st_B(c2a, eta):
    return 15.0/4.0-0.5*eta**2+2*eta*c2a+0.75*eta**2*c2a**2


def _st_C(c2a, eta):
    return -23.0/40.0+14.0/15.0*eta**2+0.25*eta*c2a-3.0/8.0*eta**2*c2a**2


def _mas_A(c2a, eta):
    return 21.0/16.0-7.0/7.0*eta*c2a+7.0/48.0*eta**2*c2a**2


def _mas_B(c2a, eta):
    return -9.0/8.0+1.0/12.0*eta**2+eta*c2a-7.0/24.0*eta**2*c2a**2


def _mas_C(c2a, eta):
    return 9.0/80.0-1.0/15.0*eta**2-0.125*eta*c2a+7.0/48.0*eta**2*c2a**2


def _gfunc(ca, cb, eta, A, B, C):
    c2a = 2*ca**2-1
    return A(c2a, eta)*cb**4+B(c2a, eta)*cb**2+C(c2a, eta)

# Flags for what to include in spectra
NMRFlags = namedtuple('NMRFlags',
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
                      MAS"""
                      )
NMRFlags = NMRFlags(
    CS_ISO=1,
    CS_ORIENT=2,
    CS=1+2,
    Q_1_ORIENT=4,
    Q_2_SHIFT=8,
    Q_2_ORIENT_STATIC=16,
    Q_2_ORIENT_MAS=32,
    Q_2_STATIC=8+16,
    Q_2_MAS=8+32,
    Q_STATIC=4+8+16,
    Q_MAS=8+32,
    STATIC=1+2+4+8+16,
    MAS=1+8+32)


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

    def __init__(self, sample, larmor_frequency=400,
                 larmor_units='MHz'):

        if not isinstance(sample, Atoms):
            raise TypeError('sample must be an ase.Atoms object')

        self._sample = sample

        # Define isotope array
        self._elems = np.array(self._sample.get_chemical_symbols())
        if self._sample.has('isotopes'):
            isos = self._sample.get_array('isotopes')
        else:
            isos = [None]*len(self._sample)
        self.set_isotopes(isos)

        self.set_larmor_frequency(larmor_frequency, larmor_units)

        self._references = {}

    def set_larmor_frequency(self, larmor_frequency=400, larmor_units='MHz',
                             element='1H'):
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
            raise ValueError('Invalid units for Larmor frequency')

        # Split isotope and element
        el, iso = _el_iso(element)

        self._B = larmor_frequency*_larm_units[larmor_units](el, iso)

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
        return self._B/_larm_units['MHz'](el, iso)

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
            raise ValueError('isotopes array should be as long as the atoms'
                             ' in sample')

        # Clean up the list, make sure it's all right
        iso_clean = []
        for i, iso in enumerate(isotopes):
            # Is it an integer?
            iso_name = ''
            if re.match('[0-9]+', str(iso)) is not None:    # numpy-proof test
                iso_name = str(iso)
                # Does it exist?
                if iso_name not in _nmr_data[self._elems[i]]:
                    raise ValueError('Invalid isotope '
                                     '{0} for element {1}'.format(iso_name,
                                                                  self.
                                                                  _elems[i]))
            elif iso is None:
                iso_name = str(_nmr_data[self._elems[i]]['iso'])
            elif iso == 'Q':
                iso_name = str(_nmr_data[self._elems[i]]['Q_iso'])
            else:
                el, iso_name = _el_iso(iso)
                # Additional test
                if el != self._elems[i]:
                    raise ValueError('Invalid element in isotope array - '
                                     '{0} in place of {1}'.format(el,
                                                                  self
                                                                  ._elems[i]))

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

        p = np.array([[np.sin(theta)*np.cos(phi),
                       np.sin(theta)*np.sin(phi),
                       np.cos(theta)]])
        w = np.array([1.0])
        t = np.array([])

        self._orients = [p, w, t]

    def set_powder(self, N=8, mode='hemisphere'):
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

        self._orients = gen_pwd_ang(N, mode)

    def spectrum_1d(self, element, min_freq=-50, max_freq=50, bins=100,
                    freq_broad=None, freq_units='ppm',
                    effects=NMRFlags.CS_ISO):
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
        |                       flags check the docstring for NMRCalculator.

        | Returns:
        |   spec (np.ndarray): array of length 'bins' containing the spectral
        |                      intensities
        |   freq (np.ndarray): array of length 'bins' containing the frequency
        |                      axis

        """

        # First, define the frequency range
        el, iso = _el_iso(element)
        larm = self._B*_nmr_data[el][iso]['gamma']/(2.0*np.pi*1e6)
        I = _nmr_data[el][iso]['I']
        # Units? We want this to be in ppm
        u = {
            'ppm': 1,
            'MHz': 1e6/larm,
        }
        try:
            freq_axis = np.linspace(min_freq, max_freq, bins)*u[freq_units]
        except KeyError:
            raise ValueError('Invalid freq_units passed to spectrum_1d')

        # If it's not a quadrupolar nucleus, no reason to keep those effects
        # around...
        if abs(I) < 1:
            effects &= ~NMRFlags.Q_STATIC
            effects &= ~NMRFlags.Q_MAS

        # Ok, so get the relevant atoms and their properties
        a_inds = np.where((self._elems == el) & (self._isos == iso))[0]

        # Are there even any such atoms?
        if len(a_inds) == 0:
            raise RuntimeError('No atoms of the desired isotopes found in the'
                               ' system')

        # Sanity check
        if (effects & NMRFlags.Q_2_ORIENT_STATIC and
                effects & NMRFlags.Q_2_ORIENT_MAS):
            # Makes no sense...
            raise ValueError('The flags Q_2_ORIENT_STATIC and Q_2_ORIENT_MAS'
                             ' can not be set at the same time')

        if effects & NMRFlags.CS:
            try:
                ms_tens = self._sample.get_array('ms')[a_inds]
            except KeyError:
                raise RuntimeError('Impossible to compute chemical shift - '
                                   'sample has no shielding data')
            ms_evals, ms_evecs = zip(*[np.linalg.eigh(t) for t in ms_tens])

        if effects & (NMRFlags.Q_STATIC | NMRFlags.Q_MAS):
            try:
                efg_tens = self._sample.get_array('efg')[a_inds]
            except KeyError:
                raise RuntimeError('Impossible to compute quadrupolar effects'
                                   ' - sample has no EFG data')
            efg_evals, efg_evecs = zip(*[np.linalg.eigh(t) for t in efg_tens])
            efg_i = (np.arange(len(efg_evals))[:, None],
                     np.argsort(np.abs(efg_evals), axis=1))
            efg_evals = np.array(efg_evals)[efg_i]
            efg_evecs = np.array(efg_evecs)[efg_i[0], :, efg_i[1]]
            Vzz = efg_evals[:, -1]
            eta_q = (efg_evals[:, 0]-efg_evals[:, 1])/Vzz
            Q = _nmr_data[el][iso]['Q']
            chi = Vzz*Q*_VzzQ_Hz

        # Reference (zero if not given)
        try:
            ref = self._references[el][iso]
        except KeyError:
            ref = 0.0

        # Let's start with peak positions - quantities non dependent on
        # orientation

        # Shape: atoms*1Q transitions
        peaks = np.zeros((len(a_inds), int(2*I)))
        # Magnetic quantum number values
        m = np.arange(-I, I+1).astype(float)[None, :]

        if effects & NMRFlags.CS_ISO:
            peaks += np.average(ms_evals, axis=1)[:, None]

        # Quadrupole second order
        if effects & NMRFlags.Q_2_SHIFT:
            nu_l = larm*1e6
            # NOTE: the last factor of two in this formula was inserted
            # despite not being present in M. J. Duer (5.9) as apparently
            # it's a mistake in the book. Other sources (like the quadrupolar
            # NMR online book by D. Freude and J. Haase, Dec. 2016) report
            # this formulation, with the factor of two, instead.
            q_shifts = np.diff(-(chi[:, None]/(4*I*(2*I-1)))**2*m/nu_l *
                                (-0.2*(I*(I+1)-3*m**2) *
                                    (3+eta_q[:, None]**2))*2)
            q_shifts /= larm

            peaks += q_shifts

        # Any orientational effects at all?
        has_orient = effects & (NMRFlags.CS_ORIENT | NMRFlags.Q_1_ORIENT |
                                NMRFlags.Q_2_ORIENT_STATIC |
                                NMRFlags.Q_2_ORIENT_MAS)
        # Are we using a POWDER average?
        use_pwd = len(self._orients[2]) > 0

        if has_orient:
            # Further expand the peaks!
            peaks = np.repeat(
                peaks[:, :, None], len(self._orients[0]), axis=-1)

        # Now compute the orientational quantities
        if effects & NMRFlags.CS_ORIENT:

            # Compute the traceless ms tensors
            ms_traceless = ms_tens - [np.identity(3)*np.average(ev)
                                      for ev in ms_evals]
            # Now get the shift contributions for each orientation
            dirs = self._orients[0]

            peaks += np.sum(dirs.T[None, :, :]*np.tensordot(ms_traceless,
                                                            dirs,
                                                            axes=((2), (1))),
                            axis=1)[:, None, :]

        if effects & NMRFlags.Q_1_ORIENT:

            # First order quadrupolar anisotropic effects
            # We consider the field aligned along Z
            cosb2 = self._orients[0][:, 1]**2
            sinb2 = 1.0 - cosb2
            cosa2 = self._orients[0][:, 0]**2

            dir_fac = 0.5*((3*cosb2[None, :]-1) +
                           eta_q[:, None]*sinb2[None, :]*(2*cosa2[None, :] -
                                                          1.0))
            m_fac = m[:, :-1]+0.5
            nu_q = chi*1.5/(I*(2*I-1.0))

            qfreqs = nu_q[:, None, None]*m_fac[:, :, None]*dir_fac[:, None, :]

            peaks += qfreqs/larm  # Already ppm being Hz/MHz

        if effects & (NMRFlags.Q_2_ORIENT_STATIC | NMRFlags.Q_2_ORIENT_MAS):
            # Which one?
            if effects & NMRFlags.Q_2_ORIENT_STATIC:
                ABC = [_st_A, _st_B, _st_C]
            else:
                ABC = [_mas_A, _mas_B, _mas_C]

            cosa = self._orients[0][:, 0]
            cosb = self._orients[0][:, 1]

            dir_fac = _gfunc(cosa[None, :], cosb[None, :], eta_q[:, None],
                             *ABC)

            m_fac = I*(I+1.0)-17.0/3.0*m[:, :-1]*(m[:, :-1]+1)-13.0/6.0
            nu_q = chi*1.5/(I*(2*I-1.0))

            qfreqs = -((nu_q**2/(6.0*larm*1e6))[:, None, None] *
                       m_fac[:, :, None] *
                       dir_fac[:, None, :])

            peaks += qfreqs/larm

        # Finally, the overall spectrum
        spec = np.zeros(freq_axis.shape)

        for p_nuc in peaks:
            for p_trans in p_nuc:
                if has_orient and use_pwd:
                    spec += pwd_avg(freq_axis, p_trans, self._orients[1],
                                    self._orients[2])

        if freq_broad is None and (not has_orient or not use_pwd):
            print('WARNING: no artificial broadening detected in a calculation'
                  ' without line-broadening contributions. The spectrum could '
                  'appear distorted or empty')

        if freq_broad is not None:
            if has_orient and use_pwd:
                fc = (max_freq+min_freq)/2.0
                bk = np.exp(-((freq_axis-fc)/freq_broad)**2.0)
                bk /= np.sum(bk)
                spec = np.convolve(spec, bk, mode='same')
            else:
                bpeaks = np.exp(-((freq_axis-peaks[:, :, None]) /
                                  freq_broad)**2)  # Broadened peaks
                # Normalise them BY PEAK MAXIMUM
                norm_max = np.amax(bpeaks, axis=-1, keepdims=True)
                norm_max = np.where(np.isclose(norm_max, 0), np.inf, norm_max)
                bpeaks /= norm_max
                spec = np.sum(bpeaks, axis=(0, 1) if not has_orient
                              else (0, 1, 2))

        # Normalize the spectrum to the number of nuclei
        normsum = np.sum(spec)
        if (np.isclose(normsum, 0)):
            print('WARNING: no peaks found in the given frequency range. '
                  'The spectrum will be empty')
        else:
            spec *= len(a_inds)*len(spec)/normsum

        return spec, freq_axis/u[freq_units]

    @property
    def B(self):
        """Static magnetic field, in Tesla"""
        return self._B
