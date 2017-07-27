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
Classes and functions for interfacing with the SIMPSON spin dynamics software.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from soprano.properties.nmr import (MSIsotropy, MSReducedAnisotropy,
                                    MSAsymmetry,
                                    MSQuaternion,
                                    EFGQuadrupolarConstant, EFGAsymmetry,
                                    EFGQuaternion,
                                    DipolarCoupling)
from soprano.selection import AtomSelection
from soprano.properties.nmr.utils import _get_nmr_data, _el_iso

_spinsys_template = """
spinsys {{
{header}
{ms}
{efg}
{dipolar}
}}
"""

_header_template = """
channels {channels}
nuclei {nuclei}
"""


def write_spinsys(s, isotope_list=None, use_ms=False, ms_iso=False,
                  q_order=0, dip_sel=None):
    """
    Write a .spinsys input file for use with SIMPSON, given the details of a
    system. This is meant to be a low-level function, used by other higher
    level interfaces in NMRCalculator.

    | Args:
    |   s (ase.Atoms): atomic structure containing the desired spins. All
    |                  atoms will be included - if that is not the desired
    |                  result, this should be accomplished by making this a
    |                  subset of the full structure.
    |   isotope_list ([int]): list of isotopes for each element in the system.
    |                         If left to None, default NMR-active isotopes
    |                         will be used.
    |   use_ms (bool): if True, include shift interactions from magnetic
    |                  shieldings.
    |   ms_iso (bool): if True, all magnetic shieldings will be made
    |                  isotropic.
    |   q_order(int): if greater than 0, include quadrupolar interactions from
    |                   Electric Field Gradients at the given order (1 or 2).
    |   dip_sel (AtomSelection): if not None, include dipolar couplings
    |                            between atoms belonging to this set.

    """

    # Start by creating a proper isotope_list
    nmr_data = _get_nmr_data()

    nuclei = s.get_chemical_symbols()

    if isotope_list is None:
        isotope_list = [int(nmr_data[n]["iso"]) for n in nuclei]

    nuclei = [str(i)+n for i, n in zip(isotope_list, nuclei)]

    # Build header
    header = _header_template.format(channels=' '.join(set(nuclei)),
                                     nuclei=' '.join(nuclei))

    # Build MS block
    ms_block = ''

    if use_ms:

        msiso = MSIsotropy.get(s)
        if not ms_iso:
            msaniso = MSReducedAnisotropy.get(s)
            msasymm = MSAsymmetry.get(s)
            eulangs = np.array([q.euler_angles()
                                for q in MSQuaternion.get(s)])*180/np.pi
        else:
            msaniso = np.zeros(len(s))
            msasymm = np.zeros(len(s))
            eulangs = np.zeros((len(s), 3))

        for i, ms in enumerate(msiso):
            ms_block += ('shift {0} {1}p {2}p '
                         '{3} {4} {5} {6}\n').format(i+1,
                                                     ms, msaniso[i],
                                                     msasymm[i], *eulangs[i])

    # Build EFG block
    efg_block = ''

    if q_order > 0:
        if q_order > 2:
            raise ValueError('Invalid quadrupolar order')
        Cq = EFGQuadrupolarConstant(isotope_list=isotope_list)(s)
        eta_q = EFGAsymmetry.get(s)
        eulangs = np.array([q.euler_angles()
                            for q in EFGQuaternion.get(s)])*180/np.pi
        for i, cq in enumerate(Cq):
            if cq == 0:
                continue
            efg_block += ('quadrupole {0} {1} {2} {3}'
                          ' {4} {5} {6}\n').format(i+1, q_order, cq, eta_q[i],
                                                   *eulangs[i])

    # Build dipolar block
    dip_block = ''

    if dip_sel is not None and len(dip_sel) > 1:
        dip = DipolarCoupling(sel_i=dip_sel, isotope_list=isotope_list)(s)
        for (i, j), (d, v) in dip.iteritems():
            a, b = (np.array([np.arccos(-v[2]),
                              np.arctan2(-v[1],
                                         -v[0])]) % (2*np.pi)) * 180/np.pi
            dip_block += ('dipole {0} {1} {2} {3}'
                          ' {4} 0.0\n').format(i+1, j+1, d*2*np.pi, a, b)

    out_file = _spinsys_template.format(header=header, ms=ms_block,
                                        efg=efg_block, dipolar=dip_block)

    return out_file
