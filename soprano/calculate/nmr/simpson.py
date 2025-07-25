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


import re
import warnings
from collections import namedtuple

import numpy as np

from soprano.data.nmr import _get_nmr_data
from soprano.properties.nmr import (
    DipolarCoupling,
    EFGAsymmetry,
    EFGQuadrupolarConstant,
    EFGQuaternion,
    MSAsymmetry,
    MSIsotropy,
    MSQuaternion,
    MSReducedAnisotropy,
)

_spinsys_template = """spinsys {{
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


def write_spinsys(
    s,
    isotope_list=None,
    use_ms=False,
    ms_iso=False,
    ms_tag='ms',
    q_order=0,
    efg_tag='efg',
    dip_sel=None,
    path=None,
    ref={},
    grad=-1.0,
    obs_nuc=None,
):
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
    |   ms_tag (str): tag for the magnetic shielding tensor array.
    |   q_order(int): if greater than 0, include quadrupolar interactions from
    |                   Electric Field Gradients at the given order (1 or 2).
    |   efg_tag (str): tag for the EFG tensor array.
    |   dip_sel (AtomSelection): if not None, include dipolar couplings
    |                            between atoms belonging to this set.
    |   path (str): path to save the newly created file to. If not provided,
    |               the contents will be simply returned as a string.
    |   ref (dict): dictionary of reference values for the calculation. This
    |               is used to convert from raw shielding values to chemical
    |               shifts. The dictionary should be in the form
    |               {element: value}, where value is the reference shielding
    |               for that element in ppm.
    |   grad (float|dict|list): gradient to use when converting from raw
    |                           shielding values to chemical shifts. If a
    |                           float is provided, it will be used for all
    |                           elements. If a dictionary is provided, it
    |                           should be in the form {element: value}, where
    |                           value is the gradient for that element. If a
    |                           list is provided, it should be have one value
    |                           per site. Defaults to a gradient of -1.0 for
    |                           all elements.
    |   obs_nuc (str) : specify the nucleus to be observed, e.g. 1H.  
    
    | Returns:
    |   file_contents (str): spinsys file in string format. Only returned if
    |                        no save path is provided.

    """

    # Start by creating a proper isotope_list
    nmr_data = _get_nmr_data()

    nuclei = s.get_chemical_symbols()

    if isotope_list is None:
        isotope_list = [int(nmr_data[n]["iso"]) for n in nuclei]

    nuclei = [str(i) + n for i, n in zip(isotope_list, nuclei)]

    # Ensure obs_nuc appears first in channels
    if obs_nuc is not None:
        # Check if obs_nuc is in nuclei
        if obs_nuc not in nuclei:
            raise ValueError(
                f"{obs_nuc} not found in the list of nuclei"
            )
        else:
            channels = [obs_nuc] + [n for n in sorted(set(nuclei)) if n != obs_nuc]
    else:
        channels = sorted(set(nuclei))

    # Build header
    header = _header_template.format(
        channels=" ".join(channels), nuclei=" ".join(nuclei)
    )

    # Build MS block
    ms_block = ""

    if use_ms:
        if not ref:
            warnings.warn(
                "No reference values provided for the calculation of "
                "chemical shifts. Assuming all zero."
                "To avoid this warning, provide a dictionary of the form "
                "{element: value}, where value is the reference shielding "
                "for that element in ppm."
            )
        msiso = MSIsotropy.get(s, ref=ref, grad=grad, tag=ms_tag)
        if not ms_iso:
            msaniso = MSReducedAnisotropy.get(s, tag=ms_tag)
            msasymm = MSAsymmetry.get(s, tag=ms_tag)
            eulangs = (
                np.array([q.euler_angles() for q in MSQuaternion.get(s, tag=ms_tag)]) * 180 / np.pi
            )
        else:
            msaniso = np.zeros(len(s))
            msasymm = np.zeros(len(s))
            eulangs = np.zeros((len(s), 3))

        for i, ms in enumerate(msiso):
            ms_block += ("shift {0} {1}p {2}p " "{3} {4} {5} {6}\n").format(
                i + 1, ms, msaniso[i], msasymm[i], *eulangs[i]
            )

    # Build EFG block
    efg_block = ""

    if q_order > 0:
        if q_order > 2:
            raise ValueError("Invalid quadrupolar order")
        Cq = EFGQuadrupolarConstant(isotope_list=isotope_list, tag=efg_tag)(s)
        eta_q = EFGAsymmetry.get(s, tag=efg_tag)
        eulangs = (
            np.array([q.euler_angles() for q in EFGQuaternion.get(s, tag=efg_tag)]) * 180 / np.pi
        )
        for i, cq in enumerate(Cq):
            if cq == 0:
                continue
            efg_block += ("quadrupole {0} {1} {2} {3}" " {4} {5} {6}\n").format(
                i + 1, q_order, cq, eta_q[i], *eulangs[i]
            )

    # Build dipolar block
    dip_block = ""

    if dip_sel is not None and len(dip_sel) > 1:
        dip = DipolarCoupling(sel_i=dip_sel, isotope_list=isotope_list)(s)
        for (i, j), (d, v) in dip.items():
            a, b = (
                (np.array([np.arccos(-v[2]), np.arctan2(-v[1], -v[0])]) % (2 * np.pi))
                * 180
                / np.pi
            )
            dip_block += (f"dipole {i + 1} {j + 1} {d * 2 * np.pi} {a}" f" {b} 0.0\n")

    out_file = _spinsys_template.format(
        header=header, ms=ms_block, efg=efg_block, dipolar=dip_block
    )

    if path is None:
        return out_file
    else:
        with open(path, "w") as of:
            of.write(out_file)


def load_simpson_dat(filename):
    """Load a SIMPSON output .dat file and return it as a numpy array."""

    dat = np.loadtxt(filename)
    return np.concatenate(
        [dat[:, 0, None], (dat[:, 1] + 1.0j * dat[:, 2])[:, None]], axis=1
    )


SimpsonTemplates = namedtuple(
    "SimpsonTemplates",
    """BASIC_SPECTRUM,
                              BROADENED_SPECTRUM
                              """,
)

SimpsonTemplates = SimpsonTemplates(
    BASIC_SPECTRUM={
        "pars": {},
        "pulseq": """
global par
delay [expr 1e6/$par(sw)]
store 1
acq $par(np) 1
""",
        "main": """
global par

set f [fsimpson]
fsave $f $par(name)_fid.dat -xreim
fft $f
fsave $f $par(name)_spe.dat -xreim
puts "Simulation complete"
""",
    },
    BROADENED_SPECTRUM={
        "pars": {
            "broad": "1",
        },
        "pulseq": """
global par
delay [expr 1e6/$par(sw)]
store 1
acq $par(np) 1
""",
        "main": """
global par

set f [fsimpson]
fsave $f $par(name)_fid.dat -xreim
faddlb $f $par(broad) 0
fft $f
fsave $f $par(name)_spe.dat -xreim
puts "Simulation complete"
""",
    },
)


def _check_template_pars(f):
    def decorated_f(self, *args, **kwargs):
        f(self, *args, **kwargs)

        # Scan for missing parameters
        par_re = re.compile("\\$par\\(([0-9a-zA-Z_]+)\\)")

        req_pars = set(par_re.findall(self.main) + par_re.findall(self.pulseq))
        # Remove 'name' since that's defined by default
        req_pars = req_pars.difference(["name"])

        if not req_pars.issubset(set(self.pars.keys())):
            print(
                "WARNING: some parameters required for this sequence seem"
                " to be not defined"
            )

    return decorated_f


class SimpsonSequence:

    """SimpsonSequence

    A class storing parameters and scripts for the production of a SIMPSON
    input file. The parameters of the simulation are stored in a dictionary
    member accessible as .pars and can be set at will.

    | Args:
    |   spinsys_source (str): path of the .spinsys file to use in the
    |                         simulation

    """

    def __init__(self, spinsys_source):
        self.spinsys_source = spinsys_source

        self.pars = {
            "proton_frequency": "4e7",
            "start_operator": "Inx",
            "detect_operator": "Inx",
            "np": "8192",
            "sw": "8000",
            "num_cores": "1",
            "crystal_file": "alpha0beta0",
        }

        self.apply_template(SimpsonTemplates.BASIC_SPECTRUM)

    @_check_template_pars
    def apply_template(self, template):
        """Apply an existing sequence template from SimpsonTemplates,
        including default parameters.

        | Args:
        |   template (dict): the template to apply

        """

        self.pars.update(template["pars"])
        self.pulseq = template["pulseq"]
        self.main = template["main"]

    @_check_template_pars
    def apply_custom_template(self, pars, pulseq, main):
        """Apply a custom sequence template, defined by pars, pulse sequence
        and main script.

        | Args:
        |   pars (dict): default parameters for this sequence. Use an empty
        |                dict if none of relevance.
        |   pulseq (str): the script for the pulse sequence block.
        |   main (str): the script for the main block.

        """

        self.pars.update(pars)
        self.pulseq = pulseq
        self.main = main

    @_check_template_pars
    def set_parameters(self, **new_pars):
        """Set one or more parameters of the calculation. Compared to editing
        .pars directly it applies some more checks and is safer. Pass the new
        parameters as named arguments to this function.

        """

        self.pars.update(new_pars)

    def write_input(self, path=None):
        """Print out the .in file.

        | Args:
        |   path (str): path to save the newly created file to. If not provided,
        |               the contents will be simply returned as a string.

        """

        outf = f"source {self.spinsys_source}\n\n"

        # Write out par block
        outf += "par {\n"
        for p, val in self.pars.items():
            outf += f"\t{p} {val}\n"
        outf += "}\n\n"

        # Write out pulseq
        outf += "proc pulseq {} {\n"
        outf += self.pulseq
        outf += "\n}\n\n"

        # Write out main
        outf += "proc main {} {\n"
        outf += self.main
        outf += "\n}\n\n"

        if path is None:
            return outf
        else:
            with open(path, "w") as of:
                of.write(outf)
