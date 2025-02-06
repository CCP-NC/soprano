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
Classes and functions for interfacing with the MR simulator software.
https://mrsimulator.readthedocs.io/


From the docs: 
> We parameterize a SymmetricTensor using the Haeberlen convention 
> with parameters zeta and eta, defined as the shielding anisotropy 
> and asymmetry, respectively. 
> The Euler angle orientations, alpha, beta, and gamma are the 
> relative orientation of the nuclear shielding tensor from 
> a common reference frame.

From the docs:


Electric quadrupole: The quadrupole coupling constant, Cq. The value is a physical quantity given in units of Hz, for example, 3.1e6 for 3.1 MHz.

J-coupling: The J-coupling anisotropy, zeta, calculated using the Haeberlen convention. The value is a physical quantity given in Hz, for example, 10 for 10 Hz.

Dipolar-coupling: The dipolar-coupling constant, D. The value is a physical quantity given in Hz, for example, 9e6 for 9 kHz.
"""


from soprano.nmr import NMRTensor
from soprano.properties.nmr import (
    DipolarCoupling,
    DipolarTensor,
    EFGAsymmetry,
    EFGQuadrupolarConstant,
    JCIsotropy,
    MSAsymmetry,
    MSIsotropy,
    MSReducedAnisotropy,
)


def atoms_to_spinsys(
    atoms,
    references,
    abundance=100,
    isotopes=None,
    inc_ms=True,
    inc_efg=True,
    inc_dipolar=True,
    inc_jcouplings=True,
    name=None,
    description=None,
):
    """
    Convert a list of atoms to a list of spin systems for the MR simulator.

    Args:
      atoms (list): list of atoms to convert
      references (dict): reference chemical shielding (ppm) for each species
      abundance (float): abundance of the spin system in percent. Default is 100 %.
      isotopes (dict): dictionary of isotopes to use for each atom species
      inc_ms (bool): include magnetic shielding data
      inc_efg (bool): include EFG data
      inc_dipolar (bool): include dipolar coupling data
      inc_jcouplings (bool): include J-coupling data
      name (str): name of the spin system (optional). Default will be the formula of the atoms object.
      description (str): description of the spin system (optional). Default will be "Generated from Soprano"

    Returns:
      list: list of spin systems
    """

    name = name or atoms.get_chemical_formula()
    description = description or "Generated from Soprano"

    spinsys = {
        "name": name,
        "description": description,
        "sites": [],
        "couplings": [],
        "abundance": abundance,  # in percent
    }

    for atom in atoms:
        if isotopes is not None:
            isotope = isotopes.get(atom.symbol, None)
        else:
            isotope = None

        site_dict = {"isotope": isotope}

        ms_euler_angles = None  # TODO
        efg_euler_angles = None  # TODO
        if inc_ms:
            site_dict["isotropic_chemical_shift"] = MSIsotropy.get(atom)
            site_dict["shielding_symmetric"] = {
                # zeta is the shielding anisotropy, calculated using the Haeberlen convention.
                # The value is a physical quantity given in ppm
                "zeta": MSReducedAnisotropy.get(atom),
                # eta is the asymmetry parameter, calculated using the Haeberlen convention
                "eta": MSAsymmetry.get(atom),
                "alpha": ms_euler_angles[0],
                "beta": ms_euler_angles[1],
                "gamma": ms_euler_angles[2],
            }
        if inc_efg:
            site_dict["quadrupolar"] = {
                # Cq is the quadrupole coupling constant, given in Hz
                "Cq": EFGQuadrupolarConstant.get(atom),
                # eta is the asymmetry parameter, calculated using the Haeberlen convention
                "eta": EFGAsymmetry.get(atom),
                "alpha": efg_euler_angles[0],
                "beta": efg_euler_angles[1],
                "gamma": efg_euler_angles[2],
            }
        spinsys["sites"].append(site_dict)

    # Done with all sites, now add couplings
    # if neither dipolar nor J-couplings are requested, return now
    if not inc_dipolar and not inc_jcouplings:
        return spinsys

    # Generate a list of all possible pairs of atoms
    pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pairs.append((i, j))
            spinsys["couplings"].append({"site_index": [i, j]})

    if inc_dipolar:
        # dip is a dictionary of the form {(i,j): [dipolar_coupling, vector]}
        dip = DipolarCoupling().get(
            atoms,  # the crystal structure (ASE Atoms object)
            isotopes=isotopes,  # You can specify whatever isotope you want here.
            self_coupling=False,  # include interactions with itself
        )
        dip_tensors = DipolarTensor().get(
            atoms,
            isotopes=isotopes,
            self_coupling=False,
        )

        for i, j in pairs:
            d, v = dip[(i, j)]
            T = NMRTensor(dip_tensors[(i, j)], order="h")
            alpha, beta, gamma = T.euler_angles("zyz")
            dip_dict = {
                "D": d,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }
            spinsys["couplings"][[i, j]]["dipolar"] = dip_dict

    if inc_jcouplings:
        # jcouplings is a dictionary of the form {(i,j): j_coupling}
        jiso = JCIsotropy().get(
            atoms,
            isotopes=isotopes,
            self_coupling=False,
        )

        for i, j in pairs:
            j_dict = {
                "isotropic_j": jiso[(i, j)],
            }
            spinsys["couplings"][[i, j]]["j_coupling"] = j_dict

    return spinsys
