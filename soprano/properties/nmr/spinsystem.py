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

"""Implementation of AtomsProperties that relate to NMR SpinSystems.

This is essentially a way to extract the NMR-relevant information from an atoms object
and store it as a SpinSystem object.
"""


from typing import Optional, Union

from ase import Atoms
import numpy as np

from soprano.data.nmr import  _get_isotope_list, get_isotope_list_from_species
from soprano.nmr.site import Site
from soprano.nmr.coupling import DipolarCoupling as SiteDipolarCoupling
from soprano.nmr.spin_system import SpinSystem
from soprano.properties import AtomsProperty
from soprano.properties.nmr import MSTensor, EFGTensor, DipolarCouplingList
from soprano.selection import AtomSelection


class NMRSpinSystem(AtomsProperty):
    """
    Extact the information needed to output a SpinSystem object from an Atoms object.

    Parameters:
    ------------
    isotopes: list/dict/None
        Specify the isotopes to be used. If None, default NMR-active isotopes
        are chosen. Can be a dict like {'H': 2, 'C': 13} or a list like
        ['2H', '13C']. If list, you can specify either one isotope per site
        or a list of the unique isotopes (in which case all sites of that element
        will have the same isotope).
    references: list/dict/None
        Specification of the references to convert magnetic shielding tensors to
        chemical shifts. Can be a dict like {'H': 30, 'C': 170} or a list like
        [30, 170]. If list, you must specify one reference per site in the system.
        If None, no conversion is done and the outputs will be shieldings.
    gradients: list/dict/float
        Specification of the gradients to convert magnetic shielding tensors to
        chemical shifts. Default is -1.0 corresponding as in the typical formula:
        delta = (reference + gradient * shielding) / (1 - reference * 1e-6).
    include_shielding: bool
        Whether to include magnetic shielding tensors in the output.
    include_efg: bool
        Whether to include electric field gradient tensors in the output.
    include_dipolar: bool
        Whether to include dipolar couplings in the output.
    include_j: bool
        Whether to include spin-spin (J) couplings in the output.
    coupling_kwargs: dict
        A dictionary of keyword arguments to pass to the Coupling constructor.
        (see the documentation of the Coupling class for more details)

    Returns:
    --------
    spin_system: SpinSystem
        The SpinSystem object containing the extracted information.
    """

    default_name = "spin_system"
    default_params = {
        "isotopes": None,
        "references": None,
        "gradients": -1.0,
        "include_shielding": True,
        "include_efg": True,
        "include_dipolar": False,
        "include_j": False,
        "use_q_isotopes": False,
        "coupling_kwargs": None,
    }

    @staticmethod
    def extract(
        s,
        isotopes,
        references,
        gradients,
        include_shielding,
        include_efg,
        include_dipolar,
        include_j,
        use_q_isotopes,
        coupling_kwargs,
    ):
        
        if coupling_kwargs is None:
            coupling_kwargs = {}


        # Generate the sites
        elements = s.get_chemical_symbols()

        if not s.has("labels"):
            # TODO what should this look like? Use MagresView labels?
            labels = ["{}{}".format(el, i) for i, el in enumerate(elements)]
            s.new_array("labels", labels, dtype=str)

        # Isotopes:
        # If isotopes is None, we get default ones, if dict, we use that
        # If list, we use that as the list of isotopes
        if isotopes is None or isinstance(isotopes, dict):
            isotope_list = _get_isotope_list(elements, isotopes=isotopes)
        elif isinstance(isotopes, list):
            isotope_list = get_isotope_list_from_species(elements, isotopes)
        else:
            raise ValueError(
                "isotopes must be either None, a dictionary or a list of species strings"
            )

        # TODO handle references and gradients

        sites = NMRSites.extract(
            s,
            isotopes=isotopes,
            references=references,
            gradients=gradients,
            include_shielding=include_shielding,
            include_efg=include_efg,
            use_q_isotopes=use_q_isotopes,
        )

        couplings = []

        if include_dipolar:
            dipolar_couplings = DipolarCouplingList.get(s, isotope_list=isotope_list, **coupling_kwargs)
            if dipolar_couplings is not None:
                couplings.extend(dipolar_couplings)
                

        # TODO handle J couplings


        # Return the SpinSystem object
        return SpinSystem(sites=sites, couplings=couplings)



        
class NMRSites(AtomsProperty):
    """
    Extract a list of Site objects from an Atoms object.

    Parameters:
    -----------
    isotopes: list/dict/None
        Specify the isotopes to be used. If None, default NMR-active isotopes
        are chosen. Can be a dict like {'H': 2, 'C': 13} or a list like
        ['2H', '13C']. If list, you can specify either one isotope per site
        or a list of the unique isotopes (in which case all sites of that element
        will have the same isotope).
    references: list/dict/None
        Specification of the references to convert magnetic shielding tensors to
        chemical shifts. Can be a dict like {'H': 30, 'C': 170} or a list like
        [30, 170]. If list, you must specify one reference per site in the system.
        If None, no conversion is done and the outputs will be shieldings.
    gradients: list/dict/float
        Specification of the gradients to convert magnetic shielding tensors to
        chemical shifts. Default is -1.0 corresponding as in the typical formula:
        delta = (reference + gradient * shielding) / (1 - reference * 1e-6).
    include_shielding: bool
        Whether to include magnetic shielding tensors in the output.
    include_efg: bool
        Whether to include electric field gradient tensors in the output.
    use_q_isotopes: bool
        Whether to use quadrupolar isotopes for elements that have them. This is 
        only relevant if isotopes is None.

    Returns:
    --------
    sites: list
        A list of Site objects extracted from the Atoms object.

    """

    default_name = "nmr_sites"
    default_params = {
        "isotopes": None,
        "references": None,
        "gradients": -1.0,
        "include_shielding": True,
        "include_efg": True,
        "use_q_isotopes": False,
    }

    @staticmethod
    def extract(s, isotopes, references, gradients, include_shielding, include_efg, use_q_isotopes) -> list[Site]:

        elements = s.get_chemical_symbols()

        # Labels:
        if not s.has("labels"):
            # TODO what should this look like? Use MagresView labels?
            labels = ["{}{}".format(el, i) for i, el in enumerate(elements)]
            s.new_array("labels", labels, dtype=str)
        
        # Isotopes:
        if isotopes is None:
            isotope_list = _get_isotope_list(elements, use_q_isotopes=use_q_isotopes)
        elif isinstance(isotopes, dict):
            isotope_list = _get_isotope_list(elements, isotopes=isotopes)
        elif isinstance(isotopes, list):
            isotope_list = get_isotope_list_from_species(elements, isotopes, use_q_isotopes=use_q_isotopes)
        else:
            raise ValueError(
                "isotopes must be either None, a dictionary or a list of species strings"
            )

        species_list = [f'{iso}{el}' for iso, el in zip(isotope_list, elements)]

        sites = [Site(isotope=specie, label=label) for specie, label in zip(species_list, s.get_array("labels"))]

        ms = None
        if include_shielding:
            ms_tensors = MSTensor.get(s, references=references, gradients=gradients)
            if ms_tensors is not None:
                if len(ms_tensors) != len(sites):
                    raise ValueError(
                        "Mismatch between number of sites and number of magnetic shielding tensors"
                    )
                for site, ms_tensor in zip(sites, ms_tensors):
                    site.ms = ms_tensor
        efg = None
        if include_efg:
            efg_tensors = EFGTensor.get(s, isotope_list=isotope_list)
            if efg_tensors is not None:
                if len(efg_tensors) != len(sites):
                    raise ValueError(
                        "Mismatch between number of sites and number of electric field gradient tensors"
                    )
                for site, efg_tensor in zip(sites, efg_tensors):
                    site.efg = efg_tensor

        return sites

def get_sites(
        atoms: Atoms,
        isotopes: Optional[dict[str, int]] = None,
        indices: Optional[list[int]] = None,
        selection: Optional[AtomSelection] = None,
        include_shielding = True,
        include_efg = True,
        use_q_isotopes = False,
    ) -> Optional[list[Site]]:
    """
    Extract a list of Site objects from an Atoms object. You can specify the indices
    of the atoms to extract as sites or use an AtomSelection object to extract the sites.
    Don't use both indices and selection at the same time.

    This is a wrapper around the NMRSites.get method - you can use that for more advanced
    options. For even more advanced use cases, you can build Site objects manually.

    Parameters:
    atoms (ase.Atoms): The Atoms object from which to extract the sites.
    isotopes (dict[str, int]): A dictionary of isotopes to use for the sites. e.g. {'H': 2, 'C': 13}
    indices (list[int]): The indices of the atoms to extract as sites.
    selection (AtomSelection): An AtomSelection object to use to extract the sites

    Returns:
    list: A list of Site objects extracted from the Atoms object.
    """
    if indices is not None and selection is not None:
        raise ValueError("You can't use both indices and selection at the same time in `get_sites`.")

    # Create a subset of the atoms object
    if indices is not None:
        atoms = atoms[indices]
    elif selection is not None:
        atoms = selection.subset(atoms)

    sites = NMRSites.get(
        atoms,
        isotopes=isotopes,
        include_shielding=include_shielding,
        include_efg=include_efg,
        use_q_isotopes=use_q_isotopes
        )

    return sites


def get_dipolar_couplings(
        atoms: Atoms,
        sel_i: Optional[Union[AtomSelection, list[int]]] = None,
        sel_j: Optional[Union[AtomSelection, list[int]]] = None,
        isotopes: Optional[dict[str, int]] = None,
        isotope_list: Optional[list[Optional[int]]] = None,
        self_coupling: bool = False,
        rotation_axis: Optional[np.ndarray] = None,
        isonuclear: bool = False,
    ) -> Optional[list[SiteDipolarCoupling]]:
    """
    Extract a list of DipolarCoupling objects from an Atoms object.

        Parameters:
      sel_i (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to compute the dipolar
                                      coupling. By default is None
                                      (= all of them).
      sel_j (AtomSelection or [int]): Selection or list of indices of atoms
                                      for which to compute the dipolar
                                      coupling with the ones in sel_i. By
                                      default is None (= same as sel_i).
      isotopes (dict): dictionary of specific isotopes to use, by element
                       symbol. If the isotope doesn't exist an error will
                       be raised. e.g. {'H': 2, 'C': 13}
      self_coupling (bool): if True, include coupling of a nucleus with its
                            own closest periodic copy. Otherwise excluded.
                            Default is False.
      block_size (int): maximum size of blocks used when processing large
                        chunks of pairs. Necessary to avoid memory problems
                        for very large systems. Default is 1000.
      rotation_axis (np.ndarray): if present, return the residual dipolar
                                  tensors after fast averaging around the
                                  given axis. Default is None.
      isonuclear (bool): if True, only compute couplings between nuclei of
                         the same element. Default is False.

    Returns:
      dip_list (list): List of `DipolarCoupling` objects.
    """

    return DipolarCouplingList.get(
        atoms,
        sel_i=sel_i,
        sel_j=sel_j,
        isotopes=isotopes,
        isotope_list=isotope_list,
        self_coupling=self_coupling,
        rotation_axis=rotation_axis,
        isonuclear=isonuclear,
    )    
    


def get_spin_system(
        atoms: Atoms,
        isotopes: Optional[dict[str, int]] = None,
        references: Optional[dict[str, float]] = None,
        gradients: Optional[Union[dict[str, float], float]] = -1,
        include_shielding: bool = True,
        include_efg: bool = True,
        include_dipolar: bool = False,
        include_j: bool = False,
        use_q_isotopes: bool = False,
        coupling_kwargs: Optional[dict] = None,
    ) -> SpinSystem:
    """
    Extract the information needed to output a SpinSystem object from an Atoms object.

    Parameters:
    ------------
    atoms (ase.Atoms): The Atoms object from which to extract the sites.
    isotopes (dict[str, int]): A dictionary of isotopes to use for the sites. e.g. {'H': 2, 'C': 13}
    references (dict[str, float]): Specification of the references to convert magnetic shielding tensors to
        chemical shifts. e.g. {'H': 30, 'C': 170}
    gradients (dict[str, float]/float): Specification of the gradients to convert magnetic shielding tensors to
        chemical shifts. Default is -1.0 corresponding as in the typical formula:
        delta = (reference + gradient * shielding) / (1 - reference * 1e-6).
    include_shielding (bool): Whether to include magnetic shielding tensors in the output.
    include_efg (bool): Whether to include electric field gradient tensors in the output.
    include_dipolar (bool): Whether to include dipolar couplings in the output.
    include_j (bool): Whether to include spin-spin (J) couplings in the output.
    use_q_isotopes (bool): Whether to use quadrupolar isotopes for elements that have them. This is 
        only relevant if isotopes is None.
    coupling_kwargs (dict): A dictionary of keyword arguments to pass to the Coupling constructor.

    Returns:
    --------
    spin_system (SpinSystem): The SpinSystem object containing the extracted information.
    """

    return NMRSpinSystem.get(
        atoms,
        isotopes=isotopes,
        references=references,
        gradients=gradients,
        include_shielding=include_shielding,
        include_efg=include_efg,
        include_dipolar=include_dipolar,
        include_j=include_j,
        use_q_isotopes=use_q_isotopes,
        coupling_kwargs=coupling_kwargs,
    )