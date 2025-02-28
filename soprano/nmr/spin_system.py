# Soprano - a library to crack crystals! by CCP-NC
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
Contains the SpinSystem class representing a set of nuclear spin sites
in a system and any couplings between sites.

This is modelled roughly on the 
MRSimulator code: https://github.com/deepanshs/mrsimulator
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from soprano.nmr.coupling import Coupling, coupling_list_to_mrsimulator
from soprano.nmr.site import Site

TensorConvention = Literal["Haeberlen", "Increasing", "Decreasing", "NQR"]


class SpinSystem(BaseModel):
    """
    Represents a set of nuclear spin sites in a system and any couplings between sites.
    """

    sites: list[Site] = Field(
        ...,
        description="A list of nuclear spin sites in the system",
    )
    couplings: list[Coupling] = Field(default_factory=list, description="A list of couplings between sites.")
    coupling_indices: dict[tuple[int, int], list[int]] = Field(
        default_factory=dict,
        description="A dictionary of coupling indices to keep track of couplings between sites")
    
    @model_validator(mode='before')
    def validate_sites_and_couplings(cls, data):
        if 'sites' in data and not all(isinstance(site, Site) for site in data['sites']):
            raise ValueError("Each site must be a Site object")
        
        if 'couplings' in data and data['couplings'] is not None:
            # Update the coupling indices
            coupling_indices = {}
            for index, coupling in enumerate(data['couplings']):
                if not isinstance(coupling, Coupling):
                    raise ValueError("Each coupling must be a Coupling object")
                key = (coupling.site_i, coupling.site_j)
                if key not in coupling_indices:
                    coupling_indices[key] = []
                coupling_indices[key].append(index)
            data['coupling_indices'] = coupling_indices
        return data
    
    def add_site(self, site: Site):
        """
        Add a site to the SpinSystem.

        Parameters:
        -----------
        site: Site
            The site to add
        """
        self.sites.append(site)
    
    def update_site(self, index: int, site: Site):
        """
        Update a site in the SpinSystem.

        Parameters:
        -----------
        index: int
            The index of the site to update
        site: Site
            The new site
        """
        self.sites[index] = site

    def remove_site(self, index: int):
        """
        Remove a site from the SpinSystem.

        Parameters:
        -----------
        index: int
            The index of the site to remove
        """
        self.sites.pop(index)


    
    def add_coupling(self, coupling: Coupling):
        self.couplings.append(coupling)
        key = (coupling.site_i, coupling.site_j)
        if key not in self.coupling_indices:
            self.coupling_indices[key] = []
        self.coupling_indices[key].append(len(self.couplings) - 1)

    def get_couplings(self, site_i: int, site_j: int) -> Optional[list[Coupling]]:
        key = (site_i, site_j)
        if key in self.coupling_indices:
            return [self.couplings[i] for i in self.coupling_indices[key]]
        return None

    def update_coupling(self, site_i: int, site_j: int, new_coupling: Coupling):
        key = (site_i, site_j)
        coupling_type = new_coupling.type
        if key in self.coupling_indices:
            for i in self.coupling_indices[key]:
                if self.couplings[i].type == coupling_type:
                    self.couplings[i] = new_coupling
                    return
        raise KeyError(f"No coupling found for sites {site_i} and {site_j} with type {coupling_type}")

    def remove_coupling(self, site_i: int, site_j: int, coupling_type: str):
        """
        Remove a coupling between two sites.
        
        Parameters:
        -----------
        site_i: int
            The index of the first site
        site_j: int
            The index of the second site
        coupling_type: str
            The type of coupling to remove: "D" or "J" for dipolar or J-coupling

        Raises:
        -------
        KeyError: If no coupling is found between the two sites with the given type
        """
        key = (site_i, site_j)
        if key in self.coupling_indices:
            indices_to_remove = [i for i in self.coupling_indices[key] if self.couplings[i].type == coupling_type]
            if indices_to_remove:
                for i in sorted(indices_to_remove, reverse=True):
                    del self.couplings[i]
                    self.coupling_indices[key].remove(i)
                if not self.coupling_indices[key]:  # Remove the key if the list is empty
                    del self.coupling_indices[key]
                return
        raise KeyError(f"No coupling found for sites {site_i} and {site_j} with type {coupling_type}")
    
    def copy(self, deep: bool = True, **kwargs) -> 'SpinSystem':
        """
        Create a copy of the SpinSystem object.
        
        This method overrides the deprecated copy method with model_copy.
        
        Parameters:
        -----------
        deep : bool, optional
            If True, performs a deep copy of nested objects. 
            Default is True to ensure all nested objects are properly copied.
        **kwargs : dict
            Additional keyword arguments to pass to model_copy
        
        Returns:
        --------
        SpinSystem
            A new SpinSystem object that is a copy of the current object
        
        Notes:
        ------
        Deprecation warning for the original copy method is automatically 
        handled by Pydantic.
        """
        return self.model_copy(deep=deep, **kwargs)

    def __eq__(self, other: 'SpinSystem') -> bool:
        # Check attributes
        if len(self.sites) != len(other.sites):
            return False
        for site, other_site in zip(self.sites, other.sites):
            if site != other_site:
                return False
        if len(self.couplings) != len(other.couplings):
            return False
        for coupling, other_coupling in zip(self.couplings, other.couplings):
            if coupling != other_coupling:
                return False

        # Check coupling indices dictionary
        if self.coupling_indices != other.coupling_indices:
            return False
    
        return True
    
    def __hash__(self) -> int:
        """
        Generate a hash for the SpinSystem object.

        The hash is based on:
        - Sites (order-dependent)
        - Couplings (order-dependent)
        - Euler angle configuration

        Returns:
        --------
        int
            A hash value for the SpinSystem object
        """
        # Hash sites (order matters)
        sites_hash = tuple(hash(site) for site in self.sites)

        # Hash couplings (order matters)
        couplings_hash = tuple(hash(coupling) for coupling in self.couplings)

        # Combine all components
        return hash((
            sites_hash,
            couplings_hash,
        ))
    

    # Interfaces:
    # =============================================================================

    def to_mrsimulator(self, **kwargs):
        """
        Convert the SpinSystem to a dictionary that can be used to create an
        MRSimulator SpinSystem object.
        """

        soprano_sites = self.sites
        soprano_couplings = self.couplings


        mrsimulator_sites = []
        for site in soprano_sites:
            mrsimulator_sites.append(site.to_mrsimulator())

        mrsimulator_couplings = coupling_list_to_mrsimulator(soprano_couplings)

        return {
            "sites": mrsimulator_sites,
            "couplings": mrsimulator_couplings,
            **kwargs
        }

    def from_mrsimulator(self, mrsimulator_spin_system):
        """
        Convert an MRSimulator SpinSystem object to a SpinSystem object.
        """
        raise NotImplementedError("Conversion from MRSimulator SpinSystem not yet implemented")

    def to_simpson(self, observed_nucleus: Optional[str]) -> str:
        """
        Convert the SpinSystem to a string that can be used to create a
        Simpson SpinSystem object.

        Parameters:
            observed_nucleus: str
                The nucleus that is being observed. This is required for the
                Simpson format. The observed nucleus should be the first that
                appears in the list of channels.

        Returns:
            str: The SpinSystem in Simpson format.
        """

        nuclei = [site.isotope for site in self.sites]
        # If the observed nucleus is not specified, use the first nucleus in the list
        if observed_nucleus is None:
            channels = sorted(set(nuclei))
        else:
            if observed_nucleus not in nuclei:
                raise ValueError(f"Observed nucleus {observed_nucleus} not found in the list of nuclei")
            else:
                channels = [observed_nucleus] + sorted(set(nuclei) - {observed_nucleus})

        ms_blocks = []
        efg_blocks = []
        for site in self.sites:
            ms_block, efg_block = site.to_simpson()
            ms_blocks.append(ms_block)
            efg_blocks.append(efg_block)

        ms_string = "\n".join(ms_blocks)
        efg_string = "\n".join(efg_blocks)

        dipolar_blocks = []
        for coupling in self.couplings:
            if coupling.type == "D":
                dipolar_blocks.append(coupling.to_simpson())
        dipolar_string = "\n".join(dipolar_blocks)

        # Combine the header and blocks into a formatted string
        return f"""spinsys{{
channels {" ".join(channels)}
nuclei {" ".join(nuclei)}

{ms_string}

{efg_string}

{dipolar_string}
}}
        """
