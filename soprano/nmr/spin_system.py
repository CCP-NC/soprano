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
    
    @property
    def n_sites(self) -> int:
        """
        Returns the number of sites in the SpinSystem.
        """
        return len(self.sites)
    @property
    def n_couplings(self) -> int:
        """
        Returns the number of couplings in the SpinSystem.
        """
        return len(self.couplings)
    @property
    def isotope_set(self) -> set[str]:
        """
        Returns a set of unique isotopes present in the SpinSystem.
        """
        return set(site.isotope for site in self.sites)
    @property
    def element_set(self) -> set[str]:
        """
        Returns a set of unique elements present in the SpinSystem.
        """
        return set(site.element for site in self.sites)
    
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
    def to_string(self, format: str = "simpson", **kwargs) -> str:
        """
        Convert the SpinSystem to a string in the specified format.

        Parameters:
        -----------
        format: str
            The format to convert to. Options are "simpson", "mrsimulator"
        kwargs: dict
            Additional keyword arguments to pass to the to_simpson or
            to_mrsimulator methods.
            
        Returns:
        --------
        str:
            The string representation of the SpinSystem in the specified format.
        """
        if format == "simpson":
            return self.to_simpson(**kwargs)
        elif format == "mrsimulator":
            mrsimulator_dict = self.to_mrsimulator(**kwargs)
            import json
            json_string = json.dumps(mrsimulator_dict, ensure_ascii=False, sort_keys=False, allow_nan=False)
            return json_string
        else:
            raise ValueError(f"Unknown format: {format}")

    def write(self, filename: Optional[str] = None, format: str = "simpson", **kwargs):
        """
        Write the SpinSystem to a file in the specified format or return as a string.

        Parameters:
        -----------
        filename: Optional[str]
            The name of the file to write to. If None, returns the string representation
            instead of writing to a file.
        format: str
            The format to write in. Options are "simpson", "mrsimulator"
        kwargs: dict
            Additional keyword arguments to pass to the to_simpson or
            to_mrsimulator methods.
            
        Returns:
        --------
        Optional[str]:
            If filename is None, returns the string representation in the specified format.
            Otherwise returns None after writing to the file.
        """
        output_string = self.to_string(format=format, **kwargs)
        
        if filename:
            with open(filename, "w") as f:
                f.write(output_string)
        else:
            print(output_string)

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

    def to_mrsimulator(
            self,
            ms_isotropic: bool = False,
            include_angles: bool = True,
            include_ms_angles: Optional[bool] = None,
            include_efg_angles: Optional[bool] = None,
            include_dipolar_angles: Optional[bool] = None,
            include_jcoupling_angles: Optional[bool] = None,
            **kwargs) -> dict:
        """
        Convert the SpinSystem to a dictionary that can be used to create an
        MRSimulator SpinSystem object.

        Parameters:
            ms_isotropic: bool
                Whether to convert the magnetic shielding/shift tensor to an isotropic one.
                (i.e. the anisotropy, asymmetry, and Euler angles are set to 0)
                Default is False.
            include_angles: bool
                Whether to include angles in the output. Default is True.
            include_ms_angles: bool
                Whether to include magnetic shielding/shift Euler angles. Default is None.
            include_efg_angles: bool
                Whether to include electric field gradient angles. Default is None.
            include_dipolar_angles: bool
                Whether to include dipolar angles. Default is None.
            include_jcoupling_angles: bool
                Whether to include J-coupling angles. Default is None.
            **kwargs: dict
                Additional keyword arguments to pass to the MRSimulator SpinSystem
                constructor.
        Returns:
            dict: The SpinSystem in MRSimulator format.
        """

        soprano_sites = self.sites
        soprano_couplings = self.couplings

        # Handle the angle logic
        include_ms_angles = include_angles if include_ms_angles is None else include_ms_angles
        include_efg_angles = include_angles if include_efg_angles is None else include_efg_angles
        include_dipolar_angles = include_angles if include_dipolar_angles is None else include_dipolar_angles
        include_jcoupling_angles = include_angles if include_jcoupling_angles is None else include_jcoupling_angles



        mrsimulator_sites = []
        for site in soprano_sites:
            mrsimulator_sites.append(
                site.to_mrsimulator(
                    ms_isotropic=ms_isotropic,
                    include_ms_angles=include_ms_angles,
                    include_efg_angles=include_efg_angles)
            )

        mrsimulator_couplings = coupling_list_to_mrsimulator(
            soprano_couplings,
            include_dipolar_angles = include_dipolar_angles,
            include_jcoupling_angles = include_jcoupling_angles,
        )

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

    def to_simpson(
            self,
            observed_nucleus: Optional[str]=None,
            q_order: Optional[int] = None,
            ms_isotropic: bool = False,
            include_angles: bool = True,
            include_ms_angles: Optional[bool] = None,
            include_efg_angles: Optional[bool] = None,
            include_dipolar_angles: Optional[bool] = None,
            include_jcoupling_angles: Optional[bool] = None,
            ) -> str:
        """
        Convert the SpinSystem to a string that can be used to create a
        Simpson SpinSystem object.

        Parameters:
            observed_nucleus: str
                The nucleus that is being observed. This is required for the
                Simpson format. The observed nucleus should be the first that
                appears in the list of channels.
            q_order: int
                The order of the quadrupole tensor. This is required for the
                Simpson format. The order should be 0, 1, or 2.
            ms_isotropic: bool
                Whether to convert the magnetic shielding/shift tensor to an isotropic one.
                (i.e. the anisotropy, asymmetry, and Euler angles are set to 0)
                Default is False.
            include_angles: bool
                Whether to include angles in the output. Default is True.
            include_ms_angles: bool
                Whether to include magnetic shielding/shift Euler angles. Default is None.
            include_efg_angles: bool
                Whether to include electric field gradient angles. Default is None.
            include_dipolar_angles: bool
                Whether to include dipolar angles. Default is None.
            include_jcoupling_angles: bool
                Whether to include J-coupling angles. Default is None.

        Returns:
            str: The SpinSystem in Simpson format.
        """

        # Handle the angle logic
        include_ms_angles = include_angles if include_ms_angles is None else include_ms_angles
        include_efg_angles = include_angles if include_efg_angles is None else include_efg_angles
        include_dipolar_angles = include_angles if include_dipolar_angles is None else include_dipolar_angles
        include_jcoupling_angles = include_angles if include_jcoupling_angles is None else include_jcoupling_angles

        nuclei = [site.isotope for site in self.sites]
        # If the observed nucleus is not specified, use the first nucleus in the list
        if not observed_nucleus:
            channels = sorted(set(nuclei))
        else:
            if observed_nucleus not in nuclei:
                raise ValueError(f"Observed nucleus {observed_nucleus} not found in the list of nuclei")
            else:
                channels = [observed_nucleus] + sorted(set(nuclei) - {observed_nucleus})

        ms_blocks = []
        efg_blocks = []
        for site in self.sites:
            ms_block, efg_block = site.to_simpson(
                q_order=q_order,
                ms_isotropic=ms_isotropic,
                include_ms_angles=include_ms_angles,
                include_efg_angles=include_efg_angles)
            ms_blocks.append(ms_block)
            efg_blocks.append(efg_block)

        ms_string = "\n".join(ms_blocks)
        efg_string = "\n".join(efg_blocks)

        dipolar_blocks = []
        for coupling in self.couplings:
            if coupling.type == "D":
                dipolar_blocks.append(coupling.to_simpson(
                    include_angles=include_dipolar_angles,
                ))
        dipolar_string = "\n".join(dipolar_blocks)

        # Combine the header and blocks into a formatted string
        output_string = f"""spinsys {{
channels {" ".join(channels)}
nuclei {" ".join(nuclei)}

{ms_string}

{efg_string}

{dipolar_string}
}}
"""
        # Trim blank lines from the end of the string
        output_string = "\n".join(line.rstrip() for line in output_string.splitlines() if line.strip())
        # Add a final line break
        output_string += "\n"
        return output_string
