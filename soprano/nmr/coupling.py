# Soprano - a library to crack crystals!
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
Contains the Coupling class, which is an extension of the NMRTensor class, and is
used to represent a generic coupling tensor between two sites.

TODO: decide if the J and dipolar coupling classes should take the raw tensor or the
tensor scaled by the gyromagnetic ratios. i.e. should we us the gamma1 and gamma2
values to scale the results or not. At the moment the J coupling class does this
but the dipolar coupling class does not.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from soprano.data.nmr import nmr_gamma
from soprano.nmr.tensor import NMRTensor
from soprano.nmr.utils import (
    _anisotropy,
    _dip_constant,
    _dip_tensor,
    _haeb_sort,
    _J_constant,
    _split_species,
)


class Coupling(BaseModel, ABC):
    site_i: int = Field(description="Index of first site")
    site_j: int = Field(description="Index of second site")
    species1: str = Field(description="Chemical/isotope symbol of first site. e.g. '13C'")
    species2: str = Field(description="Chemical/isotope symbol of second site. e.g. '1H'")
    tensor: NMRTensor = Field(description="The coupling tensor between the two sites")
    type: Literal["J", "D"] = Field(description="The type of coupling (J-coupling or dipolar coupling etc.)")
    tag: Optional[str] = Field(default=None, description="A tag to identify the coupling tensor")

    gamma1: Optional[float] = Field(
        default=None,
        description="The gyromagnetic ratio of the first site. If not provided, it is looked up in the nmrdata.json file for the specified species",
    )
    gamma2: Optional[float] = Field(
        default=None,
        description="The gyromagnetic ratio of the second site. If not provided, it is looked up in the nmrdata.json file for the specified species",
    )

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        )

    @model_validator(mode="after")
    def set_default_gammas(self):
        if not self.gamma1:
            iso, el = _split_species(self.species1)
            self.gamma1 = nmr_gamma(el, iso=iso)
        if not self.gamma2:
            iso, el = _split_species(self.species2)
            self.gamma2 = nmr_gamma(el, iso=iso)
        return self

    @field_validator("species1", "species2")
    @classmethod
    def validate_species(cls, species):
        _split_species(species)
        return species

    @property
    def is_self_coupling(self):
        return self.site_i == self.site_j

    @property
    def element_symbols(self):
        return tuple(_split_species(species)[1] for species in [self.species1, self.species2])

    @property
    def isotope_numbers(self):
        return tuple(_split_species(species)[0] for species in [self.species1, self.species2])

    @property
    @abstractmethod
    def coupling_constant(self):
        pass

    def euler_angles(self, **kwargs):
        return self.tensor.euler_angles(**kwargs)
    
    def copy(self, deep: bool = True, **kwargs) -> 'Coupling':
        """
        Create a copy of the Coupling object.
        
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
        Coupling
            A new Coupling object that is a copy of the current object
        
        Notes:
        ------
        Deprecation warning for the original copy method is automatically 
        handled by Pydantic.
        """
        return self.model_copy(deep=deep, **kwargs)
    
    @abstractmethod
    def to_mrsimulator(self):
        pass


# end class Coupling

# Sub classes for specific types of couplings (j-coupling, dipolar coupling, etc.)
class ISCoupling(Coupling):
    """
    Class to represent a indirect-spin coupling tensor between two sites.

    This expects the J coupling reduced tensor such as those found in a .magres file.

    To get the J coupling tensor, we scale by the gyromagnetic ratios of the two sites.
    This class contains methods to return the J-coupling constant, J-coupling tensor and orientation
    given by the Euler angles.

    Usage:
    ```
    # Create an ISCoupling object
    is_coupling = ISCoupling(site_i=0, site_j=1, species1='13C', species2='1H', tensor=nmr_tensor, tag='J-coupling')
    ```

    This will typically be used in a SpinSystem object to represent the coupling between two sites for export to a spin simulator such
    as MRSimulator or Simpson.

    """
    type: str = "J" # J-coupling

    @property
    def J_evals(self):
        """
        Produces the eigenvalues of the J-coupling tensor for the pair of nuclei
        in the system. The J coupling for a pair of nuclei i and j is defined as:

        .. math::

            J_{ij} = 10^{19}\\frac{h\\gamma_i\\gamma_j}{4\\pi^2}K

        where the gammas represent the gyromagnetic ratios of the nuclei and K is
        the J coupling reduced tensor found in a .magres file, in :math:`10^{19} T^2 J^{-1}`.


        """
        evals = self.tensor.eigenvalues  # eigenvalues of the IScoupling tensor.

        return _J_constant(evals, self.gamma1, self.gamma2)

    @property
    def J_eigenvectors(self):
        """
        The eigenvectors of the J-coupling tensor for the pair of nuclei in the system
        - these are the same as the eigenvectors J coupling reduced tensor found in a .magres file.

        """
        return self.tensor.eigenvectors

    @property
    def Jisotropy(self):
        """
        The isotropic J-coupling constant for the pair of nuclei in the system.

        """
        return np.average(self.J_evals)

    @property
    def J_anisotropy(self):
        """
        The anisotropic J-coupling constant for the pair of nuclei in the system.

        """
        jc_evals = self.J_evals
        jc_evals = _haeb_sort(jc_evals)
        return _anisotropy(jc_evals)

    @property
    def J_reduced_anisotropy(self):
        """
        The reduced anisotropic J-coupling constant for the pair of nuclei in the system.

        """
        jc_evals = self.J_evals
        jc_evals = _haeb_sort(jc_evals)
        return _anisotropy(jc_evals, reduced=True)

    @property
    def coupling_constant(self):
        return self.Jisotropy



# end class ISCoupling

# Alias of ISCoupling: JCoupling ?
# JCoupling = ISCoupling


class DipolarCoupling(Coupling):
    """
    Class to represent a dipolar coupling tensor between two sites.

    This expects the dipolar coupling tensor such as those found in a .magres file.

    To get the dipolar coupling tensor, we scale by the gyromagnetic ratios of the two sites.
    This class contains methods to return the dipolar coupling constant and the dipolar coupling tensor.
    """

    type: str = "D"  # Dipolar coupling

    @property
    def coupling_constant(self):
        """
        The dipolar coupling constant for the pair of nuclei in the system in Hz.

        The eigenvalues of the dipolar coupling tensor are:
        -d, -d, 2d

        We sort these in NQR ordering convention (abs increasing),
        take half the third eigenvalue and return it.

        TODO: double-check if this is general enough for all cases.

        """
        return self.tensor.eigenvalues[2] / 2

    @classmethod
    def from_distance_vector(
        cls,
        distance_vector: np.ndarray,
        species1: str,
        species2: str,
        site_i: int = 0,
        site_j: int = 1,
        gamma_1: Optional[float] = None,
        gamma_2: Optional[float] = None,
        tag: Optional[str] = None,
        euler_convention: Literal["zyz", "zxz"] = "zyz",
        euler_passive: bool = False,
        euler_degrees: bool = False,
    ) -> "DipolarCoupling":
        """
        Create a DipolarCoupling object from a distance vector.

        Usage:
        ```
        distance_vector = np.array([x, y, z])
        dipolar_coupling = DipolarCoupling.from_distance_vector(distance_vector, species1, species2)
        ```

        Args:
            distance_vector: The distance vector between the two sites in the system.
            species1: The chemical or isotope symbol of the first site.
            species2: The chemical or isotope symbol of the second site.
            site_i: The index of the first site in the coupling. Defaults to 0.
            site_j: The index of the second site in the coupling. Defaults to 1.
            gamma_1: The gyromagnetic ratio of the first site. If not provided, it is looked up in the nmrdata.json file for the specified species.
            gamma_2: The gyromagnetic ratio of the second site. If not provided, it is looked up in the nmrdata.json file for the specified species.
            tag: A tag to identify the coupling tensor.
            euler_convention: The convention used for the Euler angles.
            euler_passive: Whether the Euler angles are passive.
            euler_degrees: Whether the Euler angles are in degrees. Default is radians (False).

        Returns:
            A DipolarCoupling object.

        """
        if distance_vector.shape != (3,):
            raise ValueError("The distance vector must be a 3D vector")

        if gamma_1 is None:
            iso1, element1 = _split_species(species1)
            gamma_1 = nmr_gamma(element1, iso=iso1)
        if gamma_2 is None:
            iso2, element2 = _split_species(species2)
            gamma_2 = nmr_gamma(element2, iso=iso2)

        dipolar_coupling = dipolar_coupling_from_distance_vector(
            distance_vector, gamma_1, gamma_2
        )

        # Create the dipolar coupling tensor
        dipolar_tensor = NMRTensor(
            _dip_tensor(dipolar_coupling, distance_vector), order="i"
        )

        # Create the DipolarCoupling object
        return cls(
            site_i=site_i,
            site_j=site_j,
            species1=species1,
            species2=species2,
            tensor=dipolar_tensor,
            tag=tag,
            gamma1=gamma_1,
            gamma2=gamma_2,
        )
    
    def to_mrsimulator(self):
        """
        Convert the DipolarCoupling object to a dictionary suitable for input to MRSimulator.

        Returns:
        --------
        dict
            A dictionary containing the dipolar coupling tensor and the gyromagnetic ratios of the two sites.

        """
        # return {
        #     "dipolar_coupling": self.tensor.to_dict(),
        #     "gamma1": self.gamma1,
        #     "gamma2": self.gamma2,
        # }
        raise NotImplementedError("to_mrsimulator method not implemented for DipolarCoupling class")
    
def dipolar_coupling_from_distance_vector(r: np.ndarray, gamma1, gamma2) -> float:
    """
    Calculate the dipolar coupling constant from the distance vector r and gyromagnetic ratios gamma1 and gamma2.

    Args:
        r: The distance vector between the two sites in the system. Note that this must be in Angstroms.
            Cell periodicity is not taken into account. If you need that, check the `minimum_periodic` function in `soprano.utils`.
        gamma1: The gyromagnetic ratio of the first site.
        gamma2: The gyromagnetic ratio of the second site.

    Returns:
        The dipolar coupling constant in Hz.

    """
    # Calculate the dipolar coupling constant
    return _dip_constant(r * 1e-10, gamma1, gamma2)