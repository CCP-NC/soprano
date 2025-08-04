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
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import warnings

from soprano.data.nmr import nmr_gamma
from soprano.nmr.tensor import NMRTensor
from soprano.nmr.utils import (
    _anisotropy,
    _asymmetry,
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
    def to_mrsimulator(self, include_angles: bool = True) -> dict[str, dict[str, Union[float, np.ndarray]]]:
        pass

    @abstractmethod
    def to_simpson(self, include_angles: bool = True) -> str:
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
    def J_evals(self) -> NDArray[np.float64]:
        """
        Produces the eigenvalues of the J-coupling tensor for the pair of nuclei
        in the system. The J coupling for a pair of nuclei i and j is defined as:

        .. math::

            J_{ij} = 10^{19}\\frac{h\\gamma_i\\gamma_j}{4\\pi^2}K

        where the gammas represent the gyromagnetic ratios of the nuclei and K is
        the J coupling reduced tensor found in a .magres file, in :math:`10^{19} T^2 J^{-1}`.


        """
        evals = self.tensor.eigenvalues  # eigenvalues of the IScoupling tensor.
        if evals.size != 3:
            raise ValueError("J-coupling tensor must have exactly 3 eigenvalues.")
        return _J_constant(evals, self.gamma1, self.gamma2)

    @property
    def J_eigenvectors(self) -> NDArray[np.float64]:
        """
        The eigenvectors of the J-coupling tensor for the pair of nuclei in the system
        - these are the same as the eigenvectors J coupling reduced tensor found in a .magres file.

        """
        return self.tensor.eigenvectors

    @property
    def Jisotropy(self) -> float:
        """
        The isotropic J-coupling constant for the pair of nuclei in the system.

        """
        jc_evals = self.J_evals
        # Explicitly convert to python float to avoid type issues (.item())
        return np.mean(jc_evals).item()

    @property
    def J_anisotropy(self) -> float:
        """
        The anisotropic J-coupling constant for the pair of nuclei in the system.

        """
        jc_evals = self.J_evals
        return _anisotropy(_haeb_sort([jc_evals]))[0]

    @property
    def J_reduced_anisotropy(self) -> float:
        """
        The reduced anisotropic J-coupling constant for the pair of nuclei in the system.

        """
        jc_evals = self.J_evals
        return _anisotropy(_haeb_sort([jc_evals]), reduced=True)[0]

    @property
    def J_asymmetry(self) -> float:
        """
        The asymmetry parameter for the pair of nuclei in the system.

        """
        jc_evals = self.J_evals
        return _asymmetry(_haeb_sort([jc_evals]))[0]

    @property
    def coupling_constant(self) -> float:
        return self.Jisotropy

    def to_mrsimulator(
            self,
            include_angles: bool = True
            ) -> dict[str, Union[tuple[int, int], float, dict[str, float]]]:
        """
        The MRSimulator format expects the J-coupling tensor 
        as a dict to be something like this:
        {
            site_index=[0, 1],
            isotropic_j=15.0,  # in Hz
            j_symmetric={
                zeta=12.12,  # in Hz
                eta=0.82,
                alpha=2.45,  # in radians
                beta=1.75,  # in radians
                gamma=0.15,  # in radians
            }
        }

        Args:
            include_angles: bool, optional
                If True, include the Euler angles in the output. Default is True.

        Returns:
        -------
        dict
            A dictionary containing the J-coupling tensor parameters for MRSimulator.
        Notes:
        -----
        - The isotropic J-coupling constant is reported in Hz.
        - The J-coupling tensor is represented in a symmetric form with the parameters zeta, eta, alpha, beta, and gamma.
        - The Euler angles are in radians.
        - The site_index is a list of two integers representing the indices of the coupled sites.
        - The zeta parameter is the reduced anisotropic J-coupling constant.
        - The eta parameter is asymmetry parameter of the J-coupling tensor.
        - The alpha, beta, and gamma parameters are the Euler angles of the J-coupling tensor in radians.
        """
        angles = {}
        if include_angles:
            euler_angles = self.euler_angles()
            angles = {
                "alpha": euler_angles[0],
                "beta": euler_angles[1],
                "gamma": euler_angles[2]
            }

        j_symmetric = {
            "zeta": self.J_reduced_anisotropy,
            "eta": self.J_asymmetry,
            **angles
        }

        return {
            "site_index": (self.site_i, self.site_j),
            "isotropic_j": self.Jisotropy,
            "j_symmetric": j_symmetric
        }

    def to_simpson(self, include_angles: bool = True) -> str:
        """
        Convert the J-coupling tensor to the Simpson format.

        The expected format for Simpson is:
        ```
        jcoupling i j Jiso_ij Janiso_ij eta_ij alpha beta gamma
        ```
        where:
        - `i` and `j` are the indices of the coupled sites (1-based indexing).
        - `Jiso_ij` is the isotropic J-coupling constant in Hz.
        - `Janiso_ij` is the reduced anisotropic J-coupling constant in Hz.
        - `eta_ij` is the asymmetry parameter of the J-coupling tensor.
        - `alpha`, `beta`, and `gamma` are the Euler angles of the J-coupling tensor in radians.

        Args:
            include_angles: bool, optional
                If True, include the Euler angles in the output. Default is True.
                If False, the angles will be set to zero.

        Returns:
            str: The J-coupling tensor in the Simpson format.
        """
        EULER_ANGLE_CONVENTION = 'zyz'  # Simpson uses zyz convention for Euler angles
        EULER_ANGLE_IS_PASSIVE = True  # Simpson uses passive convention for Euler angles
        EULER_ANGLE_IS_DEGREES = True  # Simpson uses degrees for Euler angles

        angles = [0.0, 0.0, 0.0]  # Default angles if not included
        if include_angles:
            euler_angles = self.euler_angles(
                convention=EULER_ANGLE_CONVENTION,
                passive=EULER_ANGLE_IS_PASSIVE,
                degrees=EULER_ANGLE_IS_DEGREES
            )
            angles = [euler_angles[0], euler_angles[1], euler_angles[2]]
        # Format the output string
        return (
            f"jcoupling {self.site_i + 1} {self.site_j + 1} "
            f"{self.Jisotropy:.6f} "
            f"{self.J_anisotropy:.6f} "
            f"{self.J_asymmetry:.6f} "
            f"{angles[0]:.6f} {angles[1]:.6f} {angles[2]:.6f}"
        )
# end class ISCoupling

# Alias of ISCoupling: JCoupling ?
JCoupling = ISCoupling


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
    
    def to_mrsimulator(self, include_angles: bool = True) -> dict[str, dict[str, float]]:
        """
        Convert the DipolarCoupling object to a dictionary compatible with MRSimulator.

        This method prepares the dipolar coupling tensor information for simulation,
        including the coupling constant and orientation relative to the principal axis system.

        Notes:
        ------
        - Coupling constant is reported in Hz
        - Euler angles are in the convention used by the MRSimulator library (TODO: double check this!)

        Args:
        ----
        include_angles: bool, optional
            If True, include the Euler angles in the output dictionary. Default is True.

        Returns:
        --------
        dict
            A nested dictionary with dipolar coupling tensor parameters:
            - 'D': Coupling constant (Hz)
            - 'alpha': First Euler rotation angle (radians)
            - 'beta': Second Euler rotation angle (radians)
            - 'gamma': Third Euler rotation angle (radians)

        """
        result = {"dipolar": {"D": self.coupling_constant}}
        if include_angles:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")
                # Get the Euler angles in the convention used by MRSimulator
                euler_angles = self.tensor.euler_angles()
            result["dipolar"]["alpha"] = euler_angles[0]
            result["dipolar"]["beta"] = euler_angles[1]
            result["dipolar"]["gamma"] = euler_angles[2]
        return result
    
    def to_simpson(self, include_angles: bool = True) -> str:
        """
        Convert the DipolarCoupling object to a string compatible with Simpson.

        This method prepares the dipolar coupling tensor information for simulation
        in the Simpson NMR simulation software.

        Args:
        ----
        include_angles: bool, optional
            If True, include the Euler angles in the output string. Default is True.

        Returns:
        --------
        str
            A string representation of the dipolar coupling tensor in the format used by Simpson.

        """
        i, j = self.site_i, self.site_j
        result = f"dipole {i+1} {j+1} {self.coupling_constant * 2 * np.pi:.6f}"

        if include_angles:
            # TODO check simpson convention for euler angles
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")
                # Get the Euler angles in the convention used by Simpson
                euler_angles = self.tensor.euler_angles(convention='zyz', passive=True, degrees=True)
            a, b, c = euler_angles # a should be zero for dipolar couplings in simpson
            result += f" {a:.6f} {b:.6f} {c:.6f}"
        else:
            result += " 0 0 0"

        return result



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






def coupling_list_to_mrsimulator(
        coupling_list: list[Coupling],
        include_dipolar_angles: bool = True,
        include_jcoupling_angles: bool = True,
        ) -> list[dict[str, Any]]:
    """
    Convert a list of Coupling objects to a dictionary of coupling tensors for MRSimulator.

    Args:
        coupling_list: A list of Coupling objects representing inter-site couplings.
        include_dipolar_angles: bool, optional
            If True, include the Euler angles in the output dictionary for dipolar couplings.
            Default is True.
        include_jcoupling_angles: bool, optional
            If True, include the Euler angles in the output dictionary for J couplings.
            Default is True.

    Returns:
        A list of dictionaries.
    Raises:
        ValueError: If multiple coupling tensors of the same type are found 
        for a given pair of sites.
    """
    coupling_tensors = {}

    for coupling in coupling_list:
        site_pair = (coupling.site_i, coupling.site_j)
        # check type
        if coupling.type == "D":
            include_angles = include_dipolar_angles
        elif coupling.type == "J":
            include_angles = include_jcoupling_angles
        else:
            raise ValueError(f"Unsupported coupling type: {coupling.type}")
        new_coupling_dict = coupling.to_mrsimulator(include_angles = include_angles)
        
        # Get the existing dictionary for this site pair, or an empty dict if not exists
        existing_tensor = coupling_tensors.setdefault(site_pair, {})
        
        # Check for key conflicts
        shared_keys = set(new_coupling_dict.keys()) & set(existing_tensor.keys())
        
        if shared_keys:
            raise ValueError(
                f"Multiple coupling tensors with identical keys {shared_keys} "
                f"found for sites {site_pair}"
            )
        
        # Update the tensor dictionary for this site pair
        existing_tensor.update(new_coupling_dict)

    # Convert to list of dictionaries
    coupling_list_out = []
    for key, value in coupling_tensors.items():
        value["site_index"] = list(key)
        coupling_list_out.append(value)


    return coupling_list_out