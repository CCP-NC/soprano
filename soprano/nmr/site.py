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

The Site class is a pydantic model representing a single nuclear spin site in a SpinSystem.
It (can) contain:
- the isotope of the nucleus at this site
- a label for this site
- the magnetic shielding tensor at this site, in ppm
- the electric field gradient tensor at this site, in atomic units

In addition, you can set the following options:
- the convention used for the magnetic shielding tensor
- the convention used for the electric field gradient tensor
- the Euler angle conventions
- the Euler angle passive or active
- the Euler angle degrees or radians



This is modelled roughly 
on the MRSimulator code: https://github.com/deepanshs/mrsimulator
"""

import warnings
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, TensorConvention
from soprano.nmr.utils import _split_species


def check_tensor_present(tensor_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            tensor = getattr(self, tensor_name)
            if tensor is not None:
                return func(self, *args, **kwargs)
            return None
        return wrapper
    return decorator

check_magnetic_shielding_tensor = check_tensor_present('ms')
check_efg_tensor = check_tensor_present('efg')

class Site(BaseModel):
    """
    Represents a single nuclear spin site in a SpinSystem.
    """

    isotope: str = Field(
        ...,
        description="The isotope of the nucleus at this site, e.g. '1H', '13C'",
    )
    label: str = Field(
        ...,
        description="A label for this site, e.g. 'H1', 'C2'",
    )
    index: int = Field(
        ...,
        description="The index of this site in the SpinSystem",
    )
    ms: Optional[MagneticShielding] = Field(
        default=None,
        description="The magnetic shielding tensor at this site, in ppm",
    )
    efg: Optional[ElectricFieldGradient] = Field(
        default=None,
        description="The electric field gradient tensor at this site, in atomic units",
    )

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator("isotope")
    def validate_isotope(cls, v):
        return cls.validate_species(v)

    def enforce_tensor_conventions(
        self,
        ms_convention: Optional[TensorConvention] = None,
        efg_convention: Optional[TensorConvention] = None,
    ):
        def enforce_convention(tensor, convention):
            if convention is None:
                return
            input_convention = TensorConvention.from_input(convention)
            current_convention = tensor.order
            if current_convention != input_convention:
                tensor.order = input_convention
                warnings.warn(
                    f"Converted {tensor.__class__.__name__} tensor to {input_convention} convention"
                )

        enforce_convention(self.ms, ms_convention)
        enforce_convention(self.efg, efg_convention)

    @staticmethod
    def validate_species(species: str) -> str:
        """
        Validate the given species string.

        Parameters:
        species (str): The isotope string to validate, e.g., '1H', '13C'.

        Returns:
        str: The validated species string.

        Raises:
        ValueError: If the species string is not valid.
        """
        try:
            isotope_number, element = _split_species(species)
        except ValueError as e:
            raise ValueError(f"Error processing species ('{species}'): {e}")
        return species

    @property
    @check_magnetic_shielding_tensor
    def ms_iso(self):
        return self.ms.isotropy

    @property
    @check_magnetic_shielding_tensor
    def ms_aniso(self):
        return self.ms.anisotropy


    @property
    def is_quadrupole_active(self):
        if self.efg is not None:
            return self.efg.is_quadrupole_active
        return None

    @check_magnetic_shielding_tensor
    def ms_euler(self, **kwargs):
        """
        Return the Euler angles for the magnetic shielding tensor.

        Parameters:
        **kwargs: Additional keyword arguments that are passed to the
                  euler_angles method of the MagneticShielding tensor object.
                  These can include:
                  - convention (str): The convention used for the Euler angles: zyz or zxz.
                  - passive (bool): Whether the Euler angles are passive or active.
                  - degrees (bool): Whether the Euler angles are in degrees or radians.

        Returns:
        tuple: The Euler angles (alpha, beta, gamma) for the magnetic shielding tensor.
        """
        return self.ms.euler_angles(**kwargs)

    @check_efg_tensor
    def efg_euler(self, **kwargs):
        """
        Return the Euler angles for the electric field gradient tensor.

        Parameters:
        **kwargs: Arbitrary keyword arguments passed to the euler_angles method of the tensor object.
                  These can include:
                  - convention (str): The convention to use for the Euler angles.
                  - passive (boo): Whether the Euler angles are passive or active.
                  - degrees (bool): Whether to return the angles in degrees or radians.

        Returns:
        tuple: The Euler angles (alpha, beta, gamma) for the electric field gradient tensor.
        """
        return self.efg.euler_angles(**kwargs)
    


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Site':
        """
        Create a Site instance from a dictionary representation.

        Parameters:
        data (dict[str, Any]): A dictionary containing Site configuration.
                                Can include nested dictionaries for ms and efg.

        Returns:
        Site: A new Site instance created from the input dictionary.
        """
        # Create a copy of the data to avoid modifying the original
        data_copy = data.copy()

        # Handle magnetic shielding tensor
        if 'ms' in data_copy and isinstance(data_copy['ms'], dict):
            ms_data = data_copy.pop('ms')
            data_copy['ms'] = MagneticShielding(**ms_data)

        # Handle electric field gradient tensor
        if 'efg' in data_copy and isinstance(data_copy['efg'], dict):
            efg_data = data_copy.pop('efg')
            data_copy['efg'] = ElectricFieldGradient(**efg_data)

        # Create and return the Site instance
        return cls(**data_copy)


    def copy(self, deep: bool = True, **kwargs) -> 'Site':
        """
        Create a copy of the Site object.
        
        This method overrides the deprecated copy method with model_copy.
        
        Parameters:
        -----------
        deep : bool, optional
            If True, performs a deep copy of nested objects. 
            Default is True to ensure tensor objects are properly copied.
        **kwargs : dict
            Additional keyword arguments to pass to model_copy
        
        Returns:
        --------
        Site
            A new Site object that is a copy of the current object
        
        Notes:
        ------
        Deprecation warning for the original copy method is automatically 
        handled by Pydantic.
        """
        return self.model_copy(deep=deep, **kwargs)

    # Nice representation of the Site object
    def __repr__(self):
        return (f"Site(label={self.label!r}, "
                f"index={self.index!r}, "
                f"isotope={self.isotope!r}, "
                f"efg={self.efg!r}, "
                f"ms={self.ms!r})")

    def __str__(self):
        return (f"Site: {self.label} (isotope: {self.isotope}, index = {self.index})\n"
                "=============================================\n"
                f"{self.ms}\n"
                f"{self.efg}")
    
    def __eq__(self, other: 'Site') -> bool:
        """
        Check if two Site objects are equal.
        
        Sites are considered equal if they have:
        - The same isotope
        - The same label
        - Equivalent magnetic shielding tensor (if present)
        - Equivalent electric field gradient tensor (if present)
        
        Parameters:
        other (Site): Another Site object to compare against
        
        Returns:
        bool: True if sites are equivalent, False otherwise
        """
        # Quick type and identity checks
        if not isinstance(other, Site):
            return False
        
        # Compare basic attributes
        if (self.isotope != other.isotope or 
            self.label != other.label or 
            self.index != other.index):
            return False
        
        # Compare magnetic shielding tensors
        if (self.ms is None) != (other.ms is None):
            return False
        if self.ms is not None and other.ms is not None:
            if self.ms != other.ms:
                return False
        
        # Compare electric field gradient tensors
        if (self.efg is None) != (other.efg is None):
            return False
        if self.efg is not None and other.efg is not None:
            if self.efg != other.efg:
                return False
        
        return True

    def __hash__(self) -> int:
        """
        Generate a hash for the Site object.

        This allows Site objects to be used in sets and as dictionary keys.

        Returns:
        int: A hash value for the Site object
        """
        return hash((
            self.isotope,
            self.label,
            self.index,
            self.ms if self.ms is None else hash(self.ms), 
            self.efg if self.efg is None else hash(self.efg)
        ))

    # INTERFACE METHODS

    def to_mrsimulator(self):
        """
        Convert the Site object to a dictionary representation compatible with MRSimulator.

        Returns:
        dict: A dictionary representation of the Site object.
        """
        data = {
            'isotope': self.isotope,
            'label': self.label,
        }
        if self.ms is not None:
            isotropic_chemical_shift = self.ms.isotropy
            # TODO: what angle conventions are used?
            euler_angles = self.ms.euler_angles()
            shielding_symmetric = {
                "zeta": self.ms.reduced_anisotropy, # zeta = lambda_c^(s) = delta_2 - delta_iso
                "eta": self.ms.asymmetry,
                "alpha": euler_angles[0],
                "beta": euler_angles[1],
                "gamma": euler_angles[2],}
            data['isotropic_chemical_shift'] = isotropic_chemical_shift
            data['shielding_symmetric'] = shielding_symmetric

        if self.efg is not None and self.is_quadrupole_active:
            Cq = self.efg.Cq
            eta = self.efg.asymmetry
            euler_angles = self.efg.euler_angles()
            data['quadrupolar'] = {
                "Cq": Cq,
                "eta": eta,
                "alpha": euler_angles[0],
                "beta": euler_angles[1],
                "gamma": euler_angles[2],
            }

        return data

    def to_simpson(self):
        """
        Convert the Site object to a dictionary representation compatible with Simpson.

        Returns:
        dict: A dictionary representation of the Site object.
        """

        ms_block = ""
        if self.ms is not None:
            # TODO check simpson euler angle convention here
            euler_angles = self.ms.euler_angles(convention='zyz', passive=True, degrees=True)
            ms_block = f"shift {self.index + 1} {self.ms.isotropy}p {self.ms.reduced_anisotropy}p {self.ms.asymmetry} {euler_angles[0]} {euler_angles[1]} {euler_angles[2]}"

        efg_block = ""
        if self.efg is not None and self.is_quadrupole_active:
            # TODO check simpson euler angle convention here
            euler_angles = self.efg.euler_angles(convention='zyz', passive=True, degrees=True)
            # TODO check simpson quadrupole order here
            q_order = 2
            efg_block = f"quadrupole {self.index + 1} {q_order} {self.efg.Cq} {self.efg.asymmetry} {euler_angles[0]} {euler_angles[1]} {euler_angles[2]}"

        return (ms_block, efg_block)
