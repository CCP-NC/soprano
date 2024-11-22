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
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor
from soprano.nmr.utils import _split_species

TensorConvention = Literal["Haeberlen", "Increasing", "Decreasing", "NQR"]
tensor_convention_mapping = {
    "Haeberlen": NMRTensor.ORDER_HAEBERLEN,
    "Increasing": NMRTensor.ORDER_INCREASING,
    "Decreasing": NMRTensor.ORDER_DECREASING,
    "NQR": NMRTensor.ORDER_NQR,
}


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
    magnetic_shielding_tensor: MagneticShielding = Field(
        ...,
        description="The magnetic shielding tensor at this site, in ppm",
    )
    ms_tensor_convention: TensorConvention = Field(
        default="Haeberlen",
        description="The convention used for the magnetic shielding tensor",
    )
    efg_tensor: ElectricFieldGradient = Field(
        ...,
        description="The electric field gradient tensor at this site, in atomic units",
    )
    efg_tensor_convention: TensorConvention = Field(
        default="NQR",
        description="The convention used for the electric field gradient tensor",
    )
    # TODO: how should this be used, if at all?
    quadrupolar: bool = Field(
        default=False,
        description="Whether the nucleus at this site has a quadrupolar interaction",
    )
    # Euler angle conventions
    euler_convention: Literal["zyz", "zxz"] = Field(
        default="zyz",
        description="The convention used for the Euler angles" "(default is zyz",
    )
    euler_passive: bool = Field(
        default=False, description="Whether the Euler angles are passive or active. "
    )
    euler_degrees: bool = Field(
        default=False,
        description="Whether the Euler angles are in degrees or radians. "
        "Default is False (radians)",
    )

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator("isotope")
    def isotope_validator(cls, v):
        return cls.validate_species(v)

    # Run this after instantiation to ensure the tensors are valid
    @model_validator(mode="after")
    def enforce_tensor_convention(self):
        # MS
        convention = self.ms_tensor_convention
        convention = tensor_convention_mapping.get(convention)
        tensor = self.magnetic_shielding_tensor
        if convention and tensor:
            current_convention = tensor.order
            if current_convention != convention:
                tensor.order = convention
                warnings.warn(
                    f"Converted magnetic shielding tensor to {convention} convention"
                )
                # update the value in the dict
                self.magnetic_shielding_tensor = tensor
        #  EFG
        convention = self.efg_tensor_convention
        convention = tensor_convention_mapping.get(convention)
        tensor = self.efg_tensor
        if convention and tensor:
            current_convention = tensor.order
            if current_convention != convention:
                tensor.order = convention
                warnings.warn(f"Converted EFG tensor to {convention} ordering")
                # update the value in the dict
                self.efg_tensor = tensor
        return self

    @staticmethod
    def validate_species(species: str) -> str:
        try:
            isotope_number, element = _split_species(species)
        except ValueError as e:
            raise ValueError(f"Error processing species ('{species}'): {e}")
        return species

    @property
    def ms_iso(self):
        return self.magnetic_shielding_tensor.isotropy

    @property
    def ms_aniso(self):
        return self.magnetic_shielding_tensor.anisotropy
   
    def ms_euler(self, **kwargs):
        """
        Return the Euler angles for the magnetic shielding tensor.
        kwargs are passed to the euler_angles method of the tensor object.
        """
        return self.magnetic_shielding_tensor.euler_angles(**kwargs)
    
    def efg_euler(self, **kwargs):
        """
        Return the Euler angles for the electric field gradient tensor.
        kwargs are passed to the euler_angles method of the tensor object.
        """
        return self.efg_tensor.euler_angles(**kwargs)
