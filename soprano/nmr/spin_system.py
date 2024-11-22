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

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from soprano.nmr.coupling import Coupling
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
    # Make this optional
    couplings: list[Coupling] = Field(
        default=None,
        description="A list of couplings between sites in the system",
    )

    euler_convention: Literal["zyz", "zxz"] = Field(
        default="zyz",
        description="The convention used for the Euler angles",
    )
    euler_passive: bool = Field(
        default=False,
        description="Whether the Euler angles are passive",
    )
    euler_degrees: bool = Field(
        default=False,
        description="Whether the Euler angles are in degrees. Default is radians (False)",
    )

    @field_validator("sites")
    def check_sites(cls, site):
        if not isinstance(site, Site):
            raise ValueError("Each site must be a Site object")
        return site

    @field_validator("couplings")
    def check_couplings(cls, coupling):
        if not isinstance(coupling, Coupling):
            raise ValueError("Each coupling must be a Coupling object")
        return coupling
