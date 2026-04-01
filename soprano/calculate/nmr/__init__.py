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
Classes and functions for simulating NMR spectroscopic results from
structures with simplified algorithms.
"""


from soprano.calculate.nmr.config import DEFAULT_MARKER_SIZE, PlotSettings
from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.export import export_contour_data
from soprano.calculate.nmr.nmr import NMRCalculator, NMRFlags
from soprano.calculate.nmr.plot2d import NMRPlot2D

__all__ = [
	"DEFAULT_MARKER_SIZE",
	"NMRCalculator",
	"NMRData2D",
	"NMRFlags",
	"NMRPlot2D",
	"PlotSettings",
	"export_contour_data",
]
