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
Helper functions for NMR calculations and plotting
"""

import itertools
import logging
import re
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass, replace as dc_replace
from functools import wraps
from importlib.resources import files
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from scipy.optimize import minimize

from soprano.properties.labeling.labeling import MagresViewLabels
from soprano.properties.nmr.dipolar import DipolarCoupling
from soprano.properties.nmr.isc import JCIsotropy
from soprano.utils import has_cif_labels


def get_force_matrix(
            positions: np.ndarray,
            positions_original: np.ndarray,
            C:float = 0.01,
            k:float = 0.00001):
    '''
    This is a vectorised version of the above function, which is much faster

    Parameters
    ----------
    positions : np.array
        The y coordinates of the annotations
    positions_original : np.array
        The original y coordinates of the annotations
    C : float, optional
        The repulsive force constant between annotations, by default 0.01
    k : float, optional
        The spring force constant for the spring holding the annotation to it's original position, by default 0.00001

    '''
    Fmat = np.zeros((len(positions), len(positions)))
    # force from other annotations -- ignore the diagonal elements
    displacement_matrix = positions[:, np.newaxis] - positions[np.newaxis, :]
    diag_mask = np.eye(displacement_matrix.shape[0],dtype=bool)
    off_diag_mask = ~diag_mask

    # for any non-zero displacements, calculate the repulsive force
    non_zero_displacements = np.abs(displacement_matrix) > 1e-8
    off_diag_mask = np.logical_and(off_diag_mask, non_zero_displacements)
    # Fmat[off_diag_mask][non_zero_displacements] = -C * displacement_matrix[off_diag_mask][non_zero_displacements] / (np.abs(displacement_matrix[off_diag_mask][non_zero_displacements]))**3

    Fmat[off_diag_mask] += C * displacement_matrix[off_diag_mask] / (np.abs(displacement_matrix[off_diag_mask]))**4

    # if any off-diagonal elements are zero, then set a random force
    zero_off_diag_mask = np.logical_and(off_diag_mask, ~non_zero_displacements)
    # count the number of zero displacements
    if np.sum(zero_off_diag_mask) > 0:
        Fmat[zero_off_diag_mask] = np.random.randn(np.sum(zero_off_diag_mask)) * C * (1/0.05**2)
    # spring force from original position
    Fmat[diag_mask] = -2*k * (positions - positions_original)
    return Fmat

def get_total_forces(
            positions: np.ndarray,
            positions_original: np.ndarray,
            C:float = 0.01,
            k:float = 0.00001):
    '''
    Parameters
    ----------
    positions : np.array
        The y coordinates of the annotations
    positions_original : np.array
        The original y coordinates of the annotations
    C : float, optional
        The repulsive force constant between annotations, by default 0.01
    k : float, optional
        The spring force constant for the spring holding the annotation to it's original position, by default 0.00001
    '''
    Fmat = get_force_matrix(positions, positions_original, C, k)
    # sum over the columns to get the total force on each annotation
    return np.sum(Fmat, axis=1)

def get_total_forces(
            positions: np.ndarray,
            positions_original: np.ndarray,
            C:float = 0.01,
            k:float = 0.00001):
    '''
    Parameters
    ----------
    positions : np.array
        The y coordinates of the annotations
    positions_original : np.array
        The original y coordinates of the annotations
    C : float, optional
        The repulsive force constant between annotations, by default 0.01
    k : float, optional
        The spring force constant for the spring holding the annotation to it's original position, by default 0.00001
    '''
    sign = -1 # trial and error to get the right sign wrt the energy..
    Fmat = sign * get_force_matrix(positions, positions_original, C, k)
    # sum over the columns to get the total force on each annotation
    return np.sum(Fmat, axis=1)
def get_energy(positions, positions_original, C, k):
    '''
    Compute the total energy of the system

    '''
    # compute the energy
    # coulomb energy is the sum of the coulomb energy between each pair of annotations
    # spring energy is the sum of the spring energy for each annotation
    coulomb_energy = 0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if np.abs(positions[i] - positions[j]) > 1e-8:
                coulomb_energy += C / np.abs(positions[i] - positions[j])**2



    spring_energy = 0.5 * k * np.sum((positions - positions_original)**2)
    # print(f'coulomb_energy: {coulomb_energy}, spring_energy: {spring_energy}')
    energy = spring_energy + coulomb_energy
    return energy


def optimise_annotations(positions, max_iters=10000, C=0.01, k=0.00001, ftol = 1e-3):
    '''
    Parameters
    ----------
    positions : list or np.array
        The x or y coordinates of the annotations to be optimised
    max_iters : int, optional
        The maximum number of iterations to run for, by default 10000
    C : float, optional
        The repulsive force constant between annotations, by default 0.01
    k : float, optional
        The spring force constant for the spring holding the annotation to it's original position, by default 0.00001
    ftol : float, optional
        The tolerance for the forces, by default 1e-3. If all the net forces are below this value, the optimisation will stop
        even if the maximum number of iterations has not been reached.

    '''
    if len(positions) < 3:
        return positions
    # convert to numpy array
    positions = np.array(positions)
    # store original order
    order =  np.argsort(positions)
    # invert order
    inverted_order = np.argsort(order)

    # sort positions
    positions = positions[order]
    ## deep copy of the sorted positions
    positions_original = positions.copy()

    # normalise the positions to the range of the original positions
    original_range = np.max(positions_original) - np.min(positions_original)
    min_pos = np.min(positions_original)
    positions = (positions - np.min(positions_original)) / original_range
    positions_original = (positions_original - min_pos) / original_range

    res = minimize(
            get_energy,
            positions,
            args=(positions_original, C, k),
            jac = get_total_forces,
            # method='BFGS',
            options={'disp': False, 'maxiter':max_iters},
            tol=ftol,
            # add a constraint to keep the first and last annotation in the same place
            constraints = [
                {'type': 'eq', 'fun': lambda x: x[0] - positions_original[0]},
                {'type': 'eq', 'fun': lambda x: x[-1] - positions_original[-1]},
                # another set of constraints to keep all values between 0 and 1 in fractional space
                {'type': 'ineq', 'fun': lambda x: x},
                {'type': 'ineq', 'fun': lambda x: 1-x},
            ]
        )
    new_pos_frac = res.x

    # convert back to original range
    new_pos = new_pos_frac * original_range + min_pos
    positions = new_pos





    # return to original order
    positions = positions[inverted_order]
    return positions



@dataclass
class Peak2D:
    '''
    Class to hold peak data. This is used to store the peak data for a correlation peak in a 2D NMR spectrum.

    The data stored includes:
    - the peak position in x and y
    - the correlation strength
    - the x and y labels for the peak
    - the color of the peak
    - the index of the x and y labels in the list of all labels

    '''
    x: float
    y: float
    xlabel: str
    ylabel: str
    correlation_strength: float = 1
    color: str = 'C0'
    idx_x: Optional[int] = None
    idx_y: Optional[int] = None
    multiplicity: int = 1

    def __repr__(self):
        return f'Peak({self.x}, {self.y}, {self.correlation_strength}, {self.xlabel}, {self.ylabel}, {self.color})'

    def equivalent_to(self, other, xtol=0.005, ytol=0.005, corr_tol=0.1, ignore_correlation_strength=False):
        '''
        Check if two peaks are equivalent. We compare the x and y coordinates and the correlation strength.

        Args:
            other (Peak2D): The other peak to compare to
            xtol (float, optional): The tolerance for the x coordinate. Defaults to 0.005.
            ytol (float, optional): The tolerance for the y coordinate. Defaults to 0.005.
            corr_tol (float, optional): The tolerance for the correlation strength. Defaults to 0.1.
        
        Returns:
            bool: True if the peaks are equivalent, False otherwise
        '''
        x_match = np.abs(self.x - other.x) < xtol
        y_match = np.abs(self.y - other.y) < ytol

        if ignore_correlation_strength:
            corr_match = True
        else:
            corr_match = np.abs(self.correlation_strength - other.correlation_strength) < corr_tol

        return x_match and y_match and corr_match




def lorentzian(
    X: np.ndarray,
    x0: float,
    Y: np.ndarray,
    y0: float,
    x_broadening: float,
    y_broadening: float,
    normalise: bool = False,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Calculate a 2D Lorentzian (elliptical) broadening function.

    .. math::
        f(x, y) = \\frac{1}{1 + ((x-x_0)/\\gamma_x)^2 + ((y-y_0)/\\gamma_y)^2}

    where :math:`\\gamma = \\mathrm{FWHM}/2` is the half-width at half-maximum.

    Parameters
    ----------
    X : np.ndarray
        Array of x values (meshgrid).
    x0 : float
        x-coordinate of the peak center.
    Y : np.ndarray
        Array of y values (meshgrid).
    y0 : float
        y-coordinate of the peak center.
    x_broadening : float
        FWHM measured along the x-axis cross-section (i.e. the marginal
        linewidth with y fixed at the peak centre).
        Converted internally to HWHM via ``γ = FWHM / 2``.
    y_broadening : float
        FWHM measured along the y-axis cross-section.  Same convention
        as *x_broadening*.
    normalise : bool, optional
        If True, normalise such that the integral over the plane is 1.
        If False, the peak maximum is 1.
    eps : float, optional
        Small number added to broadenings for numerical stability.

    Returns
    -------
    np.ndarray
        Array of intensity values.
    """
    # Convert FWHM → HWHM (γ)
    wx = x_broadening / 2.0 + eps
    wy = y_broadening / 2.0 + eps

    denom = (
        1.0
        + ((X - x0) / wx) ** 2
        + ((Y - y0) / wy) ** 2
    )

    if normalise:
        prefactor = 1.0 / (np.pi * wx * wy)
        return prefactor / denom

    return 1.0 / denom


_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247


def gaussian(
    X: np.ndarray,
    x0: float,
    Y: np.ndarray,
    y0: float,
    x_broadening: float,
    y_broadening: float,
    normalise: bool = False,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Calculate a 2D Gaussian (elliptical) broadening function.

    .. math::
        f(x, y) = \\exp\\!\\left(
            -\\frac{(x-x_0)^2}{2\\sigma_x^2}
            -\\frac{(y-y_0)^2}{2\\sigma_y^2}
        \\right)

    where :math:`\\sigma = \\mathrm{FWHM}\\,/\,(2\\sqrt{2\\ln 2})`.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid arrays.
    x0, y0 : float
        Peak center.
    x_broadening : float
        FWHM measured along the x-axis cross-section (i.e. the marginal
        linewidth with y fixed at the peak centre).
        Converted internally to σ via ``σ = FWHM / (2√(2 ln 2))``.
    y_broadening : float
        FWHM measured along the y-axis cross-section.  Same convention
        as *x_broadening*.
    normalise : bool
        If True, normalise to unit integral.
        If False, peak maximum is 1.
    eps : float
        Small number added for numerical safety.

    Returns
    -------
    np.ndarray
    """
    # Convert FWHM → σ
    sx = x_broadening * _FWHM_TO_SIGMA + eps
    sy = y_broadening * _FWHM_TO_SIGMA + eps

    exponent = (
        ((X - x0) ** 2) / (2.0 * sx**2)
        + ((Y - y0) ** 2) / (2.0 * sy**2)
    )

    g = np.exp(-exponent)

    if normalise:
        g /= (2.0 * np.pi * sx * sy)

    return g

def generate_contour_map(
    peaks: List[Peak2D],
    grid_size: int = 100,
    broadening: str = 'lorentzian',
    x_broadening: float = 1.0,
    y_broadening: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a contour map based on the provided peaks and broadening parameters.

    The grid extent is always derived from the peak positions plus a
    broadening-dependent padding on each side.  Grid limits are intentionally
    **not** exposed as a parameter: constraining the grid to a sub-range would
    truncate Lorentzian tails from peaks near the edge and produce inconsistent
    intensities.  Display limits should be controlled separately via
    ``PlotSettings.xlim`` / ``PlotSettings.ylim``.

    Padding factors applied beyond the outermost peak:

    * **Gaussian**: 5 × FWHM — the tail at that distance is
      :math:`\\sim 10^{-30}` (machine zero).
    * **Lorentzian**: 50 × FWHM — the tail at that distance is
      :math:`1/(1+100^2) \\approx 0.01\\,\\%`, acceptable for relative-intensity
      comparisons.  5 × FWHM would leave :math:`\\sim 1\\,\\%` residual at the
      boundary, which is non-negligible when multiplicity-weighted peaks are
      compared.

    Args:
        peaks (List[Peak2D]): List of Peak2D objects containing x, y coordinates and correlation strength.
        grid_size (int, optional): Size of the grid for the contour map. Default is 100.
        broadening (str, optional): Type of broadening function to use ('lorentzian' or 'gaussian'). Default is 'gaussian'.
        x_broadening (float, optional): FWHM linewidth in the x direction. Default is 1.0.
        y_broadening (float, optional): FWHM linewidth in the y direction. Default is 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Meshgrid arrays X, Y and the intensity grid Z.
    """
    broadening = broadening.lower()
    # Grid extent always derived from the actual peak positions.
    x_min, x_max = min(peak.x for peak in peaks), max(peak.x for peak in peaks)
    y_min, y_max = min(peak.y for peak in peaks), max(peak.y for peak in peaks)

    # Gaussian tails vanish within 5 × FWHM; Lorentzian tails fall to only
    # ~1 % at 5 × FWHM, so use a much larger pad to avoid edge truncation
    # affecting relative intensities.
    _PAD = 50 if broadening == 'lorentzian' else 5

    x = np.linspace(x_min - _PAD * x_broadening, x_max + _PAD * x_broadening, grid_size)
    y = np.linspace(y_min - _PAD * y_broadening, y_max + _PAD * y_broadening, grid_size)
    X, Y = np.meshgrid(x, y)

    # Initialize the intensity grid
    Z = np.zeros_like(X)

    # Apply broadening for each peak, adding to the intensity grid.
    # Each peak is weighted by correlation_strength × multiplicity so that
    # merged sites (symmetry-equivalent or functional-group averaged) contribute
    # with the correct degeneracy.
    for peak in peaks:
        x0, y0 = peak.x, peak.y
        weight = peak.correlation_strength * peak.multiplicity
        if broadening == 'lorentzian':
            Z += weight * lorentzian(X, x0, Y, y0, x_broadening, y_broadening)
        elif broadening == 'gaussian':
            Z += weight * gaussian(X, x0, Y, y0, x_broadening, y_broadening)
        else:
            raise ValueError(f'Unknown broadening function: {broadening}')

    return X, Y, Z


# ---------------------------------------------------------------------------
# ContourData – lightweight container for a computed 2D contour grid
# ---------------------------------------------------------------------------

ContourData = namedtuple(
    'ContourData',
    [
        'X',              # np.ndarray – meshgrid x coordinates, shape (NI, NP)
        'Y',              # np.ndarray – meshgrid y coordinates, shape (NI, NP)
        'Z',              # np.ndarray – intensity grid, shape (NI, NP)
        'x_broadening',   # float – broadening applied in the x (direct) dimension
        'y_broadening',   # float – broadening applied in the y (indirect) dimension
        'broadening_type', # str – 'gaussian' or 'lorentzian'
        'xlims',          # Tuple[float, float] – (x_min, x_max) of the grid
        'ylims',          # Tuple[float, float] – (y_min, y_max) of the grid
    ],
)
ContourData.__doc__ = (
    "Immutable container for a computed 2D NMR contour grid.\n\n"
    "Fields\n------\n"
    "X, Y : np.ndarray\n    Meshgrid coordinate arrays (shape NI × NP).\n"
    "Z : np.ndarray\n    Intensity grid (shape NI × NP).\n"
    "x_broadening, y_broadening : float\n    FWHM linewidth used in each dimension.\n"
    "broadening_type : str\n    'gaussian' or 'lorentzian'.\n"
    "xlims, ylims : tuple of float\n    Axis limits of the grid.\n"
)





nmr_base_style = str(files("soprano.calculate.nmr").joinpath("soprano_nmr_base.mplstyle"))
nmr_2D_style = str(files("soprano.calculate.nmr").joinpath("soprano_nmr_2D.mplstyle"))

def styled_plot(*style_sheets):
    """Return a decorator that will apply matplotlib style sheets to a plot.
    ``style_sheets`` is a base set of styles, which will be ignored if
    ``no_base_style`` is set in the decorated function arguments.
    The style will further be overwritten by any styles in the ``style``
    optional argument of the decorated function.
    Args:
        style_sheets (:obj:`list`, :obj:`str`, or :obj:`dict`): Any matplotlib
            supported definition of a style sheet. Can be a list of style of
            style sheets.
    """

    def decorator(get_plot):
        @wraps(get_plot)
        def wrapper(*args, fonts=None, style=None, no_base_style=False, **kwargs):

            if no_base_style:
                list_style = []
            else:
                list_style = list(style_sheets)

            if style is not None:
                if isinstance(style, list):
                    list_style += style
                else:
                    list_style += [style]

            if fonts is not None:
                list_style += [{"font.family": "sans-serif", "font.sans-serif": fonts}]

            plt.style.use(list_style)
            return get_plot(*args, **kwargs)

        return wrapper

    return decorator


def get_atom_labels(atoms: Atoms, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Get the labels for the atoms in the Atoms object. If the labels are not present, they will be generated
    using the MagresViewLabels class.

    Args:
        atoms (ase.Atoms): Atoms object.
        logger (Optional[logging.Logger], optional): Logger object. Default is None.

    Returns:
        np.ndarray: Array of labels.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if has_cif_labels(atoms):
        labels = atoms.get_array('labels')
    elif atoms.has('MagresView_labels'):
        # we might have already generated the MV style labels
        labels = atoms.get_array('MagresView_labels')
    else:
        logger.info('No labels found in the atoms object. Generating MagresView-style labels from scratch.')
        labels = MagresViewLabels.get(atoms)
        # convert to numpy array
        labels = np.array(labels, dtype='U25')
    return labels

def prepare_species_labels(isotope, element):
    species_template = r'$\mathrm{^{%s}{%s}}$'
    return species_template % (isotope, element)


def extract_indices(atoms, xelement, yelement):
    idx_x = np.array([atom.index for atom in atoms if atom.symbol == xelement])
    idx_y = np.array([atom.index for atom in atoms if atom.symbol == yelement])

    return idx_x, idx_y


def validate_elements(atoms, xelement, yelement):
    all_elements = atoms.get_chemical_symbols()
    if xelement not in all_elements:
        raise ValueError(f'{xelement} not found in the file after the user-specified filters have been applied.')
    if yelement not in all_elements:
        raise ValueError(f'{yelement} not found in the file after the user-specified filters have been applied.')


def generate_peaks(
    data: List[float],
    pairs: Iterable[Tuple[int, int]],
    labels: List[str],
    markersizes: Union[List[float], float],
    yaxis_order: str,
    xelement: str,
    yelement: str,
    multiplicities: Optional[np.ndarray] = None,
) -> List[Peak2D]:
    """
    Generate peaks for the NMR data.

    Args:
        data (List[float]): The NMR data.
        pairs (Iterable[Tuple[int, int]]): Pairs of indices for the peaks.
        labels (List[str]): Labels for the peaks.
        markersizes (Union[List[float], float]): Marker sizes for the peaks.
        yaxis_order (str): Order of the y-axis.
        xelement (str): Element symbol for the x-axis.
        yelement (str): Element symbol for the y-axis.
        multiplicities (np.ndarray, optional): Per-atom multiplicity array.  When
            provided, each peak's ``multiplicity`` field is set to
            ``multiplicities[idx_x] * multiplicities[idx_y]`` so that merged
            sites (symmetry-equivalent or functional-group averaged) contribute
            with the correct degeneracy weight in the heatmap.

    Returns:
        List[Peak2D]: List of generated peaks.
    """
    peaks = []
    is_single_marker = isinstance(markersizes, float)
    for ipair, (idx_x, idx_y) in enumerate(pairs):

        x = data[idx_x]
        y = data[idx_y]
        strength = markersizes if is_single_marker else markersizes[ipair]
        xlabel = labels[idx_x]
        ylabel = labels[idx_y]

        if multiplicities is not None:
            mult = int(multiplicities[idx_x]) * int(multiplicities[idx_y])
        else:
            mult = 1

        if yaxis_order == '2Q':
            y += x
            if xelement == yelement:
                # then we might have both e.g. H1 + H2 and H2 + H1
                # let's set them both to be H1 + H2 by sorting the labels
                xlabel, ylabel = sorted([xlabel, ylabel])
            ylabel = f'{xlabel} + {ylabel}'
        peak = Peak2D(
            x=x,
            y=y,
            correlation_strength=strength,
            xlabel=xlabel,
            ylabel=ylabel,
            idx_x=idx_x,
            idx_y=idx_y,
            multiplicity=mult)
        peaks.append(peak)
    return peaks

def merge_peaks(
    peaks: Iterable[Peak2D],
    xtol: float = 1e-5,
    ytol: float = 1e-5,
    corr_tol: float = 1e-5,
    ignore_correlation_strength: bool = False
) -> List[Peak2D]:
    """
    Merge peaks that are identical.

    Args:
        peaks (Iterable[Peak2D]): List of peaks to merge.
        xtol (float): Tolerance for x-axis comparison.
        ytol (float): Tolerance for y-axis comparison.
        corr_tol (float): Tolerance for correlation strength comparison.
        ignore_correlation_strength (bool): Whether to ignore correlation strength in comparison.

    Returns:
        List[Peak2D]: List of unique merged peaks.
    """
    # first, get the unique peaks
    # peak_map preserves insertion order (Python 3.7+); when a duplicate key is
    # found, multiplicities are summed so that degenerate site-pairs contribute
    # their full combined weight to the heatmap and marker sizes.
    unique_xlabels: Dict[str, Set[str]] = {}
    unique_ylabels: Dict[str, Set[str]] = {}
    peak_map: Dict[Tuple[float, float, float], Peak2D] = {}

    for peak in peaks:
        key = (
            int(peak.x / xtol),
            int(peak.y / ytol),
            int(peak.correlation_strength / corr_tol) if not ignore_correlation_strength else 0
        )
        if key in peak_map:
            existing = peak_map[key]
            # Accumulate multiplicity: two distinct site-pairs at the same
            # (x, y, strength) are degenerate and their counts should add.
            peak_map[key] = dc_replace(existing,
                                       multiplicity=existing.multiplicity + peak.multiplicity)
            unique_xlabels[existing.xlabel].add(peak.xlabel)
            unique_ylabels[existing.ylabel].add(peak.ylabel)
        else:
            peak_map[key] = peak
            unique_xlabels[peak.xlabel] = {peak.xlabel}
            unique_ylabels[peak.ylabel] = {peak.ylabel}

    def sort_func(x: str) -> Union[int, str]:
        int_list = re.findall(r'\d+', x)
        return int(int_list[0]) if int_list else x

    result = []
    for unique_peak in peak_map.values():
        xlabel_list = sorted(unique_xlabels[unique_peak.xlabel], key=sort_func)
        ylabel_list = sorted(unique_ylabels[unique_peak.ylabel], key=sort_func)
        result.append(dc_replace(unique_peak,
                                 xlabel='/'.join(xlabel_list),
                                 ylabel='/'.join(ylabel_list)))
    return result

def sort_peaks(peaks: List[Peak2D], priority: str = 'x', reverse: bool=False) -> List[Peak2D]:
    '''
    Sort the peaks in the order of increasing x and y values
    TODO: This doesn't work in some cases. Investigate why.
    '''
    if priority == 'x':
        peaks = sorted(peaks, key=lambda x: (x.x, x.y), reverse=reverse)
    elif priority == 'y':
        peaks = sorted(peaks, key=lambda x: (x.y, x.x), reverse=reverse)
    else:
        raise ValueError(f"Unknown priority: {priority}")
    return peaks


def get_pair_dipolar_couplings(
        atoms: Atoms,
        pairs: Iterable[Tuple[int, int]],
        isotopes: Optional[dict[str, int]] = None,
        unit: str = 'kHz'
    ) -> List[float]:
    """
    Get the dipolar couplings for a list of pairs of atoms.
    For pairs where i == j, the dipolar coupling is set to zero.
    
    Parameters
    ----------
    atoms : ASE Atoms object
        The atoms object that contains the atoms.
    pairs : list of tuples
        List of pairs of atom indices.
    isotopes : dict, optional
    
    Returns
    -------
    dipolar_couplings : list of float
        List of dipolar couplings for the pairs.
    """
    if isotopes is None:
        isotopes = {}

    # Define conversion factors
    conversion_factors = {
        'Hz': 1.0,
        'kHz': 1e-3,
        'MHz': 1e-6
    }

    if unit not in conversion_factors:
        raise ValueError(f"Unsupported unit: {unit}. Supported units are: {', '.join(conversion_factors.keys())}")

    conversion_factor = conversion_factors[unit]

    dipolar_couplings = []
    # This is an explicit loop because we need to get the J coupling for each pair
    # - these might not be the same as the set of pairs that we would get by combining indices
    #   {i} and {j} from the pairs list
    for i, j in pairs:
        if i == j:
            # set the dipolar coupling to zero for pairs where i == j
            dipolar_couplings.append(0)
        else:
            coupling = list(DipolarCoupling.get(atoms, sel_i=[i], sel_j=[j], isotopes=isotopes).values())[0][0]
            dipolar_couplings.append(coupling * conversion_factor)


    return dipolar_couplings

def get_pair_j_couplings(
        atoms: Atoms,
        pairs: Iterable[Tuple[int, int]],
        isotopes: Optional[dict[str, int]] = None,
        unit: str = 'Hz',
        tag: str = 'isc'
    ) -> List[float]:
    """
    Get the J couplings for a list of pairs of atoms.
    For pairs where i == j, the J coupling is set to zero.

    Parameters
    ----------
    atoms : ASE Atoms object
        The atoms object that contains the atoms.
    pairs : list of tuples
        List of pairs of atom indices.
    isotopes : dict, optional
        Dictionary of isotopes for the atoms.
    unit : str, optional
        Unit of the J coupling. Default is 'Hz'.
    tag : str, optional
        Name of the J coupling component to return. Default is 'isc'. Magres files
        usually contain isc, isc_spin, isc_fc, isc_orbital_p and isc_orbital_d.

    Returns
    -------
    j_couplings : list of float
        List of J couplings for the pairs.
    """
    if isotopes is None:
        isotopes = {}

    # Define conversion factors
    conversion_factors = {
        'Hz': 1.0,
        'kHz': 1e-3,
        'MHz': 1e-6
    }

    if unit not in conversion_factors:
        raise ValueError(f"Unsupported unit: {unit}. Supported units are: {', '.join(conversion_factors.keys())}")

    conversion_factor = conversion_factors[unit]

    j_couplings = []
    # This is an explicit loop because we need to get the J coupling for each pair
    # - these might not be the same as the set of pairs that we would get by combining indices
    #   {i} and {j} from the pairs list
    for i, j in pairs:
        if i == j:
            # set the J coupling to zero for pairs where i == j
            j_couplings.append(0)
        else:
            coupling = list(JCIsotropy.get(atoms, sel_i=[i], sel_j=[j], tag=tag, isotopes=isotopes).values())[0]
            j_couplings.append(coupling * conversion_factor)

    return j_couplings



def process_pairs(
    idx_x: np.ndarray,
    idx_y: np.ndarray,
    pairs: Optional[List[Tuple[int, int]]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Process pairs of indices and update element indices.

    Args:
        idx_x (np.ndarray): Array of x element indices.
        idx_y (np.ndarray): Array of y element indices.
        pairs (Optional[List[Tuple[int, int]]]): List of tuples representing pairs of indices, or None.

    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray]:
            - List of pairs of indices.
            - List of pairs of element indices.
            - Updated array of x element indices.
            - Updated array of y element indices.
    """
    if pairs:
        pairs_el_idx = []
        for pair in pairs:
            xidx = np.where(idx_x == pair[0])[0][0]  # there should only be one match
            yidx = np.where(idx_y == pair[1])[0][0]  # there should only be one match
            pairs_el_idx.append((xidx, yidx))
    else:
        pairs_el_idx = list(itertools.product(range(len(idx_x)), range(len(idx_y))))
        pairs = list(itertools.product(idx_x, idx_y))

    # Update the indices
    idx_x = np.array(pairs)[:, 0]
    idx_y = np.array(pairs)[:, 1]
    return pairs, pairs_el_idx, idx_x, idx_y

def calculate_distances(pairs: List[Tuple[int, int]], atoms: Atoms) -> np.ndarray:
    """
    Calculate the distances between pairs of atoms.

    Args:
        pairs (List[Tuple[int, int]]): List of tuples representing pairs of atom indices.
        atoms (Atoms): Atoms object that provides the get_distance method.

    Returns:
        np.ndarray: Array of distances corresponding to the pairs.
    """
    pair_distances = np.zeros(len(pairs))
    for i, pair in enumerate(pairs):
        if pair[0] == pair[1]:
            pair_distances[i] = 0.0
        else:
            pair_distances[i] = atoms.get_distance(*pair, mic=True)
    return pair_distances


def filter_pairs_by_distance(
    pairs: List[Tuple[int, int]],
    pairs_el_idx: List[Tuple[int, int]],
    pair_distances: np.ndarray,
    rcut: float
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters pairs of indices based on a cutoff distance.

    Args:
        pairs (List[Tuple[int, int]]): List of tuples representing pairs of indices.
        pairs_el_idx (List[Tuple[int, int]]): List of tuples representing pairs of element indices.
        pair_distances (np.ndarray): Array of distances corresponding to the pairs.
        rcut (float): Cutoff distance for filtering pairs.

    Returns:
        Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
            - Filtered list of pairs.
            - Filtered list of element index pairs.
            - Filtered array of pair distances.
            - Unique sorted array of x indices.
            - Unique sorted array of y indices.

    Raises:
        ValueError: If no pairs are found after filtering by distance.
    """
    dist_mask = np.where(pair_distances <= rcut)[0]
    pairs_el_idx = [pairs_el_idx[i] for i in dist_mask]
    pairs = [pairs[i] for i in dist_mask]
    pair_distances = pair_distances[dist_mask]

    idx_x = np.unique([pair[0] for pair in pairs])
    idx_y = np.unique([pair[1] for pair in pairs])
    if len(idx_x) == 0 or len(idx_y) == 0:
        raise ValueError('No pairs found after filtering by distance. Try increasing the cutoff distance (rcut).')

    idx_x = np.sort(idx_x)
    idx_y = np.sort(idx_y)

    return pairs, pairs_el_idx, pair_distances, idx_x, idx_y
