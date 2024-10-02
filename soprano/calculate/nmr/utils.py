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
from collections.abc import Iterable
from dataclasses import dataclass
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




def lorentzian(X: np.ndarray, x0: float, Y: np.ndarray, y0: float, x_broadening: float, y_broadening: float) -> np.ndarray:
    """
    Calculate the Lorentzian broadening function.

    .. math::
        f(x, y) = \\frac{1}{((x - x_0) / w_x)^2 + ((y - y_0) / w_y)^2 + 1}

    where :math:`w_x` and :math:`w_y` are the broadening factors in the x and y directions, respectively. These
    correspond to the half-width at half-maximum (HWHM) of the Lorentzian function.

    Args:
        X (np.ndarray): Array of x values.
        x0 (float): x-coordinate of the peak.
        Y (np.ndarray): Array of y values.
        y0 (float): y-coordinate of the peak.
        x_broadening (float): Broadening factor in the x direction.
        y_broadening (float): Broadening factor in the y direction.

    Returns:
        np.ndarray: Array of intensity values.
    """
    return np.exp(-((X - x0)**2 / (2 * x_broadening**2) + (Y - y0)**2 / (2 * y_broadening**2)))

def gaussian(X: np.ndarray, x0: float, Y: np.ndarray, y0: float, x_broadening: float, y_broadening: float) -> np.ndarray:
    """
    Calculate the Gaussian broadening function.

    .. math::
        f(x, y) = \\exp\\left(-\\frac{(x - x_0)^2}{2 w_x^2} - \\frac{(y - y_0)^2}{2 w_y^2}\\right)

    where :math:`w_x` and :math:`w_y` are the broadening factors in the x and y directions, respectively. These
    correspond to the standard deviation of the Gaussian function.

    Args:
        X (np.ndarray): Array of x values.
        x0 (float): x-coordinate of the peak.
        Y (np.ndarray): Array of y values.
        y0 (float): y-coordinate of the peak.
        x_broadening (float): Broadening factor in the x direction.
        y_broadening (float): Broadening factor in the y direction.

    Returns:
        np.ndarray: Array of intensity values.
    """
    return np.exp(-((X - x0)**2 / (2 * x_broadening**2) + (Y - y0)**2 / (2 * y_broadening**2)))

def generate_contour_map(
    peaks: List[Peak2D],
    grid_size: int = 100,
    broadening: str = 'gaussian',
    x_broadening: float = 1.0,
    y_broadening: float = 1.0,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a contour map based on the provided peaks and broadening parameters.

    Args:
        peaks (List[Peak2D]): List of Peak2D objects containing x, y coordinates and correlation strength.
        grid_size (int, optional): Size of the grid for the contour map. Default is 100.
        broadening (str, optional): Type of broadening function to use ('lorentzian' or 'gaussian'). Default is 'lorentzian'.
        x_broadening (float, optional): Broadening factor in the x direction. Default is 1.0.
        y_broadening (float, optional): Broadening factor in the y direction. Default is 1.0.
        xlims (Optional[Tuple[float, float]], optional): Limits for the x-axis. Default is None.
        ylims (Optional[Tuple[float, float]], optional): Limits for the y-axis. Default is None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Meshgrid arrays X, Y and the intensity grid Z.
    """
    broadening = broadening.lower()
    # Create a grid of x, y values
    if xlims is None:
        x_min, x_max = min(peak.x for peak in peaks), max(peak.x for peak in peaks)
    else:
        x_min, x_max = xlims
    if ylims is None:
        y_min, y_max = min(peak.y for peak in peaks), max(peak.y for peak in peaks)
    else:
        y_min, y_max = ylims

    # Create a grid of x, y values, broadening the grid by 5 times the broadening factor
    x = np.linspace(x_min - 5 * x_broadening, x_max + 5 * x_broadening, grid_size)
    y = np.linspace(y_min - 5 * y_broadening, y_max + 5 * y_broadening, grid_size)
    X, Y = np.meshgrid(x, y)

    # Initialize the intensity grid
    Z = np.zeros_like(X)

    # Apply broadening for each peak, adding to the intensity grid
    for peak in peaks:
        x0, y0, strength = peak.x, peak.y, peak.correlation_strength
        if broadening == 'lorentzian':
            Z += strength * lorentzian(X, x0, Y, y0, x_broadening, y_broadening)
        elif broadening == 'gaussian':
            Z += strength * gaussian(X, x0, Y, y0, x_broadening, y_broadening)
        else:
            raise ValueError(f'Unknown broadening function: {broadening}')

    return X, Y, Z







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
            idx_y=idx_y)
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
    unique_peaks = []
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
            unique_peak = peak_map[key]
            unique_xlabels[unique_peak.xlabel].add(peak.xlabel)
            unique_ylabels[unique_peak.ylabel].add(peak.ylabel)
        else:
            unique_peaks.append(peak)
            peak_map[key] = peak
            unique_xlabels[peak.xlabel] = {peak.xlabel}
            unique_ylabels[peak.ylabel] = {peak.ylabel}

    def sort_func(x: str) -> Union[int, str]:
        int_list = re.findall(r'\d+', x)
        return int(int_list[0]) if int_list else x

    for unique_peak in unique_peaks:
        xlabel_list = sorted(unique_xlabels[unique_peak.xlabel], key=sort_func)
        ylabel_list = sorted(unique_ylabels[unique_peak.ylabel], key=sort_func)
        unique_peak.xlabel = '/'.join(xlabel_list)
        unique_peak.ylabel = '/'.join(ylabel_list)

    return unique_peaks

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
