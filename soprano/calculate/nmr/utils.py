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

import numpy as np

def get_force_matrix(
            positions: np.array,
            positions_original: np.array,
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
    
    Fmat[off_diag_mask] += C * displacement_matrix[off_diag_mask] / (np.abs(displacement_matrix[off_diag_mask]))**3

    # if any off-diagonal elements are zero, then set a random force
    zero_off_diag_mask = np.logical_and(off_diag_mask, ~non_zero_displacements)
    # count the number of zero displacements
    if np.sum(zero_off_diag_mask) > 0:
        print(np.sum(zero_off_diag_mask))
        Fmat[zero_off_diag_mask] = np.random.randn(np.sum(zero_off_diag_mask)) * C * (1/0.005**2)
    # spring force from original position
    Fmat[diag_mask] = -k * (positions - positions_original)**2
    return Fmat


def get_total_forces(
            positions: np.array,
            positions_original: np.array,
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
    # convert to numpy array
    positions = np.array(positions)
    # store original order
    order =  np.argsort(positions)
    # sort positions 
    positions = positions[order]
    ## deep copy of the sorted positions
    positions_original = positions.copy()

    for iter in range(max_iters):
        # get all forces
        forces = get_total_forces(positions, positions_original, C, k)
        # add forces to all y positions except the end points
        positions[1:-1] += forces[1:-1]
        if np.all(np.abs(forces[1:-1]) < ftol):
            break
    # print(f'Ran for {iter} iterations')
    # return to original order
    positions[order] = positions
    return positions

class Peak:
    '''
    Class to hold peak data. This is used to store the peak data for a correlation peak in a 2D NMR spectrum.

    The data stored includes:
    - the peak position in x and y
    - the correlation strength
    - the x and y labels for the peak
    - the color of the peak

    '''  

    def __init__(self, x, y, correlation_strength, xlabel, ylabel, color='C0'):
        self.x = x
        self.y = y
        self.correlation_strength = correlation_strength
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color = color

    def __repr__(self):
        return f'Peak({self.x}, {self.y}, {self.correlation_strength}, {self.xlabel}, {self.ylabel}, {self.color})'
    
    def __str__(self):
        return f'Peak({self.x}, {self.y}, {self.correlation_strength}, {self.xlabel}, {self.ylabel}, {self.color})'
    
    def equivalent_to(self, other, xtol=0.005, ytol=0.005, corr_tol=0.1, ignore_correlation_strength=False):
        '''
        Check if two peaks are equivalent. We compare the x and y coordinates and the correlation strength.

        Args:
            other (Peak): The other peak to compare to
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

        



