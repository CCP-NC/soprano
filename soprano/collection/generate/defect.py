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

"""Generator producing structures with a randomly positioned point defect"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import pkgutil
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
# Internal imports
import soprano.utils as utils

# Pre load VdW radii
from ase.data.vdw import vdw_radii as _vdw_radii_ase
_vdw_data = pkgutil.get_data('soprano', 'data/vdw_jmol.json').decode('utf-8')
_vdw_radii_jmol = np.array(json.loads(_vdw_data))

_vdw_radii = {
    'ase': _vdw_radii_ase,
    'jmol': _vdw_radii_jmol
}


def defectGen(struct, defect, poisson_r=None, avoid_atoms=True,
              vdw_set='jmol', vdw_scale=1, max_attempts=30):
    """Generator function to create multiple structures with a defect of a
    given element randomly added to the existing cell. The defects can be
    distributed following a random uniform distribution or a Poisson-sphere
    distribution which guarantees that no two defects are created closer than
    a certain distance. These are created using a variation on Bridson's
    algorithm. For that reason the defects will be created contiguously and
    added in succession. A full cover of all the empty space will be provided
    only by extracting structures until StopIteration is raised.

    | Args:
    |   struct (ase.Atoms): the starting structure. All defects will be added
    |                       to it.
    |   defect (str): element symbol of the defect to add.
    |   poisson_r (float): if this argument is present, the Poisson sphere
    |                      distribution rather than random uniform will be
    |                      used, and this will be the minimum distance in
    |                      Angstroms between the generated defects.
    |   avoid_atoms (bool): if True, defects too close to the existing atoms
    |                       in the cell will be rejected. The cutoff distance
    |                       will be estimated based on the Van der Waals radii
    |                       set of choice. Default is True.
    |   vdw_set({ase, jmol}): set of Van der Waals radii to use. Only relevant
    |                         if avoid_atoms is True. Default is the one
    |                         extracted from JMol.
    |   vdw_scale (float): scaling factor to apply to the base Van der Waals
    |                      radii values. Only relevant if avoid_atoms is True.
    |                      Values bigger than one make for larger empty
    |                      volumes around existing atoms.
    |   max_attempts(int): maximum number of attempts used to generate a
    |                      random point. When using a uniform distribution,
    |                      this is the maximum number of attempts that will be
    |                      done to avoid existing atoms (if avoid_atoms is 
    |                      False, no attempts are needed). When using a Poisson
    |                      sphere distribution, this is a parameter of the
    |                      Bridson-like algorithm and will include also
    |                      avoiding previously generated defects. Default
    |                      is 30.

    | Returns:
    |   defectGenerator (generator): an iterator object that yields
    |                                structures with randomly distributed
    |                                defects.
    """

    # Lattice parameters
    cell = struct.get_cell()

    # Van der Waals radii (if needed)
    if avoid_atoms:
        avoid_fpos = struct.get_scaled_positions()
        avoid_vdw = _vdw_radii[vdw_set][struct.get_atomic_numbers()]
        avoid_cut = (avoid_vdw +
                     _vdw_radii[vdw_set][atomic_numbers[defect]]
                     )*vdw_scale/2.0

    if poisson_r is None:
        # Random generation
        while True:
            good = False
            attempts = max_attempts

            while not good and attempts > 0:
                # 1. Generate point in fractional coordinates
                fp = np.random.random((3,))
                # 2. If required check it against existing atoms
                if avoid_atoms:
                    dx = np.dot(avoid_fpos-fp[None, :], cell)
                    dr = np.linalg.norm(dx, axis=1)
                    good = (dr > avoid_cut).all()
                else:
                    good = True
                attempts -= 1

            if good:
                yield Atoms(defect, scaled_positions=[fp], cell=cell)+struct
            else:
                raise RuntimeError('Failed to generate a defect obeying'
                                   ' constraints with the given number of '
                                   'attempts.')
    else:
        # Use Bridson's algorithm
        if avoid_atoms:
            gen = utils.periodic_bridson(cell, poisson_r,
                                         max_attempts=max_attempts,
                                         prepoints=avoid_fpos,
                                         prepoints_cuts=avoid_cut)
        else:
            gen = utils.periodic_bridson(cell, poisson_r,
                                         max_attempts=max_attempts)

        while True:
            # StopIteration will just bubble up when generated
            p = next(gen)
            yield Atoms(defect, positions=[p], cell=cell) + struct
