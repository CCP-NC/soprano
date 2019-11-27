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
import itertools
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
# Internal imports
import soprano.utils as utils
from soprano.selection import AtomSelection
from soprano.properties.linkage import Bonds
from soprano.data import vdw_radii


def defectGen(struct, defect, poisson_r=None, avoid_atoms=True,
              vdw_set='csd', vdw_scale=1, max_attempts=30):
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
    |   vdw_set({ase, jmolm csd}): set of Van der Waals radii to use. Only 
    |                              relevant if avoid_atoms is True. 
    |                              Default is csd [S. Alvarez, 2013].
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
        avoid_vdw = vdw_radii[vdw_set][struct.get_atomic_numbers()]
        avoid_cut = (avoid_vdw +
                     vdw_radii[vdw_set][atomic_numbers[defect]]
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
            p = next(gen, None)
            if p is None:
                return
            yield Atoms(defect, positions=[p], cell=cell) + struct


def substitutionGen(struct, subst, to_replace=None, n=1, accept=None):
    """Generator function to create multiple structures with a defect of a
    given element substituted in the existing cell. The defects will be put in
    place of the atoms passed in the to_replace selection. If none is passed,
    all atoms will be replaced in turn. Multiple defects can be included, in
    which case all permutations will be generated.
    It is also possible to reject some configurations based on the output of a
    filter function.

    | Args:
    |   struct (ase.Atoms): the starting structure. All defects will be added
    |                       to it.
    |   subst (str): element symbol of the defect to add.
    |   to_replace (AtomSelection): if present, only atoms belonging to this
    |                               selection will be substituted.
    |   n (int): number of defects to include in each structure. Default is 1.
    |   accept (function): a function that determines whether a generated
    |                      structure should be accepted or rejected. Takes as
    |                      input the generated structure and a tuple of the
    |                      indices of the substituted atoms, and must return a
    |                      bool. If False, the structure will be rejected.
    | Returns:
    |   defectGenerator (generator): an iterator object that yields
    |                                structures with all possible
    |                                substitutions.
    """

    if to_replace is None:
        to_replace = AtomSelection.all(struct)

    defconfs = itertools.combinations(to_replace.indices, n)
    elems = np.array(struct.get_chemical_symbols()).astype('<U2')

    for dc in defconfs:
        dstruct = struct.copy()
        delems = elems.copy()
        delems[list(dc)] = subst
        dstruct.set_chemical_symbols(delems)

        if accept is not None:
            if not accept(dstruct, dc):
                continue

        yield dstruct


def additionGen(struct, add, to_addition=None, n=1, add_r=1.2,
                accept=None):
    """Generator function to create multiple structures with an atom of a
    given element added in the existing cell. The atoms will be attached to 
    the atoms passed in the to_addition selection. If none is passed,
    all atoms will be additioned in turn. Multiple defects can be included, in
    which case all permutations will be generated. The algorithm will try
    adding the atom in the direction that seems most compatible with all the
    already existing bonds. If multiple directions satisfy the condition, they
    will all be tested.
    It is also possible to reject some configurations based on the output of a
    filter function.

    | Args:
    |   struct (ase.Atoms): the starting structure. All atoms will be added
    |                       to it.
    |   add (str): element symbol of the atom to add.
    |   to_addition (AtomSelection): if present, only atoms belonging to this
    |                                selection will get an addition.
    |   n (int): number of new atoms to include in each structure. Default
    |            is 1.
    |   add_r (float): distance, in Angstroms, at which to add the atoms.
    |                  Default is 1.2 Ang
    |   accept (function): a function that determines whether a generated
    |                      structure should be accepted or rejected. Takes as
    |                      input the generated structure and a tuple of
    |                      the indices of the atoms to which the new atoms
    |                      were added, and must return a bool. The newly added
    |                      atoms will always be the last n of the structure.
    |                      If False, the structure will be rejected.
    | Returns:
    |   defectGenerator (generator): an iterator object that yields
    |                                structures with all possible additions.
    """

    if to_addition is None:
        to_addition = AtomSelection.all(struct)

    # Compute bonds
    bonds = Bonds.get(struct)

    cell = struct.get_cell()
    pos = struct.get_positions()

    # Separate bonds by atoms
    atom_bonds = [[] if i in to_addition.indices else None
                  for i in range(len(struct))]
    for b in bonds:
        v = pos[b[1]]-pos[b[0]]+np.dot(b[2], cell)
        try:
            atom_bonds[b[0]].append(v.copy())
        except AttributeError:
            pass
        try:
            atom_bonds[b[1]].append(-v.copy())
        except AttributeError:
            pass

    # Compute possible attachment points for each atom
    attach_v = [None]*len(struct)
    for i, bset in enumerate(atom_bonds):
        if bset is None:
            continue
        if len(bset) == 0:
            rndv = np.random.random((1, 3))-0.5
            rndv /= np.linalg.norm(rndv, axis=1)[:, None]
            attach_v[i] = rndv
        else:
            attach_v[i] = utils.rep_alg(bset)
    attach_v = np.array(attach_v)

    addconfs = itertools.combinations(to_addition.indices, n)

    for ac in addconfs:
        addpos = itertools.product(*attach_v[list(ac)])
        for ap in addpos:
            astruct = struct.copy()
            astruct += Atoms(add*n, positions=pos[list(ac)]
                             + np.array(ap)*add_r)

            if accept is not None:
                if not accept(astruct, ac):
                    continue

            yield astruct
