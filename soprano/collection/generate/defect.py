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


import itertools
from math import factorial

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

# Internal imports
from soprano import utils
from soprano.data import vdw_radii
from soprano.selection import AtomSelection
from soprano.properties.linkage import Bonds
from soprano.rnd import Random, random_combination, random_product


def defectGen(
    struct,
    defect,
    poisson_r=None,
    avoid_atoms=True,
    vdw_set="csd",
    vdw_scale=1,
    max_attempts=30,
):
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
        avoid_cut = (
            (avoid_vdw + vdw_radii[vdw_set][atomic_numbers[defect]]) * vdw_scale / 2.0
        )

    if poisson_r is None:
        # Random generation
        while True:
            good = False
            attempts = max_attempts

            while not good and attempts > 0:
                # 1. Generate point in fractional coordinates
                fp = Random.random((3,))
                # 2. If required check it against existing atoms
                if avoid_atoms:
                    dx = np.dot(avoid_fpos - fp[None, :], cell)
                    dr = np.linalg.norm(dx, axis=1)
                    good = (dr > avoid_cut).all()
                else:
                    good = True
                attempts -= 1

            if good:
                yield Atoms(defect, scaled_positions=[fp], cell=cell) + struct
            else:
                raise RuntimeError(
                    "Failed to generate a defect obeying"
                    " constraints with the given number of "
                    "attempts."
                )
    else:
        # Use Bridson's algorithm
        if avoid_atoms:
            gen = utils.periodic_bridson(
                cell,
                poisson_r,
                max_attempts=max_attempts,
                prepoints=avoid_fpos,
                prepoints_cuts=avoid_cut,
            )
        else:
            gen = utils.periodic_bridson(cell, poisson_r, max_attempts=max_attempts)

        while True:
            p = next(gen, None)
            if p is None:
                return
            yield Atoms(defect, positions=[p], cell=cell) + struct


def substitutionGen(
        struct,
        subst,
        to_replace=None,
        n=1,
        accept=None,
        max_attempts=0,
        random = False):
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
    |                      Default is None, in which case all structures are
    |                      accepted.
    |   max_attempts (int): maximum number of attempts used to generate a
    |                       random point. Default is 0, in which case all
    |                       possible configurations are generated.
    |   random (bool): if True, the order of the loop over the possible
    |                  configurations will be randomized. Default is False.
    |                  This is useful if limiting the number of configurations
    |                  with max_attempts, to better sample the space of
    |                  possible configurations.
    | Returns:
    |   defectGenerator (generator): an iterator object that yields
    |                                structures with all possible
    |                                substitutions.
    """

    if to_replace is None:
        to_replace = AtomSelection.all(struct)

    # calculate number of possible combinations
    # n! / r! / (n-r)!
    Nreplacing = len(to_replace)
    ncombs = factorial(Nreplacing) / factorial(Nreplacing - n) / factorial(n)
    ncombs = int(ncombs)

    niters = ncombs
    if max_attempts > 0:
        niters = min(niters, max_attempts)

    defconfs = itertools.combinations(to_replace.indices, n)
    elems = np.array(struct.get_chemical_symbols()).astype("<U2")

    nsuccesful = 0
    for iiter in range(ncombs):
        if random:
            dc = random_combination(to_replace.indices, n)
        else:
            dc = next(defconfs)
        dstruct = struct.copy()
        delems = elems.copy()
        delems[list(dc)] = subst
        dstruct.set_chemical_symbols(delems)

        if accept is not None:
            if not accept(dstruct, dc):
                continue

        if iiter < niters:
            nsuccesful += 1
            yield dstruct
        else:
            break


def additionGen(
    struct,
    add,
    to_addition=None,
    n=1,
    add_r=1.2,
    accept=None,
    bonds=None,
    rep_alg_kwargs={},
    random=False,
    max_attempts=0):
    """Generator function to create multiple structures with an atom of a
    given element added in the existing cell. The atoms will be attached to
    the atoms passed in the to_addition selection. If none is passed,
    all atoms will be considered for addition. Multiple atoms can be included, in
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
    |                      Default is None, in which case all structures are
    |                      accepted.
    |   bonds (Bonds): a Bonds object containing the bonds of the structure.
    |                  If not present, they will be computed with default parameters. Default is None.
    |   rep_alg_kwargs (dict): a dictionary of keyword arguments to pass to
    |                          the repulsion algorithm. Default is {}.
    |   random (bool): if True, the order of the loop over the possible
    |                  configurations will be randomized. Default is False. This is crucial if 
    |                  limiting the number of configurations with max_attempts, to better sample the space of
    |                  possible configurations.
    |   max_attempts (int): maximum number of attempts used to generate a
    |                       random point. Default is 0, in which case all
    |                       possible configurations are generated.

    | Returns:
    |   defectGenerator (generator): an iterator object that yields
    |                                structures with all possible additions.
    """

    # Input validation
    if n <= 0:
        raise ValueError("Number of atoms to add must be positive")
    if add_r <= 0:
        raise ValueError("Addition radius must be positive")
    
    # Default to all atoms if no selection is provided
    if to_addition is None:
        to_addition = AtomSelection.all(struct)

    # Empty selection check
    if len(to_addition.indices) == 0:
        raise ValueError("The to_addition selection is empty")
    
    # Check if n is larger than the number of atoms to add to
    if n > len(to_addition.indices):
        raise ValueError(f"Number of atoms to add ({n}) exceeds the number of atoms in the selection ({len(to_addition.indices)})")

    # Compute bonds if not provided
    if bonds is None:
        bonds = Bonds.get(struct)

    cell = struct.get_cell()
    pos = struct.get_positions()

    # Separate bonds by atoms - only consider atoms in the to_addition selection
    atom_bonds = [[] if i in to_addition.indices else None for i in range(len(struct))]
    
    # Process each bond and store the bond vectors
    for b in bonds:
        i, j, cell_shift, bond_length = b
        # Calculate the bond vector
        v = pos[j] - pos[i] + np.dot(cell_shift, cell)
        
        # Store bond vector for first atom if it's in the selection
        if atom_bonds[i] is not None:
            atom_bonds[i].append(v.copy())
            
        # Store negative bond vector for second atom if it's in the selection
        if atom_bonds[j] is not None:
            atom_bonds[j].append(-v.copy())

    # Compute possible attachment points for each atom
    attach_v = [None] * len(struct)
    for i, bset in enumerate(atom_bonds):
        if bset is None:
            continue
            
        if len(bset) == 0:
            # For atoms with no bonds, generate a random normalized direction
            rndv = Random.random((1, 3)) - 0.5
            rndv /= np.linalg.norm(rndv, axis=1)[:, None]
            attach_v[i] = rndv
        else:
            # For atoms with bonds, use the repulsion algorithm to find suitable attachment points
            this_v = utils.rep_alg(bset, **rep_alg_kwargs)
            attach_v[i] = this_v
    
    n_attach_v = len([v for v in attach_v if v is not None])

    # Make sure that the number of attachment points is sufficient
    if n_attach_v < n:
        raise ValueError(f"Not enough attachment points available (found {n_attach_v}, needed {n})")

    # Check if we have any valid attachment points
    if n_attach_v == 0:
        raise ValueError("No atoms available for addition. Check the to_addition selection.")
    
    # Set the number of attempts to try
    attempts_left = max_attempts

    addconfs = itertools.combinations(to_addition.indices, n)

    # Process each configuration
    for ac in addconfs:
        # Skip if we've already tried enough configurations
        if attempts_left <= 0 and max_attempts > 0:
            break
        # Map from the original atom indices to indices in the filtered attach_v array
        chosen_v = [attach_v[iac] for iac in ac]

        # Generate positions for adding atoms
        if random:
            addpos = random_product(*chosen_v)
        else:
            addpos = itertools.product(*chosen_v)
            
        for ap in addpos:
            attempts_left -= 1
            
            # Create the new structure with added atoms
            astruct = struct.copy()
            new_atoms_positions = pos[list(ac)] + np.array(ap) * add_r
            astruct += Atoms(symbols=[add] * n, positions=new_atoms_positions)
            
            # Apply acceptance filter if provided
            if accept is not None:
                if not accept(astruct, ac):
                    continue
                    
            yield astruct
            
            # Exit if we've reached the maximum number of attempts
            if attempts_left <= 0 and max_attempts > 0:
                break
