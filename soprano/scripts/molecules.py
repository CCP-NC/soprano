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
CLI to extract all molecules in a molecular crystal (in any ASE-readable format) and output the in individual structure files.

"""

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "Dec. 13, 2023"


import logging
import os

import click
import click_log
import numpy as np
from ase import Atoms, io
from ase.data import atomic_numbers

from soprano.data import build_custom_vdw
from soprano.properties.linkage import Molecules
from soprano.scripts.cli_utils import (
    add_options,
    has_CH_bonds,
    keyvalue_parser,
    view,
    viewimages,
)
from soprano.utils import has_cif_labels

# logging
logging.captureWarnings(True)
logger = logging.getLogger("cli")
click_log.basic_config(logger)

HEADER = """
###############################################
Extracting molecules from a molecular crystal
###############################################
"""

def validate_cell(ctx, param, value):
    if value is not None:
        parts = value.split()
        num_parts = len(parts)
        if num_parts not in (1, 3, 6, 9):
            raise click.BadParameter("Cell must have 1, 3, 6, or 9 values.")
        try:
            # Convert all parts to floats
            parts = np.array([float(part) for part in parts])
        except ValueError:
            raise click.BadParameter("All cell values must be valid floats.")
        # If num_parts is 9, reshape to 3x3 matrix
        if num_parts == 9:
            parts = parts.reshape((3, 3))
        if num_parts == 1:
            parts = np.eye(3) * parts[0]
        return parts
    return value

# options for the command line
# TODO - later maybe move to the cli_utils file


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--seedname",
    "-s",
    type=click.STRING,
    default=None,
    help="Seedname for the output files. Defaults to the input filename without its extension.",
)
# output directory - create if it doesn't exist
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for the extracted molecules.",
)
@click.option(
    "--format",
    "-f",
    type=click.STRING,
    default="xyz",
    help="Output file format for the extracted molecules. This can be any format supported by ASE.",
)
# disable file output
@click.option(
    "--no-write",
    is_flag=True,
    help="Disable output of the extracted molecules to files.",
)
# redefine unit cell
@click.option(
    "--cell",
    type=click.STRING,
    callback=validate_cell,
    default=None,
    help="Redefined unit cell for the extracted molecules. Use double-quotes to pass this option in. "
    "For example: ```--cell \"10.0 10.0 15.0\"```. "
    "This can be provides as a single float (-> a cubic cell is created with that lattice constant) or "
    "as three floats (`a b c` -> an orthorhombic cell is created with the given side lengths) or "
    "as six floats (`a b c alpha beta gamma`) or "
    "as nine floats (`a11 a12 a13 a21 a22 a23 a31 a32 a33`) for the full unit cell matrix. "
    "Units are in Angstroms."
)
# center molecules in the cell
@click.option(
    "--center",
    "--centre", # British spelling allowed too...
    "-c",
    is_flag=True,
    help="Center the extracted molecules in the unit cell.",
)
# enforce minimum vacuum
@click.option(
    "--vacuum",
    "--vac",
    type=click.FLOAT,
    default=None,
    help="Enforce a minimum vacuum between the extracted molecules and the cell boundaries. Applied in all directions."
    "Only applies if the `--center` flag is set."
    "See the ASE Atoms.center() documentation for how this works.",
)
# use cell-indices: yes/no?
@click.option(
    "--cell-indices/--no-cell-indices",
    is_flag=True,
    default=True,
    help="Use cell indices when outputting the individual structures. This is useful for molecules that are split across periodic boundaries."
    "default is to use the cell-indices.",
)
# -- Soprano Molecules kwargs:
@click.option(
    "--vdw-set",
    type=click.Choice(["ase", "jmol", "csd"]),
    default="csd",
    help="Set of Van der Waals radii to use. Default is csd S. Alvarez, 2013 (`<https://doi.org/10.1039/C3DT50599E>`_).",
)
@click.option(
    "--vdw-scale",
    type=click.FLOAT,
    default=1.0,
    help="Scaling factor to apply to the base Van der Waals radii values. Values bigger than one make for more tolerant molecules.",
)
@click.option(
    "--default-vdw",
    type=click.FLOAT,
    default=2.0,
    help="Default Van der Waals radius for species for whom no data is available. Default is 2.0 Angstroms.",
)
@click.option(
    "--vdw-custom",
    type=click.STRING,
    callback=keyvalue_parser,
    default="",
    help="A comma-separated list of custom Van der Waals radii to use, overriding the existing ones, expressed as: H:1,C:2 etc. Units are in Angstroms.",
)
# verbosity flag e.g. -v -vv -vvv
@click.option(
    "--verbosity",
    "-v",
    count=True,
    help="Increase verbosity. "
    "Use -v for info, -vv for debug. Defaults to only showing warnings and errors.",
)
@add_options([view])
def splitmols(
    filename,
    seedname,
    output_dir,
    format,
    no_write,
    view,
    cell,
    center,
    vacuum,
    cell_indices,
    vdw_set,
    vdw_scale,
    default_vdw,
    vdw_custom,
    verbosity,
):
    """

    Extract all molecules in a molecular crystal (in any ASE-readable format) and output the in individual structure files.

    """

    # Set up logging
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    # Load the structure
    atoms = io.read(filename, index="-1")

    # works better if first wrapped to unit cell
    # atoms.wrap()

    if not has_CH_bonds(atoms):
        logger.warning(
            "No C-H bonds found in the structure. Are you sure this is a molecular crystal?"
        )

    # log the chosen vdW radii
    vdw_r = build_custom_vdw(vdw_set, vdw_scale, default_vdw, vdw_custom)
    elements = list(set(atoms.get_chemical_symbols()))
    logger.debug(f"Elements in the structure: {elements}")
    logger.debug("Using Van der Waals radii for the bond search:")
    for el in elements:
        logger.debug(f"{el}: {vdw_r[atomic_numbers[el]]}")

    # split into molecules
    molecules = extract_molecules(
        atoms,
        cell_indices,
        vdw_set=vdw_set,
        vdw_scale=vdw_scale,
        default_vdw=default_vdw,
        vdw_custom=vdw_custom,
    )

    Nmols = len(molecules)
    if Nmols == 0:
        err_msg = "No molecules found in the structure"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # log
    logger.info(f"Found {Nmols} molecules")
    for i, mol in enumerate(molecules):
        logger.debug(f"Molecule {i}: {mol.get_chemical_formula()}")

    # redefine unit cell if required
    for i in range(Nmols):
        molecules[i] = redefine_unit_cell(molecules[i], cell, center, vacuum)

    # View the molecules if required
    if view:
        viewimages(molecules, reload_as_molecular=False) # disable reloading as molecular as this is already done

    # write the molecules to individual files
    if seedname is None:
        seedname = os.path.splitext(os.path.basename(filename))[0]
    if not no_write:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        write_molecules(molecules, output_dir, format, seedname)



def write_molecules(molecules, output_dir, format, seedname):
    """
    Write the molecules to individual files.
    """
    Nmols = len(molecules)
    # how many leading zeros do we need?
    Nzeros = int(np.log10(Nmols)) + 1
    for i, mol in enumerate(molecules):
        # make the filename
        filename = f"{seedname}_{i:0{Nzeros}}.{format}"
        logger.info(f"Writing molecule {i} to {filename}")
        # TODO think about what info to include in the file
        # make a copy of the atoms object without the arrays
        mol_temp = Atoms(
            mol.get_chemical_symbols(),
            positions=mol.get_positions(),
            cell=mol.get_cell(),
            pbc=mol.get_pbc(),
        )
        # include labels
        if has_cif_labels(mol):
            mol_temp.set_array("labels", mol.get_array("labels"))

        mol_temp.write(
            os.path.join(output_dir, filename),
            # format=format,
            # TODO - maybe add some more options here?
        )

def extract_molecules(atoms, use_cell_indices, **kwargs):
    mols = Molecules.get(atoms, **kwargs)

    molecules = []
    for mol in mols:
        molecules.append(mol.subset(atoms, use_cell_indices=use_cell_indices))

    return molecules


def redefine_unit_cell(atoms: Atoms,
                       cell: np.ndarray,
                       center=False,
                       vacuum=None):
    """
    Redefine the unit cell of the structure.
    If center is True, the structure is centered in the cell.
    (Centers the atoms in the unit cell, so there is the same
        amount of vacuum on all sides.)
    If vacuum is not None, a minimum vacuum is enforced.
    cell can be a single float (-> a cubic cell is created with that lattice constant) or
    as three floats (`a b c` -> an orthorhombic cell is created with the given side lengths) or
    as six floats (`a b c alpha beta gamma`) or
    a 3x3 np.array defining the full unit cell.
    """
    if cell is not None:
        atoms.set_cell(cell)
        atoms.set_pbc(True)

    if not center and vacuum:
        # TODO: handle these cases - maybe translating COM back to original position after centering + vacuum?
        err_msg = "Cannot enforce a minimum vacuum without centering the molecule. Please set the --center flag."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    if center:
        logger.debug("Centering the atoms in the cell")
        if vacuum:
            logger.debug(f"Enforcing a minimum vacuum of {vacuum} Å")
        atoms.center(vacuum=vacuum)
    return atoms
