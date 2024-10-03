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

"""CLI wrapper around the ASE GUI to view structures.

This adds a few options to the ASE GUI, and allows for the use of
Soprano's selection syntax to select atoms to be highlighted/hidden.

It will also check if the material seems to be an organic crystal and, if so, 
'reload' it with the molecule components joined up correctly.

TODO: re-order the operations so that the reload is done first, then the selection.
 - allow tagging by symmetry/labels even when not reducing the structure.

"""

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "Nov 15, 2023"


import logging

import click
import click_log
from ase import Atoms
from ase.io import read

from soprano.scripts.cli_utils import (
    VIEW_OPTIONS,
    add_options,
    reload_as_molecular_crystal,
    viewimages,
)
from soprano.scripts.nmr import nmr_extract_atoms

# logging
logging.captureWarnings(True)
logger = logging.getLogger("cli")
click_log.basic_config(logger)

@click.command()
# one of more files
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
# verbosity flag e.g. -v -vv -vvv
@click.option(
    "--verbosity",
    "-v",
    count=True,
    help="Increase verbosity. "
    "Use -v for info, -vv for debug. Defaults to only showing warnings and errors.",
)
@add_options(VIEW_OPTIONS)
def view(
    files,
    subset,
    reduce,
    average_group,
    symprec,
    verbosity,
):
    """
    Visualise the structure(s) in the given file(s) using the ASE GUI.

    The user can select atoms to be tagged/hidden using the Soprano
    selection syntax.
    """



    # Set up logging
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    images = []
    for f in files:
        # get the atoms objects
        atoms_list = read(f, ":")
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]

        for atoms in atoms_list:
            # check if the material is an organic crystal
            # if so, join up the molecule components
            atoms = reload_as_molecular_crystal(atoms, force=False)
            # get the subset of atoms to be tagged/hidden
            atoms = nmr_extract_atoms(
                atoms,
                subset=subset,
                reduce=reduce,
                average_group=average_group,
                symprec=symprec,
                logger=logger,
            )
            # add the atoms to the list
            images.append(atoms)

    # view the images
    viewimages(images)
