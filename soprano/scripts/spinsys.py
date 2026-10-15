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

"""CLI to extract a spin system from a .magres file for use as input to a spin simulation.

This currectly supports Simpson and MRSimulator. If you would like to contribute or
request a new simulator, please open an issue on the GitHub repository.
"""

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "Nov 15, 2023"


import logging
import os
from pathlib import Path

from ase import Atoms
import click
import click_log
from ase.io import read

from soprano.data.nmr import _get_nmr_data
from soprano.nmr.extract import nmr_extract_atoms
from soprano.nmr.spin_system import SpinSystem
from soprano.properties.nmr import (
    averaged_dipolar_coupling_list,
    averaged_j_coupling_list,
    get_spin_system,
)
from soprano.selection import AtomSelection
from soprano.scripts.cli_utils import SPINSYS_OPTIONS, add_options, viewimages

# logging
logging.captureWarnings(True)
logger = logging.getLogger("cli")
click_log.basic_config(logger)


def _with_averaged_couplings(
    spin_system, source, site_sets, merged, isotopes, selection_i, selection_j,
    include_dip, include_j, expanded,
):
    """Add couplings averaged over each functional group's motion.

    The averaging runs over the group's atoms in the source structure, so it
    does not depend on which atom survived the merge.
    """
    sel_i = sel_j = None
    if selection_i:
        sel_i = AtomSelection.from_input(merged, selection_i).indices
    if selection_j:
        sel_j = AtomSelection.from_input(merged, selection_j).indices
    couplings = []
    if include_dip:
        couplings += averaged_dipolar_coupling_list(
            source, site_sets, isotopes=isotopes, sel_i=sel_i, sel_j=sel_j
        )
        n_nonaxial = sum(1 for c in couplings if abs(c.tensor.asymmetry) > 0.01)
        if n_nonaxial:
            logger.warning(
                f"{n_nonaxial} averaged dipolar tensors are not axial; the exported "
                "single constant keeps only the largest eigenvalue and drops the "
                "asymmetry."
            )
    if include_j:
        couplings += averaged_j_coupling_list(
            source, site_sets, isotopes=isotopes, sel_i=sel_i, sel_j=sel_j
        )
    if not expanded and max(len(s) for s in site_sets) > 1:
        logger.warning(
            "Averaged groups are exported as single spins with couplings averaged "
            "over the group's motion (fast-exchange limit). Couplings inside a "
            "group are dropped, and a collapsed A_n spin carries 1/n of the "
            "second moment that the n real spins would. Use --expand-group to "
            "keep the spins."
        )
    return SpinSystem(
        sites=spin_system.sites, couplings=list(spin_system.couplings) + couplings
    )


@click.command()
@click.argument("file", nargs=1, type=click.Path(exists=True))
@add_options(SPINSYS_OPTIONS)
def spinsys(
    file,
    output_filename,
    format,
    observed_nucleus,
    subset,
    average_group,
    expand_group,
    reduce,
    mean_merge,
    split,
    include_ms,
    include_efg,
    include_dip,
    include_j,
    ms_isotropic,
    q_order,
    include_cross_terms,
    include_angles,
    include_ms_angles,
    include_efg_angles,
    include_dipolar_angles,
    include_jcoupling_angles,
    selection_i,  # custom coupling selection i
    selection_j,  # custom coupling selection j
    isotopes,
    references,
    gradients,
    verbosity,
    symprec,
    view,
    ms_tag,
    efg_tag,
):
    """Extract a spin system from a .magres file for use as input to a Simpson or MRSimulator simulation."""

    # Set up logging
    if verbosity == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    # Load the input file
    supported_extensions = {'.magres', '.extxyz', '.xyz'}
    file_ext = Path(file).suffix.lower()
    if file_ext not in supported_extensions:
        logger.error(
            f"Unsupported file type '{file_ext}'. "
            f"Supported formats: {', '.join(sorted(supported_extensions))}"
        )
        return
    try:
        atoms = read(file)
    except Exception as e:
        logger.error(f"Error reading file {file}: {e}")
        return
    
    # Make sure the atoms object is a single Atoms object
    if not isinstance(atoms, Atoms):
        logger.error("The file contains multiple structures. Please select one.")
        return

    # Validate that required arrays are present
    missing = []
    if include_ms and not atoms.has(ms_tag):
        missing.append(f"MS array '{ms_tag}' (disable with --no-ms or change tag with --ms-tag)")
    if include_efg and not atoms.has(efg_tag):
        missing.append(f"EFG array '{efg_tag}' (disable with --no-efg or change tag with --efg-tag)")
    if missing:
        logger.error(
            "The input file is missing required data arrays:\n" +
            "\n".join(f"  - {m}" for m in missing)
        )
        return

    # An averaged group is expanded into its spins when couplings are exported
    if expand_group is None:
        expand_group = bool(average_group) and (include_dip or include_j)
        if expand_group:
            logger.info("Expanding averaged groups into spins (couplings exported)")

    # Extract the spin system
    spinsys_atoms, site_sets = nmr_extract_atoms(
        atoms,
        subset=subset,
        reduce=reduce,
        average_group=average_group,
        symprec=symprec,
        mean_merge=mean_merge,
        logger=logger,
        return_site_sets=True,
        expand_groups=expand_group,
    )
    if spinsys_atoms is None:
        return
    # Averaged groups collapse to one site, so couplings must be averaged
    # over the group in the source structure, not read off the merged one.
    average_couplings = bool(average_group) and (include_dip or include_j)

    # 
    coupling_kwargs = {
        "sel_i": selection_i,
        "sel_j": selection_j
    }

    #  --- Isotope list: ---
    if isotopes:
        logger.info("Using custom isotopes dictionary:")
        logger.info(isotopes)
    # Default isotopes dictionary:
    symbols = spinsys_atoms.get_chemical_symbols()
    nmr_data = _get_nmr_data()
    isotopes_dict = {s: nmr_data[s]["iso"] for s in set(symbols)}
    # Overriding isotopes dictionary if given:
    if isotopes:
        isotopes_dict.update(isotopes)

    # --- References: ---
    # If the references is an empty dict or doesn't contain the set(symbols),
    # then raise an error explaining how to set the references
    if (not references or not set(symbols).issubset(references.keys())) and include_ms:
        logger.error(
            f"References dictionary provided ({references}) does not contain all references for all the species" \
            f" in the (sub)system:  {set(symbols)} \n"
            "Please provide a dictionary with the isotopes and their references.\n" \
            "For example:\n" \
            "--references C:170,H:31 \n" 
        )
        return

    # Observed nucleus:
    if observed_nucleus:
        logger.info(f"Using custom observed nucleus: {observed_nucleus}")


    spin_system = get_spin_system(
        spinsys_atoms,
        include_shielding=include_ms,
        include_efg=include_efg,
        include_dipolar=include_dip and not average_couplings,
        include_j=include_j and not average_couplings,
        isotopes=isotopes_dict,
        references=references,
        gradients=gradients,
        coupling_kwargs=coupling_kwargs,
        ms_tag=ms_tag,
        efg_tag=efg_tag,
    )

    if average_couplings:
        spin_system = _with_averaged_couplings(
            spin_system, atoms, site_sets, spinsys_atoms, isotopes_dict,
            selection_i, selection_j, include_dip, include_j, expand_group,
        )

    # Process the angles option
    if include_angles == 'all':
        include_ms_angles = include_efg_angles = include_dipolar_angles = include_jcoupling_angles = True
    elif include_angles == 'none':
        include_ms_angles = include_efg_angles = include_dipolar_angles = include_jcoupling_angles = False

    writer_kwargs = {
        "ms_isotropic": ms_isotropic,
        "include_ms_angles": include_ms_angles,
        "include_efg_angles": include_efg_angles,
        "include_dipolar_angles": include_dipolar_angles,
        "include_jcoupling_angles": include_jcoupling_angles,
        "q_order": q_order,
        "include_cross_terms": include_cross_terms,
        "observed_nucleus": observed_nucleus or None,
    }

    if not split:
        # Write the spin system to a file
        spin_system.write(output_filename, format=format, **writer_kwargs)
        if output_filename and os.path.exists(output_filename):
            logger.info(f"Spin system written to {output_filename}")

        # If view is True, show the spin system
        if view:
            viewimages([spinsys_atoms], reload_as_molecular=True)
    else:
        # Split the spin system into separate files - one for each site. Raise an error if
        # there are more than one species or if there are any couplings
        
        # any couplings?
        if len(spin_system.couplings) > 0:
            logger.error("Cannot split spin system with couplings. Set --split to False or disable the inclusion of dipolar or J-couplings.")
            return
        if len(spin_system.isotope_set) > 1:
            logger.error("Cannot split spin system with multiple species. Set --split to False or reduce your subset to contain only one species.")
            return
        # Split the spin system into separate files - one for each site
        n_sites=  spin_system.n_sites
        # File string formatting - how many leading zeros?
        n_digits = len(str(n_sites))
        for i, site in enumerate(spin_system.sites):
            logger.info(f"Writing spin system {i} of {n_sites}")
            new_site = site.copy()
            new_site.index = 0
            new_spin_system = SpinSystem(sites=[new_site], couplings=[])
            if output_filename:
                fname_base, extension = os.path.splitext(output_filename)
                new_output_filename = f"{fname_base}_{i:0{n_digits}}{extension}"
            else:
                new_output_filename = None
            
            new_spin_system.write(
                new_output_filename,
                format=format,
                **writer_kwargs
            )
            logger.info(f"Spin system written to {new_output_filename}")


    logger.info("Spin system processing completed.")


    return
