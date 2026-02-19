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

"""CLI to extract and process NMR-related properties from .magres files.

TODO: add support for different shift {Haeberlen,NQR,IUPAC} and quadrupole
      {Haeberlen,NQR} conventions.
TODO: check if df is too wide to fit in window -- if so, split into multiple plots.
TODO: spinsys output is not yet implemented.
TODO: document config file setup
"""

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "July 08, 2022"


import logging

import click
import click_log

from soprano.nmr.extract import (  # noqa: F401 – re-exported for backwards compat
    EFG_MINIMAL_COLUMNS,
    MS_MINIMAL_COLUMNS,
    NMR_COLUMN_ALIASES,
    build_nmr_df,
    check_equivalent_sites_ms,
    get_efg_summary,
    get_ms_summary,
    label_atoms,
    merge_tagged_sites,
    nmr_extract_atoms,
    nmr_extract_multi,
    tag_functional_groups,
)
from soprano.scripts.cli_utils import (
    NMREXTRACT_OPTIONS,
    add_options,
    print_results,
    viewimages,
)

# logging
logging.captureWarnings(True)
logger = logging.getLogger("cli")
click_log.basic_config(logger)

HEADER = """
##########################################
#  Extracting NMR info from magres file  #
"""
FOOTER = """
# End of NMR info extraction            #
##########################################
"""

@click.command()
# one of more files
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@add_options(NMREXTRACT_OPTIONS)
def nmr(
    files,
    subset=None,
    output=None,
    output_format=None,
    merge=False,
    isotopes={},
    references={},
    gradients={},
    reduce=True,
    average_group=None,
    symprec=1e-4,
    properties=["efg", "ms"],
    precision=3,
    euler_convention="zyz",
    sortby="",
    sort_order="ascending",
    include=None,
    exclude=None,
    query=None,
    view=False,
    verbosity=0,
    ms_tag="ms",
    efg_tag="efg",
):
    """
    Extract and analyse NMR data from magres file(s) or extended XYZ files.

    Usage:
    soprano nmr seedname.magres
    soprano nmr data.xyz --ms-tag pred_ms --efg-tag ref_efg

    Processes .magres file(s) or extended XYZ files containing NMR-related properties
    and prints a summary. It defaults to printing all NMR properties
    present in the file for all the atoms.

    See the below arguments for how to extract specific information.
    """
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    dfs, images = nmr_extract_multi(
        files,
        subset=subset,
        merge=merge,
        isotopes=isotopes,
        references=references,
        gradients=gradients,
        reduce=reduce,
        average_group=average_group,
        symprec=symprec,
        properties=properties,
        euler_convention=euler_convention,
        sortby=sortby,
        sort_order=sort_order,
        include=include,
        exclude=exclude,
        query=query,
        logger=logger,
        ms_tag=ms_tag,
        efg_tag=efg_tag,
    )
    if view:
        viewimages(images)

    # write to file(s)
    print_results(dfs, output, output_format, precision, verbosity > 0)

