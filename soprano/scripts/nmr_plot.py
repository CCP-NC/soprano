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

'''CLI to plot NMR results from .magres file(s).

TODO: add stylesheets
TODO: add dipolar/j-coupling scaling of markers
    Basic version done, but maybe check the assumptions made when running with the reduce or average groups options...
TODO: 1D plots -- basic simulation of 1D NMR?
TODO: get plotting working with ./nmr.py. i.e. use that to extract the data and then plot it.
        - Need to deal with clashing options such as -p !
            - Solution is to hard-code some of the options for this case. 
TODO: [x] when query is used, the indexing fails -- fixed 
'''

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "May 09, 2023"


import click
import click_log

import numpy as np
import re
from ase import Atoms
from ase.visualize import view as aseview
from ase.units import Ha, Bohr
from soprano.properties.nmr import *
from soprano.selection import AtomSelection
import itertools
import pandas as pd
from collections import OrderedDict
from soprano.scripts.nmr import nmr_extract, print_results
from soprano.scripts.cli_utils import PLOT_OPTIONS, add_options, viewimages
from soprano.calculate.nmr import NMRCalculator, NMRFlags, Plot2D, DEFAULT_MARKER_SIZE
from soprano.properties.nmr import MSIsotropy
import logging

import matplotlib.pyplot as plt




# logging
logging.captureWarnings(True)
logger = logging.getLogger('cli')
click_log.basic_config(logger)

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))



@add_options(PLOT_OPTIONS)
def plotnmr(
    files,
    isotopes,
    average_group,
    euler_convention,
    references,
    gradients,
    subset,
    reduce,
    query,
    plot_type,
    x_element,
    y_element,
    yaxis_order,
    rcut,
    xlim,
    ylim,
    marker,
    scale_marker_by,
    max_marker_size,
    show_marker_legend,
    show_diagonal,
    show_grid,
    show_connectors,
    show_ticklabels,
    plot_filename,
    plot_shielding, ## force-plot the shielding even if references are given
    verbosity,
    symprec,
    precision,
    view
        ):
    '''
    Plot the NMR spectrum from a .magres file.
    '''
    # Hard-code a few options for nmr_extract since the user doesn't need to specify them
    properties = ['ms'] # we at least need to extract the MS
    include = ['MS_shielding'] # we only need the MS columns for plotting
    if references:
        include.append('MS_shift')
    exclude = None
    merge = True # if multiple files are given, we have to merge them
    sortby = None
    sort_order = 'ascending'
    combine_rule = 'mean'


    # set verbosity
    if verbosity == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    dfs, images = nmr_extract(
                    files,
                    subset = subset,
                    merge = merge,
                    isotopes = isotopes,
                    references = references,
                    gradients = gradients,
                    reduce = reduce,
                    average_group = average_group,
                    symprec = symprec,
                    properties = properties,
                    euler_convention = euler_convention,
                    sortby = sortby,
                    sort_order = sort_order,
                    include = include,
                    exclude = exclude,
                    query = query,
                    logger = logger,
    )

    if view:
        viewimages(images)

    # write to file(s)
    if verbosity > 0:
        print_results(dfs)
    if len(dfs) > 1:
        logger.warning("More than one dataframe extracted. Only plotting the first one.")
    if len(dfs) == 0:
        logger.error("No dataframes extracted. Aborting.")
        return 1
    atoms = images[0]
    


    
    if plot_type == '2D':
        if not y_element:
            y_element = x_element

        shift = not plot_shielding if plot_shielding is not None else references != {}
        
        plot = Plot2D(
                atoms,
                xelement = x_element,
                yelement = y_element,
                rcut = rcut,
                references=references,
                gradients=gradients,
                isotopes=isotopes,
                plot_shifts=shift,
                include_quadrupolar=False,
                yaxis_order=yaxis_order,
                xlim=xlim,
                ylim=ylim,
                marker=marker,
                max_marker_size=max_marker_size,
                plot_filename=plot_filename,
                scale_marker_by=scale_marker_by,
                show_lines = show_grid,
                show_diagonal = show_diagonal,
                show_connectors=show_connectors,
                show_labels = show_ticklabels,
                marker_color = 'C0',
                show_marker_legend=show_marker_legend,
                logger = logger)
        
        fig, ax = plot.plot()
        # if the user doesn't give an output file name, show the plot using the default matplotlib backend
        if not plot_filename:
            plt.show()
    elif plot_type == '1D':
        shift = not plot_shielding if plot_shielding is not None else references != {}
        sel = AtomSelection.all(atoms)
        element_sel = AtomSelection.from_element(atoms, x_element)
        sel = sel * element_sel
        atoms = sel.subset(atoms)
        # get the NMR calculator
        calc = NMRCalculator(atoms)
        if shift:
            logger.info(f"Setting references: {references}")
            calc.set_reference(ref = references[x_element], element=x_element)
            use_reference = True
            xlabel = f"{x_element} shift (ppm)"
        else:
            xlabel = f"{x_element} shielding (ppm)"
            use_reference = False

        if isotopes:
            calc.set_isotopes(isotopes)
        calc.set_powder(N=8)

        iso = MSIsotropy.get(atoms)
        max_iso = np.max(iso)
        min_iso = np.min(iso)
        iso_range = max_iso - min_iso
        max_iso += iso_range * 0.1
        min_iso -= iso_range * 0.1
        # get 1D plot data
        spec, freq = calc.spectrum_1d(x_element,
                                    min_freq=min_iso,
                                    max_freq=max_iso,
                                    bins=1001,
                                    freq_broad=1.5,
                                    freq_units="ppm",
                                    effects=NMRFlags.Q_1_ORIENT,
                                    use_central=True,
                                    use_reference=use_reference)
        # plot
        fig, ax = plt.subplots()
        ax.plot(freq, spec)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Intensity')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if use_reference:
            ax.invert_xaxis()
        if plot_filename:
            fig.savefig(plot_filename)
        else:
            plt.show()
    else:
        logger.error("Invalid plot type. Aborting.")

    return 0

