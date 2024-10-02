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

"""CLI to plot NMR results from .magres file(s).
TODOs:
- [ ] add stylesheets
- [ ] add dipolar/j-coupling scaling of markers
    - [ ] Basic version done, but maybe check the assumptions made when running with the reduce or average groups options...
- [ ] 1D plots -- basic simulation of 1D NMR?
"""

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "May 09, 2023"


import logging

import click
import click_log
import matplotlib.pyplot as plt
import numpy as np

from soprano.calculate.nmr import NMRCalculator
from soprano.calculate.nmr.nmr import NMRData2D, NMRPlot2D, PlotSettings
from soprano.properties.nmr import *
from soprano.properties.nmr import MSIsotropy
from soprano.scripts.cli_utils import PLOT_OPTIONS, add_options, viewimages
from soprano.scripts.nmr import nmr_extract_multi, print_results
from soprano.selection import AtomSelection

# logging
logging.captureWarnings(True)
logger = logging.getLogger("cli")
click_log.basic_config(logger)


@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
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
    show_markers,
    marker_symbol,
    scale_marker_by,
    max_marker_size,
    marker_color,
    marker_linewidth,
    show_marker_legend,
    show_diagonal,
    show_grid,
    show_connectors,
    show_ticklabels,
    show_heatmap,
    xbroadening,
    ybroadening,
    colormap,
    show_contour,
    contour_levels,
    contour_color,
    contour_linewidth,
    plot_filename,
    plot_shielding,  ## force-plot the shielding even if references are given
    verbosity,
    symprec,
    precision,
    view,
):
    """
    Plot the NMR spectrum from a .magres file.
    """
    # Hard-code a few options for nmr_extract since the user doesn't need to specify them
    properties = ["ms"]  # we at least need to extract the MS
    include = ["MS_shielding"]  # we only need the MS columns for plotting
    if references:
        include.append("MS_shift")
    exclude = None
    merge = True  # if multiple files are given, we have to merge them
    sortby = None
    sort_order = "ascending"
    combine_rule = "mean"

    # set verbosity
    if verbosity == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

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
    )

    if view:
        viewimages(images)

    # write to file(s)
    if verbosity > 0:
        print_results(dfs)
    if len(dfs) > 1:
        logger.warning(
            "More than one dataframe extracted. Only plotting the first one."
        )
    if len(dfs) == 0:
        logger.error("No dataframes extracted. Aborting.")
        return 1
    atoms = images[0]

    if plot_type == "2D":
        if not y_element:
            y_element = x_element

        shift = not plot_shielding if plot_shielding is not None else references != {}

        # Create NMRData2D instance
        nmr_data = NMRData2D(
            atoms=atoms,
            xelement=x_element,
            yelement=y_element,
            rcut=rcut,
            references=references,
            gradients=gradients,
            isotopes=isotopes,
            is_shift=shift,
            include_quadrupolar=False,
            yaxis_order=yaxis_order,
            correlation_strength_metric=scale_marker_by,
        )

        # Define plot settings
        plot_settings = PlotSettings(
            xlim=xlim,
            ylim=ylim,
            show_markers=show_markers,
            marker=marker_symbol,
            max_marker_size=max_marker_size,
            marker_linewidth=marker_linewidth,
            plot_filename=plot_filename,
            show_lines=show_grid,
            show_diagonal=show_diagonal,
            show_connectors=show_connectors,
            show_labels=show_ticklabels,
            show_heatmap=show_heatmap,
            show_contour=show_contour,
            colormap=colormap,
            marker_color=marker_color,
            show_legend=show_marker_legend,
            contour_levels=contour_levels,
            contour_color=contour_color,
            contour_linewidth=contour_linewidth,
            x_broadening=xbroadening,
            y_broadening=ybroadening,

        )

        # Create NMRPlot2D instance
        nmr_plot = NMRPlot2D(
            nmr_data=nmr_data,
            plot_settings=plot_settings,
        )

        # Generate the plot
        fig, ax = nmr_plot.plot()
        # if the user doesn't give an output file name, show the plot using the default matplotlib backend
        if not plot_filename:
            plt.show()
    elif plot_type == "1D":
        shift = not plot_shielding if plot_shielding is not None else references != {}
        sel = AtomSelection.all(atoms)
        element_sel = AtomSelection.from_element(atoms, x_element)
        sel = sel * element_sel
        atoms = sel.subset(atoms)
        # get the NMR calculator
        calc = NMRCalculator(atoms)
        if shift:
            logger.info(f"Setting references: {references}")
            calc.set_reference(ref=references[x_element], element=x_element)
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
        spec, freq = calc.spectrum_1d(
            x_element,
            min_freq=min_iso,
            max_freq=max_iso,
            bins=1001,
            freq_broad=0.05,
            freq_units="ppm",
            # effects=NMRFlags.Q_1_ORIENT,
            use_central=True,
            use_reference=use_reference,
        )
        # plot
        fig, ax = plt.subplots()
        ax.plot(freq, spec)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Intensity")
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
