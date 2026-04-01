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
from pathlib import Path

import click
import click_log
import matplotlib.pyplot as plt
import numpy as np

from soprano.calculate.nmr import NMRCalculator
from soprano.calculate.nmr.nmr import NMRData2D, NMRPlot2D, PlotSettings
from soprano.properties.nmr import MSIsotropy
from soprano.nmr.extract import nmr_extract_multi
from soprano.scripts.cli_utils import PLOT_OPTIONS, add_options, print_results, viewimages, reload_as_molecular_crystal
from soprano.selection import AtomSelection
import ase.io as _ase_io

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
    weight_by,
    rss_cutoff,
    rss_expand_j,
    scale_markers,
    max_marker_size,
    marker_color,
    marker_linewidth,
    show_marker_legend,
    show_diagonal,
    show_grid,
    show_connectors,
    show_ticklabels,
    show_heatmap,
    heatmap_levels,
    xbroadening,
    ybroadening,
    grid_max,
    colormap,
    show_contour,
    contour_levels,
    intensity_range,
    contour_range,
    heatmap_range,
    contour_color,
    contour_linewidth,
    plot_filename,
    plot_shielding,  ## force-plot the shielding even if references are given
    export_files,
    export_format,
    x_larmor_freq_mhz,
    y_larmor_freq_mhz,
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
        sortby=None,
        sort_order="ascending",
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

    # For 2D plots with reduce=True, pass the raw (unmerged) unit-cell atoms
    # to NMRData2D and let it handle reduction internally.  NMRData2D stores
    # the pre-reduction atoms as atoms_full so RSS expansion can find all Z
    # copies of each site.  When reduce=False, images[0] already contains all
    # atoms so we can pass it directly.
    atoms_for_2d = atoms
    if reduce:
        try:
            _raw = _ase_io.read(files[0])
            atoms_for_2d = reload_as_molecular_crystal(_raw)
        except Exception as e:
            logger.warning(
                f"Could not reload raw atoms for 2D: {e}. "
                "Using pre-reduced atoms; RSS expansion may be incomplete for Z > 1."
            )

    if plot_type == "2D":
        if not y_element:
            y_element = x_element

        shift = not plot_shielding if plot_shielding is not None else references != {}

        # Create NMRData2D instance
        nmr_data = NMRData2D(
            atoms=atoms_for_2d,
            xelement=x_element,
            yelement=y_element,
            rcut=rcut,
            references=references,
            gradients=gradients,
            isotopes=isotopes,
            is_shift=shift,
            include_quadrupolar=False,
            yaxis_order=yaxis_order,
            correlation_strength_metric=weight_by,
            rss_cutoff=rss_cutoff,
            rss_expand_j=rss_expand_j,
            reduce=reduce,
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
            heatmap_levels=heatmap_levels,
            show_contour=show_contour,
            colormap=colormap,
            marker_color=marker_color,
            show_legend=show_marker_legend,
            contour_levels=contour_levels,
            intensity_range=intensity_range,
            contour_range=contour_range,
            heatmap_range=heatmap_range,
            contour_color=contour_color,
            contour_linewidth=contour_linewidth,
            x_broadening=xbroadening,
            y_broadening=ybroadening,
            grid_max=grid_max,
            scale_markers=scale_markers,
        )

        # Export contour data if requested
        if export_files:
            _EXT_TO_FMT = {
                '.spe': 'simpson',
                '.sim': 'simpson',
                '.npz': 'npz',
                '.csv': 'csv',
                '.json': 'json',
            }
            for export_path in export_files:
                fmt = export_format or _EXT_TO_FMT.get(Path(export_path).suffix.lower(), 'simpson')
                logger.info(f"Exporting contour data to '{export_path}' (format={fmt}).")
                nmr_data.export_contour_data(
                    path=export_path,
                    fmt=fmt,
                    x_broadening=xbroadening,
                    y_broadening=ybroadening,
                    grid_max=grid_max,
                    x_larmor_freq_mhz=x_larmor_freq_mhz,
                    y_larmor_freq_mhz=y_larmor_freq_mhz,
                )

        # Create NMRPlot2D instance
        nmr_plot = NMRPlot2D(
            nmr_data=nmr_data,
            plot_settings=plot_settings,
        )

        # Generate the plot
        result = nmr_plot.plot()
        # Matplotlib returns (fig, ax); Plotly returns a single Figure object
        if isinstance(result, tuple):
            fig, ax = result
        else:
            fig = result
        # if the user doesn't give an output file name, show the plot using the default matplotlib backend
        if not plot_filename:
            plt.show()
        return 0
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
            if x_element not in references:
                logger.error(
                    f"No reference found for element '{x_element}'. "
                    "Provide one with --references or use --shielding to plot shielding instead."
                )
                return 1
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
        return 0
    else:
        logger.error("Invalid plot type. Aborting.")
        return 1

    return 0
