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

Roadmap / open items:
- [ ] Add selectable matplotlib stylesheets.
- [x] Dipolar / J-coupling scaling of markers (implemented via --weight-by).
      Caveat: verify the correlation-strength assumptions when combined with the
      ``reduce`` or functional-group averaging options (multiplicity weighting).
- [ ] 1D plots -- basic simulation of a 1D NMR spectrum.
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
from click.core import ParameterSource

from soprano.calculate.nmr import NMRCalculator
from soprano.calculate.nmr.export import ExportConfig, guess_format_from_path
from soprano.calculate.nmr.nmr import NMRData2D, NMRPlot2D, PlotSettings
from soprano.nmr.extract import nmr_extract_multi
from soprano.properties.nmr import MSIsotropy
from soprano.scripts.cli_utils import PLOT_OPTIONS, add_options, print_results, viewimages
from soprano.selection import AtomSelection

# logging
logging.captureWarnings(True)
logger = logging.getLogger("cli")
click_log.basic_config(logger)


def _given_on_cli(ctx, name):
    return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE


def _check_plotnmr_options(
    ctx,
    export_files,
    export_format,
    b0_field_tesla,
    spectrometer_freq_mhz,
    x_larmor_freq_mhz,
    y_larmor_freq_mhz,
    weight_by,
    rss_cutoff,
    rss_expand_j,
):
    """Raise a UsageError for invalid option combinations, and warn about ignored options."""
    if b0_field_tesla is not None and spectrometer_freq_mhz is not None:
        raise click.UsageError(
            "--b0-field-tesla and --spectrometer-freq are mutually exclusive. Give only one."
        )

    has_field = any(
        freq is not None
        for freq in (b0_field_tesla, spectrometer_freq_mhz, x_larmor_freq_mhz)
    )
    for path in export_files:
        try:
            fmt = (export_format or guess_format_from_path(path)).lower()
        except ValueError as err:
            raise click.UsageError(str(err)) from None
        if fmt in ("json", "ssnake", "bruker") and not has_field:
            raise click.UsageError(
                f"Export format '{fmt}' for '{path}' needs a Larmor frequency for the ppm axes. "
                "Give --b0-field-tesla, --spectrometer-freq or --x-larmor-freq."
            )

    if not export_files:
        for opt, name in (
            ("--export-format", "export_format"),
            ("--x-larmor-freq", "x_larmor_freq_mhz"),
            ("--y-larmor-freq", "y_larmor_freq_mhz"),
        ):
            if _given_on_cli(ctx, name):
                logger.warning(f"{opt} has no effect without --export-file.")

    if weight_by != "dipolar_rss":
        for opt, name in (("--rss-cutoff", "rss_cutoff"), ("--rss-expand-j", "rss_expand_j")):
            if _given_on_cli(ctx, name):
                logger.warning(f"{opt} only applies with --weight-by dipolar_rss. Ignoring it.")


def _plot_2d(
    atoms,
    x_element,
    y_element,
    shift,
    rcut,
    references,
    gradients,
    isotopes,
    yaxis_order,
    weight_by,
    rss_cutoff,
    rss_expand_j,
    reduce,
    average_group,
    xlim,
    ylim,
    show_markers,
    marker_symbol,
    max_marker_size,
    marker_linewidth,
    plot_filename,
    show_grid,
    show_diagonal,
    show_connectors,
    show_ticklabels,
    show_heatmap,
    heatmap_levels,
    heatmap_grid_size,
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
    marker_color,
    show_marker_legend,
    scale_markers,
    export_files,
    export_format,
    overwrite,
    b0_field_tesla,
    spectrometer_freq_mhz,
    x_larmor_freq_mhz,
    y_larmor_freq_mhz,
):
    """Build, export and plot a 2D correlation spectrum."""
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
        correlation_strength_metric=weight_by,
        rss_cutoff=rss_cutoff,
        rss_expand_j=rss_expand_j,
        reduce=reduce,
        average_group=average_group or "",
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
        heatmap_grid_size=heatmap_grid_size,
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
    for export_path in export_files:
        fmt = export_format or guess_format_from_path(export_path)
        logger.info(f"Exporting contour data to '{export_path}' (format={fmt}).")
        config = ExportConfig(
            x_broadening=xbroadening,
            y_broadening=ybroadening,
            grid_max=grid_max,
            x_larmor_freq_mhz=x_larmor_freq_mhz,
            y_larmor_freq_mhz=y_larmor_freq_mhz,
            b0_field_tesla=b0_field_tesla,
            spectrometer_freq_mhz=spectrometer_freq_mhz,
            overwrite=overwrite,
        )
        try:
            nmr_data.export_contour_data(
                path=export_path,
                fmt=fmt,
                config=config,
            )
        except FileExistsError as err:
            raise click.UsageError(str(err)) from None

    # Create NMRPlot2D instance
    nmr_plot = NMRPlot2D(
        nmr_data=nmr_data,
        plot_settings=plot_settings,
    )

    # Generate the plot
    nmr_plot.plot()
    # if the user doesn't give an output file name, show the plot using the default matplotlib backend
    if not plot_filename:
        plt.show()


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
    mean_merge,
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
    heatmap_grid_size,
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
    overwrite,
    b0_field_tesla,
    spectrometer_freq_mhz,
    x_larmor_freq_mhz,
    y_larmor_freq_mhz,
    verbosity,
    symprec,
    view,
):
    """
    Plot the NMR spectrum from a .magres file.
    """
    _check_plotnmr_options(
        click.get_current_context(),
        export_files=export_files,
        export_format=export_format,
        b0_field_tesla=b0_field_tesla,
        spectrometer_freq_mhz=spectrometer_freq_mhz,
        x_larmor_freq_mhz=x_larmor_freq_mhz,
        y_larmor_freq_mhz=y_larmor_freq_mhz,
        weight_by=weight_by,
        rss_cutoff=rss_cutoff,
        rss_expand_j=rss_expand_j,
    )

    shift = not plot_shielding if plot_shielding is not None else bool(references)

    # Only extract data for the element(s) being plotted, so that partial
    # reference dictionaries (e.g. C and H only) don't fail on other elements
    # present in the structure (e.g. O).
    plot_elements = [x_element]
    if plot_type == "2D" and y_element and y_element != x_element:
        plot_elements.append(y_element)
    if shift:
        missing = [el for el in plot_elements if el not in (references or {})]
        if missing:
            raise click.UsageError(
                f"No reference found for element(s) {', '.join(missing)}. "
                "Provide them with --references (e.g. --references "
                f"{x_element}:170) or use --shielding to plot shielding instead."
            )

    # Hard-code a few options for nmr_extract since the user doesn't need to specify them
    properties = ["ms"]  # we at least need to extract the MS
    include = ["MS_shielding"]  # we only need the MS columns for plotting
    if shift:
        include.append("MS_shift")
    exclude = None
    merge = True  # if multiple files are given, we have to merge them
    subset = ",".join(plot_elements)

    # set verbosity
    if verbosity == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    dfs, images, sources = nmr_extract_multi(
        files,
        return_sources=True,
        subset=subset,
        merge=merge,
        isotopes=isotopes,
        references=references if shift else None,
        gradients=gradients if shift else None,
        reduce=reduce,
        average_group=average_group,
        mean_merge=mean_merge,
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
        raise click.ClickException("No NMR data extracted. Nothing to plot.")
    atoms = images[0]

    # 2D plots take the structure as it was read and reduce it internally, so
    # that RSS expansion can still see all Z copies of each site.  images[0]
    # has already been reduced, hence sources[0] rather than atoms here.
    atoms_for_2d = sources[0]

    if plot_type == "2D":
        if not y_element:
            y_element = x_element

        try:
            _plot_2d(
                atoms_for_2d,
                x_element=x_element,
                y_element=y_element,
                shift=shift,
                rcut=rcut,
                references=references,
                gradients=gradients,
                isotopes=isotopes,
                yaxis_order=yaxis_order,
                weight_by=weight_by,
                rss_cutoff=rss_cutoff,
                rss_expand_j=rss_expand_j,
                reduce=reduce,
                average_group=average_group,
                xlim=xlim,
                ylim=ylim,
                show_markers=show_markers,
                marker_symbol=marker_symbol,
                max_marker_size=max_marker_size,
                marker_linewidth=marker_linewidth,
                plot_filename=plot_filename,
                show_grid=show_grid,
                show_diagonal=show_diagonal,
                show_connectors=show_connectors,
                show_ticklabels=show_ticklabels,
                show_heatmap=show_heatmap,
                heatmap_levels=heatmap_levels,
                heatmap_grid_size=heatmap_grid_size,
                xbroadening=xbroadening,
                ybroadening=ybroadening,
                grid_max=grid_max,
                colormap=colormap,
                show_contour=show_contour,
                contour_levels=contour_levels,
                intensity_range=intensity_range,
                contour_range=contour_range,
                heatmap_range=heatmap_range,
                contour_color=contour_color,
                contour_linewidth=contour_linewidth,
                marker_color=marker_color,
                show_marker_legend=show_marker_legend,
                scale_markers=scale_markers,
                export_files=export_files,
                export_format=export_format,
                overwrite=overwrite,
                b0_field_tesla=b0_field_tesla,
                spectrometer_freq_mhz=spectrometer_freq_mhz,
                x_larmor_freq_mhz=x_larmor_freq_mhz,
                y_larmor_freq_mhz=y_larmor_freq_mhz,
            )
        except ValueError as err:
            if weight_by == "dipolar_rss" and rss_expand_j == "cif_labels":
                raise click.UsageError(
                    f"--rss-expand-j cif_labels needs a .magres file with CIF labels. {err}"
                ) from None
            raise
    elif plot_type == "1D":
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
        raise click.UsageError(f"Invalid plot type '{plot_type}'.")
