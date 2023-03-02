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

'''CLI to plot the extracted NMR results. Note, please run the nmr command first
with the flag -o seedname.csv to generate the csv file needed for this command.

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
__date__ = "July 12, 2022"


import click
import click_log

import numpy as np
import re
import os
import sys
import re
from ase import Atoms
from ase.visualize import view as aseview
from ase.units import Ha, Bohr
from soprano.properties.labeling import UniqueSites, MagresViewLabels
from soprano.properties.nmr import *
from soprano.data.nmr import _el_iso, _get_isotope_list
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels
import itertools
import pandas as pd
from collections import OrderedDict
from soprano.scripts.nmr import nmr_extract, print_results
from soprano.scripts.cli_utils import PLOT_OPTIONS, add_options, DEFAULT_MARKER_SIZE
from soprano.scripts.dipolar import extract_dipolar_couplings
from soprano.calculate.nmr import NMRCalculator, NMRFlags
from soprano.properties.nmr import MSIsotropy
import logging

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

MARKER_INFO = {
    'distance': {
        'label': 'Distance',
        'unit': 'Å',
        'fmt': '{x:.1f}'
    },
    'inversedistance': {
        'label': '1/Distance',
        'unit': r'Å$^{{-1}}$',
        'fmt': '{x:.3f}'
    },
    'dipolar': {
        'label': 'Dipolar Coupling',
        'unit': 'kHz',
        'fmt': '{x:.1f}'
    },
    'jcoupling': {
        'label': 'J Coupling',
        'unit': 'Hz',
        'fmt': '{x:.1f}'
    },
    'fixed': {
        'label': 'Fixed',
        'unit': '',
        'fmt': '{x:.1f}'
    }


    }


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
    selection,
    reduce,
    combine_rule,
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
    plot_filename,
    shift,
    quiet,
    symprec,
    precision,
    view
        ):
    '''
    Plot the NMR spectrum from a csv file.
    '''
    # Hard-code a few options for nmr_extract since the user doesn't need to specify them
    properties = ['ms'] # we at least need to extract the MS
    include = ['MS_shielding'] # we only need the MS columns for plotting
    if references:
        include.append('MS_shift')
    exclude = None
    merge = True # if multiple files are given, we have to merge them
    sortby = None
    sort_order = None


    # if quiet:
    #     logging.basicConfig(level=logging.WARNING)
    # else:
    logger.setLevel(logging.INFO)
    dfs, images = nmr_extract(files, selection, merge, isotopes, references, gradients, reduce, average_group, combine_rule, symprec, properties, precision, euler_convention, sortby, sort_order, include, exclude, query, view)
    
    # write to file(s)
    print_results(dfs)
    if len(dfs) > 1:
        logger.warning("More than one dataframe extracted. Only plotting the first one.")
    if len(dfs) == 0:
        logger.error("No dataframes extracted. Aborting.")
        return 1
    df = dfs[0]
    atoms = images[0]
    


    # if the user hasn't specified plot_shifts, then we 
    logger.info(f'plot_shielding_shift:  {shift}')
    if references:
        if shift is None:
            shift = True
            logger.info("Plotting chemical shifts since references are given. "
                        "To instead plot shielding, use the --shielding flag or do not give shift references.")
        elif not shift:
            logger.warning("--shielding flag is set, but references are given. "
                           "Chemical shielding will be plotted, not shifts.")

    if plot_type == '2D':
        if not y_element:
            y_element = x_element
        
        plot = Plot2D(
                df,
                atoms,
                x_element,
                y_element,
                rcut = rcut,
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
                marker_color = 'C1',
                show_marker_legend=show_marker_legend)
        
        plot.plot()
    elif plot_type == '1D':
        sel = AtomSelection.all(atoms)
        element_sel = AtomSelection.from_element(atoms, x_element)
        sel = sel * element_sel
        atoms = sel.subset(atoms)
        # get the NMR calculator
        calc = NMRCalculator(atoms)
        use_reference = False
        if shift:
            logger.info(f"Setting references: {references}")
            calc.set_reference(ref = references[x_element], element=x_element)
            use_reference = True
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
        ax.set_xlabel('Frequency (ppm)')
        ax.set_ylabel('Intensity')
        if use_reference:
            ax.invert_xaxis()
        if plot_filename:
            fig.savefig(plot_filename)
        else:
            plt.show()
    else:
        logger.error("Invalid plot type. Aborting.")

    return 0

class Plot2D:
    '''
    Class to handle the 2D plotting of NMR data.
    '''
    def __init__(self, 
                df, 
                atoms:Atoms,
                xelement,
                yelement,
                rcut = None,
                isotopes=None,
                plot_shifts=False,
                include_quadrupolar=False,
                yaxis_order='1Q',
                xlim=None,
                ylim=None,
                marker='x',
                scale_marker_by = 'fixed',
                max_marker_size=DEFAULT_MARKER_SIZE,
                show_ticks=True,
                show_lines=True,
                plot_filename=None,
                marker_color = 'C1',
                show_marker_legend=False
                ):
        self.df = df
        self.atoms = atoms
        self.xelement = xelement
        self.yelement = yelement
        self.rcut = rcut
        self.isotopes = isotopes
        self.plot_shifts = plot_shifts
        self.include_quadrupolar = include_quadrupolar
        self.yaxis_order = yaxis_order
        self.xaxis_order = '1Q'
        self.xlim = xlim
        self.ylim = ylim
        self.marker = marker
        self.scale_marker_by = scale_marker_by
        self.max_marker_size = max_marker_size
        self.show_ticks = show_ticks
        self.show_lines = show_lines
        self.plot_filename = plot_filename
        
        self.marker_unit = MARKER_INFO[self.scale_marker_by]['unit']
        self.marker_label = MARKER_INFO[self.scale_marker_by]['label']
        self.marker_fmt = MARKER_INFO[self.scale_marker_by]['fmt']
        self.marker_color = marker_color
        self.show_marker_legend = show_marker_legend


    def get_2D_plot_data(self):
        '''
        Get the data for a 2D NMR plot from 
        a dataframe with columns:
        'MS_shift/ppm' or 'MS_shielding/ppm'
        
        If include_quadrupolar is True, then the quadrupolar
        couplings should also be included in the df.
        

        '''
        # split the species into isotope, element
        iso_el_df = self.df['species'].str.split('(\d+)([A-Za-z]+)', expand=True)
        iso_el_df = iso_el_df.loc[:,[1,2]]
        iso_el_df.rename(columns={1:'isotope', 2:'element'}, inplace=True)
        isotopes = iso_el_df['isotope']
        elements = iso_el_df['element']
        if self.xelement not in elements.values:
            raise ValueError(f'{self.xelement} not found in the file after the user-specified filters have been applied.')
        if self.yelement not in elements.values:
            raise ValueError(f'{self.yelement} not found in the file after the user-specified filters have been applied.')
        self.idx_x = np.where(elements == self.xelement)[0]
        self.idx_y = np.where(elements == self.yelement)[0]
        # idx_x is the row index of the xelement in the dataframe
        # now we want the actual index of the dataframe
        self.idx_x = self.df.iloc[self.idx_x].index.values
        self.idx_y = self.df.iloc[self.idx_y].index.values
        logger.debug(f'Indices of {self.xelement} in the dataframe: {self.idx_x}')
        logger.debug(f'Indices of {self.yelement} in the dataframe: {self.idx_y}')
        species_template =  r'$\mathrm{^{%s}{%s}}$'
        self.xspecies = species_template % (isotopes[self.idx_x[0]], elements[self.idx_x[0]])
        self.yspecies = species_template % (isotopes[self.idx_y[0]], elements[self.idx_y[0]])
        # log species
        logger.debug(f'X species: {self.xspecies}')
        logger.debug(f'Y species: {self.yspecies}')
        self.get_axis_labels()
        col = "MS_shielding/ppm"
        if self.plot_shifts:
            col = "MS_shift/ppm"

        # get pairs
        self.get_plot_pairs()
        # marker sizes
        self.get_marker_sizes()

        self.x = self.df.loc[self.idx_x][col]
        self.y = self.df.loc[self.idx_y][col]

        # log the x and y values
        logger.debug(f'X values: {self.x}')
        logger.debug(f'Y values: {self.y}')

        # tick labels and ticks
        self.get_ticks()

    def get_axis_labels(self):
        if self.plot_shifts:
            
            axis_label = r"$\delta_{\mathrm{%s}}$ / ppm"
        else:
            axis_label = r"$\sigma_{\mathrm{%s}}$ / ppm"

        self.x_axis_label = f'{self.xspecies} ' + axis_label % self.xaxis_order
        self.y_axis_label = f'{self.yspecies} ' + axis_label % self.yaxis_order



    def get_ticks(self):
        '''
        Get the tick labels and ticks for the plot
        '''

        if not self.show_ticks:
            self.xticks = []
            self.yticks = []
            self.xticks_labels = []
            self.yticks_labels = []
            return
        
        # get the x and y tick labels
        if 'MagresView_labels' in self.df.columns:
            col = 'MagresView_labels'
        else:
            col = 'labels'
        
        # custom ticks
        self.xticks = self.x
        self.xticks_labels = self.df.loc[self.idx_x][col]

        if self.yaxis_order == '1Q':
            self.yticks = self.y
            self.yticks_labels = self.df.loc[self.idx_y][col]

        elif self.yaxis_order == '2Q':
            # loop over pairs
            self.yticks = []
            self.yticks_labels = []
            for pair in self.pairs:
                ticky = self.x[pair[0]] + self.y[pair[1]]
                self.yticks.append(ticky)
                ticky_label = f'{self.xticks_labels[pair[0]]} + {self.df.loc[pair[1]][col]}'
                self.yticks_labels.append(ticky_label)


    def get_plot_pairs(self):
        # marker sizes is based on chosen property: fixed, dipolar, J-coupling, distance etc.
        self.pairs = list(itertools.product(self.idx_x, self.idx_y))
        
        # remove any pairs where the x and y indices are the same
        # this should only be the case if xelement == yelement
        # do we want to allow this if yaxis_order == '1Q'?
        # self.pairs = [pair for pair in self.pairs if pair[0] != pair[1]]
        
        # check if any two indices in a pair are the same if marker size is not fixed
        if self.scale_marker_by != 'fixed':
            for pair in self.pairs:
                if len(set(pair)) != 2:
                    raise ValueError("""
                    Two indices in a pair are the same but
                    the marker size is based on distance between sites.
                    It's unclear """)
        
        
        if len(self.pairs) == 0:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        self.pair_distances = np.zeros(len(self.pairs))
        for i, pair in enumerate(self.pairs):
            if pair[0] == pair[1]:
                # (rather than looking for periodic images of the same atom)
                self.pair_distances[i] = 0.0
            else:
                self.pair_distances[i] = self.atoms.get_distance(*pair, mic=True)
        
        if self.rcut:
            # now filter out those pairs that are too far away
            logger.info(f"Filtering out pairs that are further than {self.rcut} Å apart.")
            logger.info(f"Number of pairs before filtering: {len(self.pairs)}")
            
            dist_mask = np.where(self.pair_distances <= self.rcut)[0]
            self.pairs = [self.pairs[i] for i in dist_mask]
            self.pair_distances = self.pair_distances[dist_mask]
            # update the idx_x and idx_y
            self.idx_x = np.unique([pair[0] for pair in self.pairs])
            self.idx_y = np.unique([pair[1] for pair in self.pairs])
            if len(self.idx_x) == 0 or len(self.idx_y) == 0:
                raise ValueError(f'No pairs found after filtering by distance. Try increasing the cutoff distance (rcut).')
            logger.info(f"Number of pairs remaining: {len(self.pairs)}")

    def get_marker_sizes(self):
        
        if self.scale_marker_by == 'fixed':
            logger.info("Using fixed marker size.")
            # get all unique pairs of x and y indices
            # set the marker size to be the same for all pairs
            self.markersizes = np.ones(len(self.pairs))
            
        elif self.scale_marker_by == 'dipolar':
            logger.info("Using dipolar coupling as marker size.")
            logger.debug(f"Using custom isotopes: {self.isotopes}")
            # DipolarCoupling.get returns a dictionary with the dipolar coupling but we oly have one element in the dictionary.
            # We need the first element of the value of this item.
            dip = []
            for i, j in self.pairs:
                if i == j:
                    # set the dipolar coupling to zero for pairs where i == j
                    dip.append(0)
                else:
                    dip.append(list(DipolarCoupling.get(self.atoms, 
                                                        sel_i=[i],
                                                        sel_j=[j],
                                                        isotopes=self.isotopes).values())[0][0])
            # convert to kHz
            dip = np.array(dip) * 1e-3
            self.markersizes = np.array(dip)
        elif self.scale_marker_by == 'distance' or self.scale_marker_by == 'inversedistance':
            log_message = "Using minimum image convention {isinverse}distance as marker size."
            isinverse = ''

            # now we can use ASE get_distance to get the distances for each pair
            self.markersizes[i] = self.pair_distances
            if self.scale_marker_by == 'inversedistance':
                self.markersizes = 1 / self.markersizes
                isinverse = 'inverse '
            logger.info(log_message.format(isinverse=isinverse))
            

        elif self.scale_marker_by == 'J':
            logger.info("Using J-coupling as marker size.")
            raise NotImplementedError("J-coupling scaling not implemented yet.")

        else:
            raise ValueError(f"Unknown scale_marker_by option: {self.scale_marker_by}")
        
        logger.debug(f"markersizes: {self.markersizes}")

        #
        # log pair with smallest and largest marker size
        smallest_pair = self.pairs[np.argmin(np.abs(self.markersizes))]
        largest_pair = self.pairs[np.argmax(np.abs(self.markersizes))]
        # labels for the smallest and largest pairs
        smallest_pair_labels = [self.df.loc[smallest_pair[0]]['labels'], self.df.loc[smallest_pair[1]]['labels']]
        largest_pair_labels = [self.df.loc[largest_pair[0]]['labels'], self.df.loc[largest_pair[1]]['labels']]
        logger.info(f"Pair with smallest (abs) {self.marker_label}: {smallest_pair_labels}")
        logger.info(f"Pair with largest (abs) {self.marker_label}: {largest_pair_labels}")


    def plot(self):
        '''
        Plot a 2D NMR spectrum from a dataframe with columns 'MS_shift/ppm' or 'MS_shielding/ppm'

        '''
        logger.info("Plotting 2D NMR spectrum...")
        logger.info(f"Plotting {self.xelement} vs {self.yelement}.")

        # get the data
        self.get_2D_plot_data()

        # some syle
        linealpha = 0.5
        linecolor = '0.75'
        linewidth = 0.5
        linestyle = '-'
        matplotlib.style.use('seaborn-paper')

        # make the plot!
        fig, ax = plt.subplots()

        ax_r = ax.secondary_yaxis('right')
        ax_t = ax.secondary_xaxis('top')
        ax_r.tick_params(axis='y', direction='out', labelrotation=0)
        ax_t.tick_params(axis='x', direction='out', labelrotation=90)

        # set the ticks
        ax_t.set_xticks(self.xticks)
        ax_r.set_yticks(self.yticks)
        # and tick labels
        ax_r.set_yticklabels(self.yticks_labels)
        ax_t.set_xticklabels(self.xticks_labels)


        if self.show_lines:
            for i, xval in enumerate(self.xticks):
                ax.axvline(xval, ls=linestyle, alpha= linealpha, lw=linewidth, c=linecolor)
            for i, yval in enumerate(self.yticks):
                ax.axhline(yval, ls=linestyle, alpha= linealpha, lw=linewidth, c=linecolor)
        
        # --- plot the markers ---
        xvals = [self.x[pair[0]] for pair in self.pairs]
        yvals = [self.y[pair[1]] for pair in self.pairs]
        if self.yaxis_order == '2Q':
            yvals = [x+y for x, y in zip(xvals, yvals)]
        # make sure the marker sizes are all positive
        markersizes = np.abs(self.markersizes)
        marker_size_range = np.max(markersizes) - np.min(markersizes)
        if self.scale_marker_by != 'fixed':
            logger.info(f"Marker size range: {marker_size_range} {self.marker_unit}")
        max_abs_marker = np.max(markersizes)
        # normalise the marker sizes such that the maximum marker size is self.max_marker_size
        markersizes = markersizes / max_abs_marker * self.max_marker_size
        # plot the markers
        scatter = ax.scatter(
            xvals,
            yvals,
            s=markersizes,
            marker=self.marker,
            c=self.marker_color,
            zorder=10)
        

        # --- plot the axis labels ---
        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)

        # if shifts are plotted, invert the axes
        if self.plot_shifts:
            ax.invert_xaxis()
            ax.invert_yaxis()

        # other plot options
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)

        if self.xelement == self.yelement:
            # use self.xlim and self.ylim to draw a diagonal line
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
            ax.plot(xlims, ylims, ls='--', c='k', lw=1, alpha=0.2)
        
        # add marker size legend
        if self.scale_marker_by != 'fixed' and self.show_marker_legend:
            # produce a legend with a cross-section of sizes from the scatter
            kw = dict(prop="sizes", num=5, color=self.marker_color, 
                      fmt=self.marker_fmt + f" {self.marker_unit}",
                      func=lambda s: s*max_abs_marker / self.max_marker_size)
            handles, labels = scatter.legend_elements(**kw)
            ax.legend(handles, labels,
                      loc="upper left",
                      title=self.marker_label,
                      fancybox=True,
                      framealpha=0.8).set_zorder(11)


        if self.plot_filename:
            logger.debug(f"Saving to {self.plot_filename}")
            fig.savefig(self.plot_filename)
        else:
            plt.show()

        return fig, ax