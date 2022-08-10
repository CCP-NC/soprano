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
TODO: 1D plots -- basic simulation of 1D NMR?
'''

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "July 12, 2022"


import click
import numpy as np
import re
import os
import sys
import re
from ase import io
from ase.visualize import view as aseview
from ase.units import Ha, Bohr
from soprano.properties.labeling import UniqueSites, MagresViewLabels
from soprano.properties.nmr import *
from soprano.data.nmr import _el_iso, _get_isotope_list
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels
import pandas as pd
import warnings
from collections import OrderedDict


@click.command()

# one of more files
@click.argument('csv_file',
                nargs=1,
                type=click.Path(exists=True),
                required=True)
# plot type argument
@click.option('--plot_type',
                '-p',
                type=click.Choice(['2D', '1D']),
                default='2D',
                help='Plot type')
# x-element
@click.option('--xelement',
                '-x',
                'x_element',
                type=str,
                required = True,
                )
# y element
@click.option('--yelement',
                '-y',
                'y_element',
                type=str,
                required = False,
                help = 'Element to plot on the y-axis. '
                'If not specified, but a 2D plot is requested, the x-element is used.'
                )
# flip x and y
@click.option('--flipx',
                '-fx',
                'flip_x',
                is_flag=True,
                default=False,
                help='Flip x axis')
@click.option('--flipy',
                '-fy',
                'flip_y',
                is_flag=True,
                default=False,
                help='Flip y axis')
# x-axis label
@click.option('--xlabel',
                type=str,
                default='',
                help='Custom X-axis label')
# y-axis label
@click.option('--ylabel',
                type=str,
                default='',
                help='Custom Y-axis label.')
# x-axis range
@click.option('--xlim',
                nargs=2,
                type=float,
                default=None,
                help='X-axis range. For example ``--xlim 20 100``')
# y-axis range
@click.option('--ylim',
                nargs=2,
                type=float,
                default=None,
                help='Y-axis range. For example ``--ylim 20 100``')
# marker
@click.option('--marker',
                type=str,
                default='+',
                help='Marker type. '
                'For example ``--marker o``. '
                'Accepts any matplotlib marker type.')
# marker size
@click.option('--markersize',
               '-ms',
               type=float,
               default=200,
               help='Marker size. Default is 200.')
# plot filename
@click.option('--output',
                '-o',
                'plot_file',
                type=click.Path(exists=False),
                required = False,
                help = "Name of the plot file. "
                "If not specified, the plot will be displayed in a window. "
                "The file extension determines the plot type (svg or pdf recommended)."
                )
def plotnmr(csv_file,
            x_element,
            y_element,
            plot_type,
            plot_file,
            flip_x,
            flip_y, 
            xlabel,
            ylabel,
            xlim,
            ylim,
            marker,
            markersize,
            ):
    '''
    Plot the NMR spectrum from a csv file.
    '''
    df = pd.read_csv(csv_file)
    
    if plot_type == '2D':
        if not y_element:
            y_element = x_element
        plot_2D_nmr(df,
                    x_element,
                    y_element,
                    plot_filename=plot_file,
                    flip_x = flip_x,
                    flip_y = flip_y,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    xlim   = xlim,
                    ylim   = ylim,
                    marker=marker,
                    markersize=markersize,
                    )
    return 0




def plot_2D_nmr(df, 
                xelement,
                yelement,
                plot_filename = None,
                include_quadrupolar=False,
                yaxis_order='1Q', 
                plot_shifts=False, 
                marker='+', 
                markersize = 200,
                flip_x=False,
                flip_y=False, 
                xlabel='',
                ylabel='',
                xlim=None,
                ylim=None,
                show_labels=True, 
                show_lines=True ):
    '''
    Plot a 2D NMR spectrum from a dataframe with columns 'frequency', 'intensity'
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    # some syle
    linealpha = 1.0
    linecolor = '0.75'
    linewidth = 0.5
    linestyle = '-'
    matplotlib.style.use('seaborn-paper')


    # split the species into isotope, element
    iso_el_df = df['species'].str.split('(\d+)([A-Za-z]+)', expand=True)
    iso_el_df = iso_el_df.loc[:,[1,2]]
    iso_el_df.rename(columns={1:'isotope', 2:'element'}, inplace=True)
    isotopes = iso_el_df['isotope']
    elements = iso_el_df['element']
    idx_x = elements == xelement
    idx_y = elements == yelement
    if plot_shifts:
        axis_label = f'$\delta$ ({yaxis_order}, ppm)'
        col = "MS_shift/ppm"
    else:
        axis_label = f'$\sigma$ ({yaxis_order}, ppm)'
        col = "MS_shielding/ppm"

    x = df.loc[idx_x][col]
    y = df.loc[idx_y][col]


    # make the plot!
    fig, ax = plt.subplots()
    if show_lines:
        for i, xval in enumerate(x):
            ax.axvline(xval, ls=linestyle, alpha= linealpha, lw=linewidth, c=linecolor)
        for i, yval in enumerate(y):
            ax.axhline(yval, ls=linestyle, alpha= linealpha, lw=linewidth, c=linecolor)
    
    # Mark intersections with points
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):
            ax.scatter(xval, yval, s=markersize, marker=marker, c='C1', zorder=10)
    
    
    
    ax.set_xlabel(f'{xelement} {axis_label}')
    ax.set_ylabel(f'{yelement} {axis_label}')

    #TODO  set these with x and y ticks ont he top/right
    if show_labels:
        if 'MagresView_labels' in df.columns:
            col = 'MagresView_labels'
        else:
            col = 'labels'
        ax_r = ax.secondary_yaxis('right')
        ax_t = ax.secondary_xaxis('top')
        ax_r.tick_params(axis='y', direction='out', labelrotation=0)
        ax_t.tick_params(axis='x', direction='out', labelrotation=90)
        # custom ticks

        ax_r.set_yticks(y)
        ax_r.set_yticklabels(df.loc[idx_y][col])

        ax_t.set_xticks(x)
        ax_t.set_xticklabels(df.loc[idx_x][col])
        # for i, txt in enumerate(df[col]):
        #     ax.annotate(txt, (x.iloc[i], y.iloc[i]))
    # if shfits are plotted, invert the axes
    if plot_shifts:
        ax.invert_xaxis()
        ax.invert_yaxis()

    # other plot options
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if xelement == yelement:
        # might be nice to have a diagonal line
        ax.plot([0, 1], [0, 1],
                transform=ax.transAxes,
                c='k',
                ls='-',
                alpha=0.2,
                )
    
    if plot_filename:
        fig.savefig(plot_filename)
    else:
        plt.show()

    return fig, ax