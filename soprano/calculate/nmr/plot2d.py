"""2D NMR plotting orchestrator."""

import logging
from typing import Optional

import numpy as np
from matplotlib.axes import Axes

from soprano.calculate.nmr.backends import MatplotlibBackend, PlotlyBackend
from soprano.calculate.nmr.config import DEFAULT_MAX_NUM_LEGEND_ELEMENTS, PlotSettings
from soprano.calculate.nmr.data2d import NMRData2D
from soprano.calculate.nmr.utils import nmr_2D_style, nmr_base_style, styled_plot

class NMRPlot2D:
    '''
    Class to plot 2D NMR data with pluggable backends.
    
    Parameters
    ----------
    nmr_data : NMRData2D
        The NMR data to plot
    plot_settings : Optional[PlotSettings]
        Plot settings to use. If None, defaults are used.
    backend : str
        Backend to use for plotting. Options: 'matplotlib' (default), 'plotly'
    ax : Optional[Axes]
        For matplotlib backend: existing axis to plot on. If None, creates new figure.
    '''
    def __init__(self,
                nmr_data: NMRData2D,
                plot_settings: Optional[PlotSettings] = None,
                backend: str = 'matplotlib',
                ax: Optional[Axes] = None):

        self.nmr_data = nmr_data
        self.backend_name = backend
        
        # store the data as numpy arrays for plotting
        npeaks = len(self.nmr_data.peaks)
        self.x = np.zeros(npeaks)
        self.y = np.zeros(npeaks)
        self.sizes = np.zeros(npeaks)

        for i, peak in enumerate(self.nmr_data.peaks):
            self.x[i] = peak.x
            self.y[i] = peak.y
            self.sizes[i] = peak.correlation_strength * peak.multiplicity


        # Use default plot settings if none are provided
        if plot_settings is None:
            plot_settings = PlotSettings()
        # Let backends know about yaxis_order so they can draw the correct diagonal
        if plot_settings.yaxis_order is None:
            plot_settings.yaxis_order = nmr_data.yaxis_order
        self.plot_settings = plot_settings

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        # If not set, set number of legend elements to
        # minimum of number of peaks and 5
        if self.plot_settings.num_legend_elements is None:
            self.plot_settings.num_legend_elements = min(npeaks, DEFAULT_MAX_NUM_LEGEND_ELEMENTS)
        
        # Initialize the appropriate backend
        if backend == 'matplotlib':
            self.backend = MatplotlibBackend(ax=ax)
        elif backend == 'plotly':
            if ax is not None:
                self.logger.warning("ax parameter is ignored for Plotly backend")
            self.backend = PlotlyBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'matplotlib' or 'plotly'.")

    def plot(self):
        '''
        Plot the 2D NMR data using the configured backend.

        Returns
        -------
        For matplotlib backend:
            fig : Figure
                The figure object
            ax : Axes
                The axis object
        
        For plotly backend:
            fig : go.Figure
                The Plotly figure object
        '''
        
        # For matplotlib, we need to apply the styled_plot decorator
        if self.backend_name == 'matplotlib':
            return self._plot_matplotlib()
        else:
            return self._plot_generic()
    
    @styled_plot(nmr_base_style, nmr_2D_style)
    def _plot_matplotlib(self):
        """Plot using matplotlib backend with styling"""
        return self._plot_generic()
    
    def _plot_generic(self):
        """Generic plotting logic that works with any backend"""
        
        # Prepare axis labels
        x_axis_label = self.plot_settings.x_axis_label if self.plot_settings.x_axis_label else self.nmr_data.x_axis_label
        y_axis_label = self.plot_settings.y_axis_label if self.plot_settings.y_axis_label else self.nmr_data.y_axis_label
        
        # Normalize xlim and ylim
        if self.plot_settings.xlim:
            xlim = self.plot_settings.xlim
            self.plot_settings.xlim = (min(xlim), max(xlim))
        
        if self.plot_settings.ylim:
            ylim = self.plot_settings.ylim
            self.plot_settings.ylim = (min(ylim), max(ylim))

        # Auto-compute axis limits from peak positions when not user-specified.
        # This MUST happen before drawing the contour/heatmap: the contour grid
        # is padded by up to 50× the broadening beyond the outermost peak, so if
        # we don't set explicit limits first, matplotlib autoscales to the grid
        # extent and the visible area ends up far wider than the peak range.
        # Buffer = 2× the FWHM broadening so the outermost line shape is fully
        # visible with a little breathing room.
        if not self.plot_settings.xlim and len(self.x):
            x_buf = (self.plot_settings.x_broadening or 1.0) * 2
            self.plot_settings.xlim = (
                float(self.x.min()) - x_buf,
                float(self.x.max()) + x_buf,
            )

        if not self.plot_settings.ylim and len(self.y):
            y_buf = (self.plot_settings.y_broadening or 1.0) * 2
            self.plot_settings.ylim = (
                float(self.y.min()) - y_buf,
                float(self.y.max()) + y_buf,
            )

        # Set axis properties first (backends may need this for subsequent operations)
        self.backend.set_axis_properties(
            x_axis_label, y_axis_label,
            self.plot_settings.xlim, self.plot_settings.ylim,
            self.nmr_data.is_shift
        )
        
        # Plot heatmap and contour first (background layers)
        if self.plot_settings.show_heatmap or self.plot_settings.show_contour:
            X, Y, Z = self._get_contour_data_for_backend()
            
            if self.plot_settings.show_heatmap:
                self.backend.plot_heatmap(X, Y, Z, self.plot_settings)
            
            if self.plot_settings.show_contour:
                self.backend.plot_contour(X, Y, Z, self.plot_settings)
        
        # Plot reference lines at peak locations
        if self.plot_settings.show_lines:
            self.backend.plot_axlines(self.x, self.y, self.plot_settings)
        
        # Plot diagonal line for homo-nuclear spectra
        xelem_same_as_yelem = (self.nmr_data.xelement == self.nmr_data.yelement and 
                               self.nmr_data.xelement is not None)
        if xelem_same_as_yelem and self.plot_settings.show_diagonal:
            self.backend.plot_diagonal(self.plot_settings)
        
        # Plot connectors between peaks
        if (self.plot_settings.show_connectors and 
            self.nmr_data.yaxis_order == '2Q' and 
            self.nmr_data.xelement == self.nmr_data.yelement):
            self.backend.plot_connectors(self.x, self.y, self.plot_settings)
        
        # Plot scatter markers
        if self.plot_settings.show_markers:
            colors = self._get_marker_colors()
            if self.plot_settings.scale_markers:
                normalized_sizes = self._normalize_marker_sizes(self.sizes)
            else:
                normalized_sizes = np.full(len(self.sizes), self.plot_settings.max_marker_size)
            
            # Prepare correlation info for legend
            correlation_info = {
                'label': self.nmr_data.correlation_label,
                'unit': self.nmr_data.correlation_unit,
                'fmt': self.nmr_data.correlation_fmt,
                'max_size': np.abs(self.sizes).max(),
                'num_legend_elements': self.plot_settings.num_legend_elements
            }
            
            # Extract labels for hover text
            xlabels = [peak.xlabel for peak in self.nmr_data.peaks]
            ylabels = [peak.ylabel for peak in self.nmr_data.peaks]
            
            self.backend.plot_markers(
                self.x, self.y, normalized_sizes, colors,
                self.plot_settings, correlation_info,
                xlabels, ylabels,
                correlation_values=self.sizes  # Pass actual correlation values
            )
        
        # Plot site annotations/labels
        if self.plot_settings.show_labels:
            xlabels = [peak.xlabel for peak in self.nmr_data.peaks]
            ylabels = [peak.ylabel for peak in self.nmr_data.peaks]
            self.backend.plot_annotations(self.x, self.y, xlabels, ylabels, self.plot_settings)
        
        # Finalize and return
        return self.backend.finalize(self.plot_settings.plot_filename)
    
    def _get_marker_colors(self):
        """Get marker colors from peaks or use settings"""
        if self.plot_settings.marker_color is None:
            colors = [peak.color for peak in self.nmr_data.peaks]
            # If all colors are the same, use single color
            if len(set(colors)) == 1:
                colors = colors[0]
            return colors
        else:
            return self.plot_settings.marker_color
    
    def _get_contour_data_for_backend(self):
        """Delegate contour generation to NMRData2D.get_contour_data().

        Grid limits come from ``PlotSettings.xlim`` / ``PlotSettings.ylim``
        when set explicitly, or are auto-computed from the peak positions when
        those are *None*.

        Note: we intentionally do NOT read the live matplotlib axis limits
        here.  The contour is drawn as the first (background) layer, before
        any markers or other data are plotted, so the axis has not been
        auto-scaled yet.  Reading it would always return matplotlib's default
        (0, 1) initialisation, producing a completely wrong grid range.
        """
        xlims = self.plot_settings.xlim
        ylims = self.plot_settings.ylim

        cd = self.nmr_data.get_contour_data(
            x_broadening=self.plot_settings.x_broadening,
            y_broadening=self.plot_settings.y_broadening,
            grid_max=self.plot_settings.grid_max,
            broadening_type=self.plot_settings.broadening_type,
            grid_size=self.plot_settings.heatmap_grid_size,
            xlims=xlims,
            ylims=ylims,
        )
        return cd.X, cd.Y, cd.Z


    def _normalize_marker_sizes(self, sizes):
        """Normalize marker sizes for consistent display"""
        sizes = np.abs(sizes)
        marker_size_range = np.max(sizes) - np.min(sizes)
        self.logger.info(f"Marker size range: {marker_size_range} {self.nmr_data.correlation_unit}")
        max_abs_marker = np.max(sizes)
        # Normalize such that max marker size is self.plot_settings.max_marker_size
        return sizes / max_abs_marker * self.plot_settings.max_marker_size
