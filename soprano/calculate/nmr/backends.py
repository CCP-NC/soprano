"""Plot backend implementations for 2D NMR plotting."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.axes import Axes

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from soprano.calculate.nmr.config import (
    ANNOTATION_FONT_SCALE,
    ANNOTATION_LINE_WIDTH,
    MPL_TO_PLOTLY_COLORMAP,
    MPL_TO_PLOTLY_MARKER,
)

def _resolve_levels(
    Z: np.ndarray,
    levels: Union[int, Iterable[float]],
    contour_range: Tuple[float, float],
) -> np.ndarray:
    """Return concrete contour level values from either a count or explicit list.

    Parameters
    ----------
    Z : np.ndarray
        The intensity grid (used only when *levels* is an integer).
    levels : int or iterable of float
        *int* – generate this many evenly-spaced levels inside *contour_range*.
        *iterable* – used directly as absolute intensity values;
        *contour_range* is then ignored.
    contour_range : (float, float)
        ``(lo, hi)`` expressed as **percentages of Z.max()** (0–100 scale),
        applied only when *levels* is an integer.

    Returns
    -------
    np.ndarray
        1-D array of level values.
    """
    if isinstance(levels, (int, float)):
        z_max = float(Z.max())
        lo = contour_range[0] / 100.0 * z_max
        hi = contour_range[1] / 100.0 * z_max
        return np.linspace(lo, hi, int(levels))
    else:
        return np.asarray(levels)


class PlotBackend(ABC):
    """Abstract base class for plot backends"""
    
    @abstractmethod
    def create_figure(self):
        """Create a new figure/chart object"""
        pass
    
    @abstractmethod
    def plot_markers(self, x: np.ndarray, y: np.ndarray, sizes: np.ndarray, 
                    colors: Union[str, list], settings: 'PlotSettings',
                    correlation_info: Optional[dict] = None,
                    xlabels: Optional[list] = None,
                    ylabels: Optional[list] = None,
                    correlation_values: Optional[np.ndarray] = None) -> Any:
        """Plot scatter markers
        
        Args:
            x: x coordinates
            y: y coordinates
            sizes: marker sizes (normalized for plotting)
            colors: marker colors (single color or list)
            settings: PlotSettings object
            correlation_info: Optional dict with correlation metadata for legend
            xlabels: Optional list of x-axis labels for hover text
            ylabels: Optional list of y-axis labels for hover text
            correlation_values: Optional array of actual correlation values (unnormalized)
        """
        pass
    
    @abstractmethod
    def plot_heatmap(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                     settings: 'PlotSettings') -> Any:
        """Plot heatmap contour fill"""
        pass
    
    @abstractmethod
    def plot_contour(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                    settings: 'PlotSettings') -> Any:
        """Plot contour lines"""
        pass
    
    @abstractmethod
    def plot_connectors(self, x: np.ndarray, y: np.ndarray, 
                       settings: 'PlotSettings') -> Any:
        """Plot connecting lines between points"""
        pass
    
    @abstractmethod
    def plot_axlines(self, x: np.ndarray, y: np.ndarray, 
                    settings: 'PlotSettings') -> Any:
        """Plot reference lines at peak positions"""
        pass
    
    @abstractmethod
    def plot_diagonal(self, settings: 'PlotSettings') -> Any:
        """Plot diagonal line"""
        pass
    
    @abstractmethod
    def plot_annotations(self, x: np.ndarray, y: np.ndarray, 
                        xlabels: list, ylabels: list, 
                        settings: 'PlotSettings') -> Any:
        """Plot annotations/labels"""
        pass
    
    @abstractmethod
    def set_axis_properties(self, xlabel: str, ylabel: str, 
                          xlim: Optional[Tuple[float, float]], 
                          ylim: Optional[Tuple[float, float]], 
                          invert_axes: bool) -> None:
        """Set axis labels, limits, and inversions"""
        pass
    
    @abstractmethod
    def finalize(self, filename: Optional[str] = None) -> Any:
        """Finalize and return the plot object"""
        pass


class MatplotlibBackend(PlotBackend):
    """Matplotlib backend implementation (preserves original functionality)"""
    
    def __init__(self, ax: Optional[Axes] = None):
        """Initialize with optional existing axis"""
        if ax is None:
            self.fig, self.ax = plt.subplots()
        elif isinstance(ax, Axes):
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            raise TypeError("ax must be an Axes object or None.")
        
        self.logger = logging.getLogger(__name__)
    
    def create_figure(self):
        """Figure already created in __init__"""
        return self.fig, self.ax
    
    def plot_markers(self, x, y, sizes, colors, settings, correlation_info=None, xlabels=None, ylabels=None, correlation_values=None):
        """Plot scatter markers using matplotlib"""
        scatter = self.ax.scatter(
            x, y, s=sizes, c=colors,
            marker=settings.marker,
            linewidths=settings.marker_linewidth,
            zorder=10
        )
        
        # Add legend if requested
        if settings.show_legend and correlation_info:
            kw = dict(
                prop="sizes", 
                num=correlation_info.get('num_legend_elements', 5),
                color=colors if isinstance(colors, str) else colors[0],
                fmt=correlation_info.get('fmt', '{x:.1f}') + f" {correlation_info.get('unit', '')}",
                func=lambda s: s * correlation_info.get('max_size', 1) / settings.max_marker_size
            )
            handles, labels = scatter.legend_elements(**kw)
            self.ax.legend(
                handles, labels,
                title=correlation_info.get('label', 'Correlation'),
                fancybox=True,
                framealpha=0.8
            ).set_zorder(12)
        
        return scatter
    
    def plot_heatmap(self, X, Y, Z, settings):
        """Plot heatmap using matplotlib contourf"""
        levels = _resolve_levels(Z, settings.heatmap_levels, settings.heatmap_range)
        return self.ax.contourf(X, Y, Z, cmap=settings.colormap,
                               zorder=-1, levels=levels)

    def plot_contour(self, X, Y, Z, settings):
        """Plot contour lines using matplotlib"""
        levels = _resolve_levels(Z, settings.contour_levels, settings.contour_range)
        return self.ax.contour(
            X, Y, Z,
            colors=settings.contour_color,
            linewidths=settings.contour_linewidth,
            levels=levels
        )
    
    def plot_connectors(self, x, y, settings):
        """Plot connecting lines between points with same y value"""
        y_order = np.argsort(y)
        for i, idx in enumerate(y_order):
            if i > 0 and np.isclose(y[idx], y[y_order[i-1]], atol=1e-6):
                self.ax.plot(
                    [x[idx], x[y_order[i-1]]],
                    [y[idx], y[y_order[i-1]]],
                    c='0.25', lw=0.75, ls='-', zorder=1
                )
    
    def plot_axlines(self, x, y, settings):
        """Plot reference lines at peak positions"""
        xticks = np.unique(np.round(x, 6))
        yticks = np.unique(np.round(y, 6))
        
        for x_val in xticks:
            self.ax.axvline(x_val, zorder=0)
        for y_val in yticks:
            self.ax.axhline(y_val, zorder=0)
    
    def plot_diagonal(self, settings):
        """Plot diagonal line.

        For 2Q (DQ/SQ) mode the diagonal marks the auto-correlation condition
        DQ = 2 × SQ, i.e. y = 2x.  For all other modes the conventional
        y = x identity line is drawn.
        """
        xlims = self.ax.get_xlim()
        if getattr(settings, 'yaxis_order', None) == '2Q':
            # DQ/SQ diagonal: y = 2x
            y_vals = [2 * xlims[0], 2 * xlims[1]]
        else:
            y_vals = list(self.ax.get_ylim())
        self.ax.plot(xlims, y_vals, ls='--', c='k', lw=1, alpha=0.2)
    
    def plot_annotations(self, x, y, xlabels, ylabels, settings):
        """Plot annotations with arrows (matplotlib approach)"""
        font_size = settings.label_fontsize
        if font_size is None:
            font_size = self.ax.xaxis.label.get_fontsize() * ANNOTATION_FONT_SCALE
        
        # Get unique labels and positions
        xlabels_unique, xidx = np.unique(xlabels, return_index=True)
        ylabels_unique, yidx = np.unique(ylabels, return_index=True)
        xpos = x[xidx]
        ypos = y[yidx]
        
        labels_offset = 0.10
        armA = 15 if settings.plot_filename is None else (3 if settings.plot_filename.endswith('.pdf') else 20)
        armB = 15 if settings.plot_filename is None else (5 if settings.plot_filename.endswith('.pdf') else 30)
        
        annotations = []
        
        # X labels at top
        texts = []
        for i, xlabel in enumerate(xlabels_unique):
            an = self.ax.annotate(
                xlabel,
                xy=(xpos[i], 1.0),
                xycoords=('data', 'axes fraction'),
                xytext=(xpos[i], 1+labels_offset),
                textcoords=('data', 'axes fraction'),
                fontsize=font_size,
                ha='center', va='bottom',
                rotation=90,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc,angleA=-90,armA={armA},angleB=90,armB={armB},rad=0",
                    relpos=(0.5, 0.0),
                    lw=ANNOTATION_LINE_WIDTH,
                    shrinkA=0.0, shrinkB=0.0,
                ),
            )
            texts.append(an)
        
        if settings.auto_adjust_labels:
            adjust_text(
                texts, ensure_inside_axes=False, avoid_self=False,
                force_pull=(0.0, 0.0), force_text=(0.3, 0.0),
                force_explode=(1.5, 0.0), expand=(1.3, 1.0), max_move=2,
            )
        annotations.extend(texts)
        
        # Y labels at right
        texts = []
        for i, ylabel in enumerate(ylabels_unique):
            an = self.ax.annotate(
                ylabel,
                xy=(1.0, ypos[i]),
                xycoords=('axes fraction', 'data'),
                xytext=(1+labels_offset, ypos[i]),
                textcoords=('axes fraction', 'data'),
                fontsize=font_size,
                ha='left', va='center',
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"arc,angleA=180,armA={armA},angleB=0,armB={armB},rad=0",
                    relpos=(0.0, 0.5),
                    lw=ANNOTATION_LINE_WIDTH,
                    shrinkA=0.0, shrinkB=0.0,
                ),
            )
            texts.append(an)
        
        if settings.auto_adjust_labels:
            adjust_text(
                texts, ensure_inside_axes=False, avoid_self=False,
                force_pull=(0.0, 0.0), force_text=(0.4, 0.8),
                force_explode=(0.0, 1.2), expand=(1.0, 1.8), max_move=1,
            )
        annotations.extend(texts)
        
        return annotations
    
    def set_axis_properties(self, xlabel, ylabel, xlim, ylim, invert_axes):
        """Set axis properties"""
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        if xlim:
            self.ax.set_xlim(min(xlim), max(xlim))
        if ylim:
            self.ax.set_ylim(min(ylim), max(ylim))
        
        if invert_axes:
            self.ax.invert_xaxis()
            self.ax.invert_yaxis()
    
    def finalize(self, filename=None):
        """Finalize the plot"""
        self.fig.tight_layout()
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        return self.fig, self.ax


class PlotlyBackend(PlotBackend):
    """Plotly backend implementation for interactive web-based plots with full contour support"""
    
    def __init__(self):
        """Initialize Plotly backend"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyBackend. Install with: pip install plotly")
        
        self.xlabel = ""
        self.ylabel = ""
        self.xlim = None
        self.ylim = None
        self.invert_axes = False
        self.logger = logging.getLogger(__name__)
        # Create the figure immediately
        self.fig = go.Figure()
    
    def create_figure(self):
        """Create a Plotly figure"""
        if self.fig is None:
            self.fig = go.Figure()
        return self.fig
    
    def plot_markers(self, x, y, sizes, colors, settings, correlation_info=None, xlabels=None, ylabels=None, correlation_values=None):
        """Plot scatter markers using Plotly"""
        # Handle colors
        if isinstance(colors, str):
            marker_colors = colors
        else:
            marker_colors = colors
        
        # Map matplotlib marker to Plotly symbol (using -open versions for hollow markers)
        symbol = MPL_TO_PLOTLY_MARKER.get(settings.marker, 'circle-open')
        
        # Normalize sizes for Plotly (scale to reasonable pixel values)
        size_scale = settings.max_marker_size / np.max(sizes) if np.max(sizes) > 0 else 1
        plotly_sizes = sizes * size_scale
        
        # Use actual correlation values if provided, otherwise fall back to sizes
        values_to_display = correlation_values if correlation_values is not None else sizes
        
        # Get format string and unit from correlation_info
        if correlation_info:
            fmt = correlation_info.get('fmt', '{x:.2f}')
            unit = correlation_info.get('unit', '')
            label = correlation_info.get('label', 'Strength')
        else:
            fmt = '{x:.2f}'
            unit = ''
            label = 'Strength'
        
        # Create hover text with labels if available
        if xlabels is not None and ylabels is not None:
            hovertext = [f"{xl}--{yl}<br>x: {xi:.2f}<br>y: {yi:.2f}<br>{label}: {fmt.format(x=vi)} {unit}" 
                         for xl, yl, xi, yi, vi in zip(xlabels, ylabels, x, y, values_to_display)]
        else:
            hovertext = [f"x: {xi:.2f}<br>y: {yi:.2f}<br>{label}: {fmt.format(x=vi)} {unit}" 
                         for xi, yi, vi in zip(x, y, values_to_display)]
        
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=plotly_sizes,
                color='rgba(0,0,0,0)',  # Transparent fill for hollow markers
                symbol=symbol,
                line=dict(width=settings.marker_linewidth, color=marker_colors)
            ),
            hovertext=hovertext,
            hoverinfo='text',
            showlegend=settings.show_legend,
            name=correlation_info.get('label', 'Correlation') if correlation_info else 'Peaks'
        )
        
        self.fig.add_trace(trace)
        return trace
    
    def plot_heatmap(self, X, Y, Z, settings):
        """Plot heatmap using Plotly"""
        colorscale = MPL_TO_PLOTLY_COLORMAP.get(settings.colormap, settings.colormap)
        levels = _resolve_levels(Z, settings.heatmap_levels, settings.heatmap_range)

        trace = go.Heatmap(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale=colorscale,
            zmin=float(levels[0]),
            zmax=float(levels[-1]),
            showscale=False,
            hoverinfo='skip'
        )

        # Insert as first trace (background)
        self.fig.add_trace(trace)
        # Move to back
        self.fig.data = (self.fig.data[-1],) + self.fig.data[:-1]
        return trace

    def plot_contour(self, X, Y, Z, settings):
        """Plot contour lines using Plotly"""
        colorscale = MPL_TO_PLOTLY_COLORMAP.get(settings.colormap, settings.colormap)
        levels = _resolve_levels(Z, settings.contour_levels, settings.contour_range)
        n = len(levels)
        size = float(levels[-1] - levels[0]) / (n - 1) if n > 1 else 0.0

        trace = go.Contour(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale=colorscale,
            showscale=False,
            contours=dict(
                start=float(levels[0]),
                end=float(levels[-1]),
                size=size,
                coloring='lines',
                showlabels=True,
                labelfont=dict(size=8)
            ),
            line=dict(width=settings.contour_linewidth),
            hoverinfo='x+y+z'
        )

        self.fig.add_trace(trace)
        return trace
    
    def plot_connectors(self, x, y, settings):
        """Plot connecting lines between points with same y value"""
        y_order = np.argsort(y)
        
        for i, idx in enumerate(y_order):
            if i > 0 and np.isclose(y[idx], y[y_order[i-1]], atol=1e-6):
                trace = go.Scatter(
                    x=[x[y_order[i-1]], x[idx]],
                    y=[y[y_order[i-1]], y[idx]],
                    mode='lines',
                    line=dict(color='gray', width=0.75),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                )
                self.fig.add_trace(trace)
    
    def plot_axlines(self, x, y, settings):
        """Plot reference lines at peak positions"""
        xticks = np.unique(np.round(x, 6))
        yticks = np.unique(np.round(y, 6))
        
        # Add vertical lines
        for xt in xticks:
            self.fig.add_vline(
                x=xt,
                line=dict(color='lightgray', width=0.5),
                opacity=0.3
            )
        
        # Add horizontal lines
        for yt in yticks:
            self.fig.add_hline(
                y=yt,
                line=dict(color='lightgray', width=0.5),
                opacity=0.3
            )
    
    def plot_diagonal(self, settings):
        """Plot diagonal line.

        For 2Q (DQ/SQ) mode the diagonal marks the auto-correlation condition
        DQ = 2 × SQ, i.e. y = 2x.  For all other modes the conventional
        y = x identity line is drawn.
        """
        if self.xlim:
            x_vals = [self.xlim[0], self.xlim[1]]
            if getattr(settings, 'yaxis_order', None) == '2Q':
                # DQ/SQ diagonal: y = 2x
                y_vals = [2 * self.xlim[0], 2 * self.xlim[1]]
            elif self.ylim:
                y_vals = [self.ylim[0], self.ylim[1]]
            else:
                y_vals = x_vals
            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                opacity=0.2,
                showlegend=False,
                hoverinfo='skip'
            )
            self.fig.add_trace(trace)
            return trace
        
        return None
    
    def plot_annotations(self, x, y, xlabels, ylabels, settings):
        """Plot text labels as annotations"""
        # Get unique labels
        xlabels_unique, xidx = np.unique(xlabels, return_index=True)
        ylabels_unique, yidx = np.unique(ylabels, return_index=True)
        xpos = x[xidx]
        ypos = y[yidx]
        
        font_size = settings.label_fontsize or 10
        
        # X labels (top)
        y_top = self.ylim[1] if self.ylim else max(y)
        for xp, label in zip(xpos, xlabels_unique):
            self.fig.add_annotation(
                x=xp,
                y=y_top,
                text=label,
                showarrow=False,
                textangle=270,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=font_size)
            )
        
        # Y labels (right)
        x_right = self.xlim[1] if self.xlim else max(x)
        for yp, label in zip(ypos, ylabels_unique):
            self.fig.add_annotation(
                x=x_right,
                y=yp,
                text=label,
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=font_size)
            )
    
    def set_axis_properties(self, xlabel, ylabel, xlim, ylim, invert_axes):
        """Store axis properties for later application"""
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.invert_axes = invert_axes
    
    def finalize(self, filename=None):
        """Apply final layout settings and return figure"""
        if self.fig is None:
            raise ValueError("No figure to finalize")
        
        # Determine axis ranges
        xrange = None
        yrange = None
        x_autorange = True
        y_autorange = True
        
        if self.xlim:
            xrange = [self.xlim[1], self.xlim[0]] if self.invert_axes else list(self.xlim)
            x_autorange = False
        
        if self.ylim:
            yrange = [self.ylim[1], self.ylim[0]] if self.invert_axes else list(self.ylim)
            y_autorange = False
        
        # Set autorange to 'reversed' when invert_axes is True and limits are auto
        x_autorange_setting = 'reversed' if (self.invert_axes and x_autorange) else x_autorange
        y_autorange_setting = 'reversed' if (self.invert_axes and y_autorange) else y_autorange
        
        # Update layout
        self.fig.update_layout(
            xaxis=dict(
                title=self.xlabel,
                range=xrange,
                autorange=x_autorange_setting,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                title=self.ylabel,
                range=yrange,
                autorange=y_autorange_setting,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            width=700,
            height=600,
            hovermode='closest',
            plot_bgcolor='white',
            showlegend=True
        )
        
        # Save if filename provided
        if filename:
            if filename.endswith('.html'):
                self.fig.write_html(filename)
            elif filename.endswith('.json'):
                self.fig.write_json(filename)
            elif filename.endswith('.png'):
                self.fig.write_image(filename)
            elif filename.endswith('.svg'):
                self.fig.write_image(filename)
            elif filename.endswith('.pdf'):
                self.fig.write_image(filename)
            else:
                self.logger.warning(f"Unsupported file format: {filename}")
        
        return self.fig


