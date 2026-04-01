"""Configuration objects and constants for 2D NMR plotting."""

from collections.abc import Iterable
from typing import Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DEFAULT_MARKER_SIZE = 50
ANNOTATION_LINE_WIDTH = 0.15
ANNOTATION_FONT_SCALE = 0.5
DEFAULT_MAX_NUM_LEGEND_ELEMENTS = 6

MPL_TO_PLOTLY_COLORMAP = {
    "bone_r": "greys_r",
    "bone": "greys",
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "hot": "hot",
    "cool": "ice",
    "gray": "greys",
    "grey": "greys",
}

MPL_TO_PLOTLY_MARKER = {
    "o": "circle",
    "s": "square",
    "^": "triangle-up",
    "v": "triangle-down",
    "D": "diamond",
    "p": "pentagon",
    "*": "star",
    "x": "x-thin",
    "+": "cross-thin",
}

MARKER_INFO = {
    "distance": {"label": "Distance", "unit": "A", "fmt": "{x:.1f}"},
    "inversedistance": {
        "label": "1/Distance",
        "unit": "A^-1",
        "fmt": "{x:.3f}",
    },
    "dipolar": {"label": "Dipolar Coupling", "unit": "kHz", "fmt": "{x:.1f}"},
    "dipolar2": {
        "label": "Dipolar Coupling^2",
        "unit": "kHz^2",
        "fmt": "{x:.1f}",
    },
    "jcoupling": {"label": "J Coupling", "unit": "Hz", "fmt": "{x:.1f}"},
    "fixed": {"label": "Fixed", "unit": "", "fmt": "{x:.1f}"},
    "custom": {"label": "Correlation strength", "unit": "", "fmt": "{x:.1f}"},
    "dipolar_rss": {"label": "Dipolar RSS", "unit": "kHz", "fmt": "{x:.1f}"},
}


class PlotSettings(BaseModel):
    """Validated plotting settings for 2D NMR plots."""

    model_config = ConfigDict(validate_assignment=True)

    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    plot_filename: Optional[str] = None
    show_markers: bool = True
    marker: str = "+"
    marker_linewidth: float = 0.5
    max_marker_size: int = 10
    show_labels: bool = True
    auto_adjust_labels: bool = True
    label_fontsize: Optional[int] = None
    show_lines: bool = True
    show_diagonal: bool = True
    show_connectors: bool = True
    marker_color: Optional[str] = None
    show_legend: bool = False
    num_legend_elements: Optional[int] = None
    show_heatmap: bool = False
    heatmap_levels: Union[int, Iterable[float]] = 20
    show_contour: bool = False
    x_broadening: Optional[float] = None
    y_broadening: Optional[float] = None
    grid_max: Optional[float] = None
    broadening_type: str = "lorentzian"
    heatmap_grid_size: Optional[int] = None
    colormap: str = "bone_r"
    contour_color: str = "C1"
    contour_linewidth: float = 0.2
    intensity_range: Tuple[float, float] = (10.0, 100.0)
    contour_range: Optional[Tuple[float, float]] = None
    heatmap_range: Optional[Tuple[float, float]] = None
    contour_levels: Union[Iterable[float], int] = 10
    scale_markers: bool = True
    yaxis_order: Optional[str] = None

    @field_validator("broadening_type")
    @classmethod
    def _validate_broadening(cls, value: str) -> str:
        value = value.lower()
        if value not in {"gaussian", "lorentzian"}:
            raise ValueError("broadening_type must be 'gaussian' or 'lorentzian'")
        return value

    @field_validator("max_marker_size")
    @classmethod
    def _validate_max_marker_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_marker_size must be > 0")
        return value

    @field_validator("grid_max")
    @classmethod
    def _validate_grid_max(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("grid_max must be > 0")
        return value

    @field_validator("marker")
    @classmethod
    def _validate_marker(cls, value: str) -> str:
        if value not in MPL_TO_PLOTLY_MARKER:
            raise ValueError(f"marker must be one of {sorted(MPL_TO_PLOTLY_MARKER)}")
        return value

    @field_validator("yaxis_order")
    @classmethod
    def _validate_yaxis_order(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in {"1Q", "2Q"}:
            raise ValueError("yaxis_order must be '1Q' or '2Q'")
        return value

    @model_validator(mode="after")
    def _validate_ranges(self) -> "PlotSettings":
        lo, hi = self.intensity_range
        if lo >= hi:
            raise ValueError("intensity_range lower bound must be < upper bound")

        if self.contour_range is not None:
            clo, chi = self.contour_range
            if clo >= chi:
                raise ValueError("contour_range lower bound must be < upper bound")
        else:
            self.contour_range = self.intensity_range

        if self.heatmap_range is not None:
            hlo, hhi = self.heatmap_range
            if hlo >= hhi:
                raise ValueError("heatmap_range lower bound must be < upper bound")
        else:
            self.heatmap_range = self.intensity_range

        if self.heatmap_grid_size is None:
            self.heatmap_grid_size = 600 if self.broadening_type == "lorentzian" else 150
        return self
