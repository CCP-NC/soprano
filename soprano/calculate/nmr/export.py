"""Contour export helpers for 2D NMR data.

This module owns all contour-grid serialization logic (SIMPSON/NPZ/CSV/JSON)
and is the separation-of-concerns boundary between data extraction and file I/O.

Public entrypoints:
    export_contour_data
    ExportConfig
    guess_format_from_path
    compute_larmor_frequency
    compute_b0_from_spectrometer_freq

Compatibility:
    NMRData2D.export_contour_data delegates to this module.
"""

import csv
import json
from dataclasses import dataclass, replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple, TYPE_CHECKING

import numpy as np

from soprano.nmr.utils import compute_b0_from_spectrometer_freq, compute_larmor_frequency

if TYPE_CHECKING:
    from soprano.calculate.nmr.utils import ContourData, Peak2D


# ---------------------------------------------------------------------------
# Extension → format mapping
# ---------------------------------------------------------------------------
_EXT_TO_FMT = {
    ".spe": "simpson",
    ".sim": "simpson",
    ".npz": "npz",
    ".csv": "csv",
    ".json": "json",
    ".ssnake": "json",
    ".txt": "plain",
}


# ---------------------------------------------------------------------------
# Larmor frequency helpers
# ---------------------------------------------------------------------------


def _resolve_larmor_freqs(
    nmr_data: Any,
    x_larmor: Optional[float],
    y_larmor: Optional[float],
    b0_tesla: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (x_larmor, y_larmor) with auto-computation if needed.

    Auto-computation requires ``b0_tesla`` to be provided.  When it is None
    and the element is known, a warning is emitted and the frequency is left
    as None (callers fall back to ppm-based output or raise as appropriate).
    """
    xelement = getattr(nmr_data, "xelement", None)
    yelement = getattr(nmr_data, "yelement", None)

    _needs_auto = (x_larmor is None and xelement is not None) or (
        y_larmor is None and yelement is not None
    )
    if _needs_auto and b0_tesla is None:
        _log_warning(
            nmr_data,
            "No B0 field or spectrometer frequency specified; cannot "
            "auto-compute Larmor frequencies. Pass b0_field_tesla or "
            "spectrometer_freq_mhz (ExportConfig) / --b0-field-tesla or "
            "--spectrometer-freq (CLI).",
        )

    def _resolve(freq: Optional[float], element: Optional[str]) -> Optional[float]:
        if freq is not None:
            return freq
        if element is None or b0_tesla is None:
            return None
        try:
            return compute_larmor_frequency(element, b0_tesla=b0_tesla)
        except ValueError:
            return None

    x_resolved = _resolve(x_larmor, xelement)
    y_resolved = _resolve(y_larmor, yelement)

    # For homonuclear spectra, y falls back to x
    if y_resolved is None and yelement == xelement and x_resolved is not None:
        y_resolved = x_resolved

    return x_resolved, y_resolved


# ---------------------------------------------------------------------------
# Export configuration
# ---------------------------------------------------------------------------
@dataclass
class ExportConfig:
    """Configuration for contour data export.

    All parameters have sensible defaults.  Only override what you need.
    """

    # Grid parameters
    x_broadening: Optional[float] = None
    y_broadening: Optional[float] = None
    grid_max: Optional[float] = None
    broadening_type: str = "lorentzian"
    grid_size: int = 500
    xlims: Optional[Tuple[float, float]] = None
    ylims: Optional[Tuple[float, float]] = None
    use_signed: bool = False

    # Spectrometer parameters (auto-computed if None)
    x_larmor_freq_mhz: Optional[float] = None
    y_larmor_freq_mhz: Optional[float] = None
    b0_field_tesla: Optional[float] = None
    spectrometer_freq_mhz: Optional[float] = None  # ¹H freq; mutually exclusive with b0_field_tesla

    # Output options
    include_peaks: bool = True

    def __post_init__(self) -> None:
        if self.b0_field_tesla is not None and self.spectrometer_freq_mhz is not None:
            raise ValueError(
                "Specify either b0_field_tesla or spectrometer_freq_mhz, not both. "
                f"Got b0_field_tesla={self.b0_field_tesla} T and "
                f"spectrometer_freq_mhz={self.spectrometer_freq_mhz} MHz."
            )

    def resolve_larmor_freqs(self, nmr_data: Any) -> Tuple[Optional[float], Optional[float]]:
        """Return resolved Larmor frequencies, using auto-computation if needed."""
        b0 = self.b0_field_tesla
        if b0 is None and self.spectrometer_freq_mhz is not None:
            b0 = compute_b0_from_spectrometer_freq(self.spectrometer_freq_mhz)
        return _resolve_larmor_freqs(
            nmr_data,
            self.x_larmor_freq_mhz,
            self.y_larmor_freq_mhz,
            b0,
        )

    def has_rendering_overrides(self) -> bool:
        """Return True if any grid-rendering field differs from the default."""
        d = ExportConfig()
        return (
            self.x_broadening != d.x_broadening
            or self.y_broadening != d.y_broadening
            or self.grid_max != d.grid_max
            or self.broadening_type != d.broadening_type
            or self.grid_size != d.grid_size
            or self.xlims != d.xlims
            or self.ylims != d.ylims
            or self.use_signed != d.use_signed
        )


# ---------------------------------------------------------------------------
# Protocol (kept for type-checking)
# ---------------------------------------------------------------------------
class _NMRData2DExportProtocol(Protocol):
    def get_contour_data(
        self,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        grid_max: Optional[float] = None,
        broadening_type: str = "lorentzian",
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
        use_signed: bool = False,
    ) -> "ContourData": ...

    def get_peaks(self) -> list["Peak2D"]: ...


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _log_info(nmr_data: Any, message: str) -> None:
    logger = getattr(nmr_data, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)


def _log_warning(nmr_data: Any, message: str) -> None:
    logger = getattr(nmr_data, "logger", None)
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(message)


def _write_peaks_csv(peaks: list["Peak2D"], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_ppm", "y_ppm", "xlabel", "ylabel", "correlation_strength"])
        for p in peaks:
            writer.writerow([p.x, p.y, p.xlabel, p.ylabel, p.correlation_strength])


# ---------------------------------------------------------------------------
# Format inference
# ---------------------------------------------------------------------------
def guess_format_from_path(path: str) -> str:
    """Guess export format from file extension.

    Parameters
    ----------
    path : str
        Output file path.

    Returns
    -------
    str
        Format name (e.g. ``'simpson'``, ``'json'``).

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    ext = Path(path).suffix.lower()
    fmt = _EXT_TO_FMT.get(ext)
    if fmt is None:
        known = ", ".join(sorted(set(_EXT_TO_FMT.values())))
        raise ValueError(
            f"Cannot infer export format from extension '{ext}'. "
            f"Known formats: {known} (plus 'ssnake' as alias for 'json'). "
            f"Use fmt=... to specify explicitly."
        )
    return fmt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_contour_data(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    fmt: Optional[str] = None,
    config: Optional[ExportConfig] = None,
    contour_data: Optional["ContourData"] = None,
) -> None:
    """Export contour data for an NMRData2D instance.

    Parameters
    ----------
    nmr_data : NMRData2D
        The 2D NMR data object to export.
    path : str
        Output file path.
    fmt : str, optional
        Export format.  If *None*, inferred from the file extension.
        Supported: ``'simpson'``, ``'npz'``, ``'csv'``, ``'json'``,
        ``'plain'``, ``'bruker'``.  ``'ssnake'`` is an alias for ``'json'``.
        ``'bruker'`` outputs a directory tree (not a single file).
    config : ExportConfig, optional
        Export configuration.  Uses :class:`ExportConfig` defaults when
        not provided.
    contour_data : ContourData, optional
        Pre-computed contour grid.  When provided, skips
        :meth:`get_contour_data` entirely — useful when the caller already
        holds the grid that was used for plotting.
    """
    effective = dc_replace(config) if config is not None else ExportConfig()

    # Resolve format
    if fmt is None:
        fmt = guess_format_from_path(path)
    fmt = fmt.lower().strip()

    # Compute contour data (or reuse pre-computed grid)
    if contour_data is not None:
        cd = contour_data
    else:
        cd = nmr_data.get_contour_data(
            x_broadening=effective.x_broadening,
            y_broadening=effective.y_broadening,
            grid_max=effective.grid_max,
            broadening_type=effective.broadening_type,
            grid_size=effective.grid_size,
            xlims=effective.xlims,
            ylims=effective.ylims,
            use_signed=effective.use_signed,
        )

    # Dispatch — Larmor frequencies resolved only for formats that need them
    if fmt == "simpson":
        x_larmor, y_larmor = effective.resolve_larmor_freqs(nmr_data)
        _export_simpson(nmr_data, path, cd, x_larmor, y_larmor, include_peaks=effective.include_peaks)
    elif fmt == "npz":
        _export_npz(nmr_data, path, cd, include_peaks=effective.include_peaks)
    elif fmt == "csv":
        _export_csv_grid(nmr_data, path, cd, include_peaks=effective.include_peaks)
    elif fmt in ("json", "ssnake"):
        x_larmor, y_larmor = effective.resolve_larmor_freqs(nmr_data)
        _export_json_ssnake(nmr_data, path, cd, x_larmor, y_larmor, include_peaks=effective.include_peaks)
    elif fmt == "plain":
        _export_plain(nmr_data, path, cd, include_peaks=effective.include_peaks)
    elif fmt == "bruker":
        x_larmor, y_larmor = effective.resolve_larmor_freqs(nmr_data)
        _export_bruker(
            nmr_data, path, cd, x_larmor, y_larmor,
            include_peaks=effective.include_peaks,
        )
    else:
        raise ValueError(
            f"Unknown export format '{fmt}'. "
            f"Choose from {', '.join(sorted(set(_EXT_TO_FMT.values())))} "
            f"(or 'ssnake' as an alias for 'json', 'bruker' for Bruker TopSpin)."
        )

    _log_info(nmr_data, f"Exported contour data to '{path}' (format={fmt}).")


# ---------------------------------------------------------------------------
# Per-format exporters
# ---------------------------------------------------------------------------
def _export_simpson(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    x_larmor_freq_mhz: Optional[float],
    y_larmor_freq_mhz: Optional[float],
    include_peaks: bool = True,
) -> None:
    """Write a SIMPSON TEXT (.spe) file readable by nmrglue and ssNake."""
    Z = cd.Z
    ni, np_ = Z.shape
    sw_ppm = cd.xlims[1] - cd.xlims[0]
    sw1_ppm = cd.ylims[1] - cd.ylims[0]

    y_freq = y_larmor_freq_mhz if y_larmor_freq_mhz is not None else x_larmor_freq_mhz

    if x_larmor_freq_mhz is not None:
        sw = sw_ppm * x_larmor_freq_mhz
        sw1 = sw1_ppm * y_freq
        sw_unit = "Hz"
    else:
        sw = sw_ppm
        sw1 = sw1_ppm
        sw_unit = "ppm"
        _log_warning(
            nmr_data,
            "Exporting SIMPSON .spe without Larmor frequencies: SW/SW1 are "
            "written in ppm. ssNake cannot select ppm as a unit without "
            "spectrometer frequencies. Provide --x-larmor-freq or --b0-field-tesla "
            "to fix this."
        )

    # Build metadata block
    meta_lines = [
        "# Exported by Soprano NMRData2D.export_contour_data",
        f"# timestamp={datetime.now(timezone.utc).isoformat()}",
        f"# SW_unit={sw_unit}",
    ]
    if x_larmor_freq_mhz is not None:
        meta_lines.extend([
            f"# SPECFREQ_x={x_larmor_freq_mhz:.6g} MHz  (direct dim)",
            f"# SPECFREQ_y={y_freq:.6g} MHz  (indirect dim)",
            "# ssNake: Axes -> Edit axes, set carriers to these values",
        ])
    meta_lines.extend([
        f"# x_broadening={cd.x_broadening:.6g} ppm",
        f"# y_broadening={cd.y_broadening:.6g} ppm",
        f"# broadening_type={cd.broadening_type}",
        f"# xlims_ppm={cd.xlims[0]:.6g} {cd.xlims[1]:.6g}",
        f"# ylims_ppm={cd.ylims[0]:.6g} {cd.ylims[1]:.6g}",
    ])

    with open(path, "w") as f:
        f.write("SIMP\n")
        f.write(f"NP={np_}\n")
        f.write(f"NI={ni}\n")
        f.write(f"SW={sw:.8g}\n")
        f.write(f"SW1={sw1:.8g}\n")
        f.write("TYPE=SPE\n")
        for line in meta_lines:
            f.write(line + "\n")
        f.write("DATA\n")
        for i in range(ni):
            for j in range(np_):
                f.write(f"{Z[i, j]:.8g} 0.0\n")
        f.write("END")

    if include_peaks:
        peaks_path = path + ".peaks.csv"
        _write_peaks_csv(nmr_data.get_peaks(), peaks_path)
        _log_info(nmr_data, f"Peak list written to '{peaks_path}'.")


# ---------------------------------------------------------------------------
# Per-format exporters
# ---------------------------------------------------------------------------
def _export_npz(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    include_peaks: bool = True,
) -> None:
    """Write a NumPy compressed archive with the grid and metadata."""
    arrays: dict = dict(
        X=cd.X,
        Y=cd.Y,
        Z=cd.Z,
        x_broadening=cd.x_broadening,
        y_broadening=cd.y_broadening,
        broadening_type=np.bytes_(cd.broadening_type),
        xlims=np.array(cd.xlims),
        ylims=np.array(cd.ylims),
        timestamp=np.bytes_(datetime.now(timezone.utc).isoformat()),
    )
    if include_peaks:
        peaks = nmr_data.get_peaks()
        arrays.update(
            peak_x=np.array([p.x for p in peaks]),
            peak_y=np.array([p.y for p in peaks]),
            peak_strength=np.array([p.correlation_strength for p in peaks]),
            peak_xlabels=np.array([p.xlabel for p in peaks]),
            peak_ylabels=np.array([p.ylabel for p in peaks]),
        )
        _log_info(nmr_data, f"Peak data embedded in '{path}'.")
    np.savez_compressed(path, **arrays)


def _export_csv_grid(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    include_peaks: bool = True,
) -> None:
    """Write a flat CSV with columns x, y, intensity."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_ppm", "y_ppm", "intensity"])
        ni, np_ = cd.Z.shape
        for i in range(ni):
            for j in range(np_):
                writer.writerow([cd.X[i, j], cd.Y[i, j], cd.Z[i, j]])

    if include_peaks:
        peaks_path = path + ".peaks.csv"
        peaks = nmr_data.get_peaks()
        with open(peaks_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_ppm", "y_ppm", "xlabel", "ylabel", "correlation_strength"])
            for p in peaks:
                writer.writerow([p.x, p.y, p.xlabel, p.ylabel, p.correlation_strength])
        _log_info(nmr_data, f"Peak list written to '{peaks_path}'.")


def _export_plain(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    include_peaks: bool = True,
) -> None:
    """Write a plain-text file with space-separated x, y, z columns."""
    ni, np_ = cd.Z.shape
    flat = np.column_stack([
        cd.X.ravel(),
        cd.Y.ravel(),
        cd.Z.ravel(),
    ])
    header = (
        f"# Exported by Soprano\n"
        f"# timestamp={datetime.now(timezone.utc).isoformat()}\n"
        f"# x_broadening={cd.x_broadening:.6g} ppm\n"
        f"# y_broadening={cd.y_broadening:.6g} ppm\n"
        f"# xlims_ppm={cd.xlims[0]:.6g} {cd.xlims[1]:.6g}\n"
        f"# ylims_ppm={cd.ylims[0]:.6g} {cd.ylims[1]:.6g}\n"
        f"# shape=({ni}, {np_})\n"
        f"# x_ppm y_ppm intensity"
    )
    np.savetxt(path, flat, header=header, comments="", fmt="%.8g")
    _log_info(nmr_data, f"Plain text contour data written to '{path}'.")

    if include_peaks:
        peaks_path = path + ".peaks.csv"
        peaks = nmr_data.get_peaks()
        with open(peaks_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_ppm", "y_ppm", "xlabel", "ylabel", "correlation_strength"])
            for p in peaks:
                writer.writerow([p.x, p.y, p.xlabel, p.ylabel, p.correlation_strength])
        _log_info(nmr_data, f"Peak list written to '{peaks_path}'.")


def _export_json_ssnake(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    x_larmor_freq_mhz: Optional[float],
    y_larmor_freq_mhz: Optional[float],
    include_peaks: bool = True,
) -> None:
    """Write an ssNake-native JSON file with Larmor frequencies embedded."""
    if x_larmor_freq_mhz is None:
        x_el = getattr(nmr_data, "xelement", "x")
        raise ValueError(
            f"Cannot determine Larmor frequency for '{x_el}'. "
            f"Provide x_larmor_freq_mhz, b0_field_tesla, or ensure the element "
            f"has known gyromagnetic ratio data."
        )
    y_freq_mhz = y_larmor_freq_mhz if y_larmor_freq_mhz is not None else x_larmor_freq_mhz

    x_freq_hz = x_larmor_freq_mhz * 1e6
    y_freq_hz = y_freq_mhz * 1e6

    sw_x_hz = (cd.xlims[1] - cd.xlims[0]) * x_larmor_freq_mhz
    sw_y_hz = (cd.ylims[1] - cd.ylims[0]) * y_freq_mhz

    ref_x = x_freq_hz
    ref_y = y_freq_hz

    ni, np_ = cd.Z.shape
    data_3d = cd.Z.reshape(1, ni, np_)
    flat_real = data_3d.tolist()
    flat_imag = np.zeros((1, ni, np_)).tolist()

    xax_x = (np.linspace(cd.xlims[0], cd.xlims[1], np_) * x_larmor_freq_mhz).tolist()
    xax_y = (np.linspace(cd.ylims[0], cd.ylims[1], ni) * y_freq_mhz).tolist()

    struct: dict[str, Any] = {
        "dataReal": flat_real,
        "dataImag": flat_imag,
        "hyper": [0],
        "freq": [y_freq_hz, x_freq_hz],
        "sw": [sw_y_hz, sw_x_hz],
        "spec": [1, 1],
        "wholeEcho": [0, 0],
        "ref": [ref_y, ref_x],
        "xaxArray": [xax_y, xax_x],
        "history": ["Exported by Soprano NMRData2D.export_contour_data"],
        "metaData": {
            "x_larmor_MHz": x_larmor_freq_mhz,
            "y_larmor_MHz": y_freq_mhz,
            "x_broadening_ppm": cd.x_broadening,
            "y_broadening_ppm": cd.y_broadening,
            "broadening_type": cd.broadening_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    if include_peaks:
        peaks = nmr_data.get_peaks()
        struct["peaks"] = [
            {
                "x": p.x,
                "y": p.y,
                "xlabel": p.xlabel,
                "ylabel": p.ylabel,
                "correlation_strength": p.correlation_strength,
            }
            for p in peaks
        ]

    with open(path, "w") as f:
        json.dump(struct, f, indent=2)
    _log_info(
        nmr_data,
        f"ssNake JSON written to '{path}' (x={x_larmor_freq_mhz} MHz, y={y_freq_mhz} MHz)."
    )


def _export_bruker(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    x_larmor_freq_mhz: Optional[float],
    y_larmor_freq_mhz: Optional[float],
    include_peaks: bool = True,
) -> None:
    """Write a Bruker TopSpin 2D processed data directory (pdata/1/2rr)."""
    if x_larmor_freq_mhz is None:
        x_el = getattr(nmr_data, "xelement", "x")
        raise ValueError(
            f"Cannot determine Larmor frequency for '{x_el}'. "
            f"Provide x_larmor_freq_mhz, b0_field_tesla, or spectrometer_freq_mhz in ExportConfig."
        )

    try:
        import nmrglue as ng
    except ImportError:
        raise ImportError(
            "nmrglue is required for Bruker export. "
            "Install it with: pip install nmrglue  or  pip install soprano[nmr-io]"
        )

    if not getattr(nmr_data, "is_shift", False):
        raise ValueError(
            "Bruker export requires chemical shift data (is_shift=True). "
            "Provide references when constructing NMRData2D to convert shieldings to shifts."
        )

    # Homonuclear fallback: reuse the direct-dimension Larmor for the indirect
    # dimension when a separate value is not supplied.  Note that multiple-
    # quantum (e.g. 2Q) scaling of the indirect axis is NOT applied here.
    if y_larmor_freq_mhz is None:
        y_larmor_freq_mhz = x_larmor_freq_mhz

    ni, np_ = cd.Z.shape

    # Bruker stores data with descending ppm (downfield first); flip both axes.
    Z_bruker = np.ascontiguousarray(cd.Z[::-1, ::-1], dtype=np.float64)

    def _make_procs(sf_mhz: float, sw_ppm: float, offset_ppm: float, si: int) -> dict:
        # nmrglue/TopSpin store the processed spectral width (SW_p) in Hz, while
        # OFFSET (downfield edge) is in ppm; convert the ppm range with SF (MHz).
        return {
            "_comments": [],
            "_coreheader": ["##NMRGLUE automatically created parameter file"],
            "SF": float(sf_mhz),
            "SW_p": float(sw_ppm * sf_mhz),
            "OFFSET": float(offset_ppm),
            "SI": si,
            "NC_proc": -6,      # intensity scaling exponent (data * 2**NC_proc)
            "BYTORDP": 1,
            "XDIM": si,
            "STSI": 0,
            "STSR": 0,
            "FT_mod": 6,        # marks the dimension as Fourier-transformed
            "PHC0": 0.0,
            "PHC1": 0.0,
        }

    dic = {
        "procs": _make_procs(
            x_larmor_freq_mhz,
            cd.xlims[1] - cd.xlims[0],
            cd.xlims[1],
            np_,
        ),
        "proc2s": _make_procs(
            y_larmor_freq_mhz,
            cd.ylims[1] - cd.ylims[0],
            cd.ylims[1],
            ni,
        ),
    }

    Path(path).mkdir(parents=True, exist_ok=True)

    ng.fileio.bruker.write_pdata(
        path, dic, Z_bruker,
        write_procs=True,
        pdata_folder=True,
        overwrite=True,
        submatrix_shape=(ni, np_),
    )

    if include_peaks:
        peaks_path = str(Path(path) / "peaks.csv")
        _write_peaks_csv(nmr_data.get_peaks(), peaks_path)
        _log_info(nmr_data, f"Peak list written to '{peaks_path}'.")

    _log_info(
        nmr_data,
        f"Bruker TopSpin data written to '{path}' "
        f"(x={x_larmor_freq_mhz} MHz, y={y_larmor_freq_mhz} MHz)."
    )
