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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, TYPE_CHECKING

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

    def _resolve(freq: Optional[float], element: Optional[str], isotope: Optional[int]) -> Optional[float]:
        if freq is not None:
            return freq
        if element is None or b0_tesla is None:
            return None
        try:
            return compute_larmor_frequency(element, b0_tesla=b0_tesla, isotope=isotope)
        except ValueError:
            return None

    x_resolved = _resolve(x_larmor, xelement, getattr(nmr_data, "xisotope", None))
    y_resolved = _resolve(y_larmor, yelement, getattr(nmr_data, "yisotope", None))

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
    overwrite: bool = False  # replace an existing Bruker dataset at the target path

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


def _sw_ppm_from_span(span_ppm: float, n: int) -> float:
    # The contour grid is an endpoint-inclusive linspace with step span/(n-1),
    # whereas ssNake's FFT axis has spacing SW/n.  Scale the width by n/(n-1)
    # so the per-point step matches.  Inferred from ssNake's reader (SIMPSON
    # manual unavailable).
    return span_ppm * n / (n - 1) if n > 1 else span_ppm


def _carrier_ppm(lo_ppm: float, span_ppm: float, n: int) -> float:
    # ssNake places point k at (k - n//2) * SW/n from the carrier (fftshift
    # order), so the carrier is put at grid point n//2 and that point reads back
    # as its own ppm value.  Assumes ascending frequency order in the data block.
    step = span_ppm / (n - 1) if n > 1 else 0.0
    return lo_ppm + (n // 2) * step


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
    effective = config if config is not None else ExportConfig()

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
            overwrite=effective.overwrite,
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
    sw_ppm = _sw_ppm_from_span(cd.xlims[1] - cd.xlims[0], np_)
    sw1_ppm = _sw_ppm_from_span(cd.ylims[1] - cd.ylims[0], ni)

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
        x_carrier_ppm = _carrier_ppm(cd.xlims[0], cd.xlims[1] - cd.xlims[0], np_)
        y_carrier_ppm = _carrier_ppm(cd.ylims[0], cd.ylims[1] - cd.ylims[0], ni)
        meta_lines.extend([
            f"# SPECREF_x={x_larmor_freq_mhz:.15g} MHz  (0 ppm reference, direct dim)",
            f"# SPECFREQ_x={x_larmor_freq_mhz * (1 + x_carrier_ppm * 1e-6):.15g} MHz  (carrier, direct dim)",
            f"# SPECREF_y={y_freq:.15g} MHz  (0 ppm reference, indirect dim)",
            f"# SPECFREQ_y={y_freq * (1 + y_carrier_ppm * 1e-6):.15g} MHz  (carrier, indirect dim)",
            # Conventions inferred from ssNake's reader (SIMPSON manual unavailable).
            "# ssNake: Axes -> Edit axes, set Reference = SPECREF and Frequency (carrier) = SPECFREQ",
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
        f.write(f"SW={sw:.12g}\n")
        f.write(f"SW1={sw1:.12g}\n")
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
        _write_peaks_csv(nmr_data.get_peaks(), peaks_path)
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
        _write_peaks_csv(nmr_data.get_peaks(), peaks_path)
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

    ni, np_ = cd.Z.shape
    sw_x_hz = _sw_ppm_from_span(cd.xlims[1] - cd.xlims[0], np_) * x_larmor_freq_mhz
    sw_y_hz = _sw_ppm_from_span(cd.ylims[1] - cd.ylims[0], ni) * y_freq_mhz

    # ref is the 0 ppm frequency; freq is the carrier (see _carrier_ppm).  Both in Hz.
    ref_x = x_freq_hz
    ref_y = y_freq_hz
    carrier_x_hz = x_freq_hz * (1 + _carrier_ppm(cd.xlims[0], cd.xlims[1] - cd.xlims[0], np_) * 1e-6)
    carrier_y_hz = y_freq_hz * (1 + _carrier_ppm(cd.ylims[0], cd.ylims[1] - cd.ylims[0], ni) * 1e-6)

    data_3d = cd.Z.reshape(1, ni, np_)
    flat_real = data_3d.tolist()
    flat_imag = np.zeros((1, ni, np_)).tolist()

    xax_x = (np.linspace(cd.xlims[0], cd.xlims[1], np_) * x_larmor_freq_mhz).tolist()
    xax_y = (np.linspace(cd.ylims[0], cd.ylims[1], ni) * y_freq_mhz).tolist()

    struct: dict[str, Any] = {
        "dataReal": flat_real,
        "dataImag": flat_imag,
        "hyper": [0],
        "freq": [carrier_y_hz, carrier_x_hz],
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


def _bruker_sw_ppm(sw_ppm: float, si: int) -> float:
    # Bruker frequency axes are endpoint-EXCLUSIVE: point i sits at
    #   ppm(i) = OFFSET - i * (SW_p / SF) / SI
    # whereas the contour grid is an endpoint-inclusive linspace with
    # step sw_ppm / (SI - 1).  Scale SW_p by SI/(SI-1) so the per-point
    # step matches and the last point lands exactly on the upfield limit.
    return sw_ppm * si / (si - 1) if si > 1 else sw_ppm


def _bruker_nc_proc(Z: np.ndarray) -> int:
    # nmrglue's array_to_int multiplies the data by 2 until max|Z| reaches
    # [2**28, 2**29) (at most 30 times).  Recording the same factor as
    # NC_proc = -k gives real = int * 2**NC_proc.
    if not np.all(np.isfinite(Z)):
        raise ValueError("Cannot export Bruker data: the contour grid contains NaN or inf.")
    max_abs = float(np.max(np.abs(Z)))
    k = 0
    while k < 30 and max_abs * 2.0**k < 2**28:
        k += 1
    scaled_max = max_abs * 2.0**k
    if scaled_max < 1.0:
        raise ValueError(
            f"Cannot export Bruker data: max|intensity| = {max_abs:g} is too small to "
            "store as integers; every point would be written as zero."
        )
    if scaled_max > 2**31 - 1:
        raise ValueError(
            f"Cannot export Bruker data: max|intensity| = {max_abs:g} overflows int32."
        )
    return -k


def _bruker_jcamp_header() -> List[str]:
    # TopSpin refuses parameter files that lack the JCAMP-DX core header
    # ("getpar failed ... Parameter file has unknown format").
    return [
        "##TITLE= Parameter file, Soprano",
        "##JCAMPDX= 5.0",
        "##DATATYPE= Parameter Values",
        "##ORIGIN= Soprano",
        "##OWNER= soprano",
    ]


def _bruker_procs(
    sf_mhz: float, sw_ppm: float, offset_ppm: float, si: int, nucleus: str, nc_proc: int,
) -> dict:
    # nmrglue/TopSpin store the processed spectral width (SW_p) in Hz, while
    # OFFSET (downfield edge) is in ppm; convert the ppm range with SF (MHz).
    return {
        "_comments": [],
        "_coreheader": _bruker_jcamp_header(),
        "SF": float(sf_mhz),
        "SW_p": float(_bruker_sw_ppm(sw_ppm, si) * sf_mhz),
        "OFFSET": float(offset_ppm),
        "SI": si,
        "NC_proc": nc_proc,      # intensity scaling exponent (data * 2**NC_proc)
        "BYTORDP": 1,
        "DTYPP": 0,         # data type: 0 = int32
        "AXNUC": nucleus,     # write_jcamp adds the <...> brackets
        "XDIM": si,
        "STSI": 0,
        "STSR": 0,
        "FT_mod": 6,        # marks the dimension as Fourier-transformed
        "PHC0": 0.0,
        "PHC1": 0.0,
    }


def _bruker_clevels(z_max: float, n_levels: int = 8, lowest: float = 0.1, highest: float = 0.95) -> dict:
    # Without pdata/1/clevels TopSpin falls back to levels that lie outside
    # the data range and draws nothing. Levels are geometric, as TopSpin's own
    # (POSBASE × POSINCR**k), from `lowest` to `highest` of the maximum, in
    # display units (stored value × 2**NC_proc). Positive levels only: Soprano
    # grids are non-negative unless use_signed is set.
    base = lowest * z_max
    incr = (highest / lowest) ** (1.0 / (n_levels - 1))
    levels = [float(base * incr**k) for k in range(n_levels)]
    return {
        "_comments": [],
        "_coreheader": _bruker_jcamp_header(),
        "LEVELS": levels + [0.0] * (256 - n_levels),
        "LEVSIGN": 1,
        "MAXLEV": n_levels,
        "METHOD": 0,
        "NEGBASE": -base,
        "NEGINCR": incr,
        "POSBASE": base,
        "POSINCR": incr,
    }


def _bruker_acqus(
    sf_mhz: float, sw_ppm: float, offset_ppm: float, si: int, nucleus: str, indirect: bool,
) -> dict:
    # Minimal acquisition-parameter set so that TopSpin can open the
    # dataset and nmrglue's guess_udic can reconstruct the axes.
    # The carrier sits at the centre of the spectral window:
    # guess_udic derives car = (SFO1 - SF) * 1e6 Hz.
    sw_ppm_b = _bruker_sw_ppm(sw_ppm, si)
    center_ppm = offset_ppm - sw_ppm_b / 2.0
    o1_hz = center_ppm * sf_mhz          # ppm * MHz = Hz
    sfo1 = float(sf_mhz + o1_hz * 1e-6)
    sw_h = float(sw_ppm_b * sf_mhz)
    acqus = {
        "_comments": [],
        "_coreheader": _bruker_jcamp_header(),
        "SFO1": sfo1,
        "BF1": float(sf_mhz),
        "O1": float(o1_hz),
        "SW_h": sw_h,
        # SW is SW_h / SFO1 (TopSpin convention).  nmrglue's guess_udic
        # reads the indirect axis width as SW * SFO1, so this keeps both
        # dimensions consistent with the grid.
        "SW": sw_h / sfo1,
        "TD": 2 * si,
        "NUC1": nucleus,
        "BYTORDA": 1,
        "DTYPA": 0,
        "PARMODE": 1,       # 2D dataset (PARMODE = ndim - 1)
    }
    if indirect:
        acqus["FnMODE"] = 1     # QF (magnitude): simulated real data
    else:
        acqus["AQ_mod"] = 0     # qf: simulated real data
    return acqus


def _bruker_nucleus_label(nmr_data: Any, element: Optional[str], isotope: Optional[int]) -> str:
    # Isotope label in TopSpin convention, e.g. '1H', '13C'.
    if element is None:
        return "off"
    if isotope is not None:
        return f"{isotope}{element}"
    isotopes = getattr(nmr_data, "isotopes", None)
    if isinstance(isotopes, dict) and element in isotopes:
        return f"{isotopes[element]}{element}"
    return str(element)


def _export_bruker(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    x_larmor_freq_mhz: Optional[float],
    y_larmor_freq_mhz: Optional[float],
    include_peaks: bool = True,
    overwrite: bool = False,
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
    except ImportError as exc:
        raise ImportError(
            "nmrglue is required for Bruker export. "
            "Install it with: pip install nmrglue  or  pip install soprano[nmr-io]"
        ) from exc

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
        y_el = getattr(nmr_data, "yelement", None)
        if y_el is not None and y_el != getattr(nmr_data, "xelement", None):
            # The ppm axis still round-trips (ppm = SW_p / SF), so the overlay
            # remains usable; only the Hz scale of the indirect dimension is
            # wrong, and it is labelled with the y nucleus via AXNUC.
            _log_warning(
                nmr_data,
                f"No Larmor frequency for the indirect dimension ('{y_el}'); "
                f"falling back to the direct-dimension value "
                f"({x_larmor_freq_mhz} MHz). The indirect ppm axis is still "
                f"correct, but its Hz scale will not be. Pass "
                f"y_larmor_freq_mhz, or b0_field_tesla/spectrometer_freq_mhz "
                f"so it can be computed for '{y_el}'.",
            )

    ni, np_ = cd.Z.shape

    # Bruker stores data with descending ppm (downfield first); flip both axes.
    Z_bruker = np.ascontiguousarray(cd.Z[::-1, ::-1], dtype=np.float64)
    nc_proc = _bruker_nc_proc(Z_bruker)
    # The stored int32 values, as nmrglue's array_to_int will produce them
    # (it scales Z_bruker in place, so take the range before writing).
    z_int = (Z_bruker * 2.0 ** (-nc_proc)).astype(np.int32)
    data_range = {"YMAX_p": int(z_int.max()), "YMIN_p": int(z_int.min())}
    # TopSpin shows (and sets contour levels in) stored value × 2**NC_proc.
    z_display_max = data_range["YMAX_p"] * 2.0 ** nc_proc

    x_nuc = _bruker_nucleus_label(
        nmr_data, getattr(nmr_data, "xelement", None), getattr(nmr_data, "xisotope", None)
    )
    y_nuc = _bruker_nucleus_label(
        nmr_data, getattr(nmr_data, "yelement", None), getattr(nmr_data, "yisotope", None)
    )
    x_sw_ppm = cd.xlims[1] - cd.xlims[0]
    y_sw_ppm = cd.ylims[1] - cd.ylims[0]

    dic = {
        "procs": {
            **_bruker_procs(x_larmor_freq_mhz, x_sw_ppm, cd.xlims[1], np_, x_nuc, nc_proc),
            **data_range,
        },
        "proc2s": {
            **_bruker_procs(y_larmor_freq_mhz, y_sw_ppm, cd.ylims[1], ni, y_nuc, nc_proc),
            **data_range,
        },
        "acqus": _bruker_acqus(
            x_larmor_freq_mhz, x_sw_ppm, cd.xlims[1], np_, x_nuc, indirect=False,
        ),
        "acqu2s": _bruker_acqus(
            y_larmor_freq_mhz, y_sw_ppm, cd.ylims[1], ni, y_nuc, indirect=True,
        ),
    }

    root = Path(path)
    existing = [p for p in (root / "acqus", root / "pdata" / "1" / "2rr") if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Bruker dataset already exists: '{existing[0]}' is present. "
            "Pass overwrite=True (ExportConfig(overwrite=True) or --overwrite on the CLI) "
            "to replace it."
        )

    root.mkdir(parents=True, exist_ok=True)

    ng.fileio.bruker.write_pdata(
        path, dic, Z_bruker,
        write_procs=True,
        pdata_folder=True,
        overwrite=overwrite,
        submatrix_shape=(ni, np_),
        big=True,
    )

    # write_pdata only writes the procs files; write acqus/acqu2s (and the
    # acqu/acqu2 copies TopSpin expects) at the experiment root ourselves.
    # TopSpin needs them to open the dataset and nmrglue's guess_udic reads
    # axis information from them.
    for src_name, dst_names in (
        ("acqus", ("acqus", "acqu")),
        ("acqu2s", ("acqu2s", "acqu2")),
    ):
        for dst in dst_names:
            ng.fileio.bruker.write_jcamp(
                dic[src_name], str(root / dst), overwrite=overwrite
            )

    ng.fileio.bruker.write_jcamp(
        _bruker_clevels(z_display_max),
        str(root / "pdata" / "1" / "clevels"),
        overwrite=overwrite,
    )

    # Human-readable title shown in the TopSpin browser
    title_path = root / "pdata" / "1" / "title"
    title_path.write_text(
        "Soprano simulated 2D NMR spectrum\n"
        f"x: {x_nuc} ({x_larmor_freq_mhz:.4f} MHz)  "
        f"y: {y_nuc} ({y_larmor_freq_mhz:.4f} MHz)\n"
        f"exported {datetime.now(timezone.utc).isoformat()}\n"
    )

    if include_peaks:
        peaks_path = str(root / "peaks.csv")
        _write_peaks_csv(nmr_data.get_peaks(), peaks_path)
        _log_info(nmr_data, f"Peak list written to '{peaks_path}'.")

    _log_info(
        nmr_data,
        f"Bruker TopSpin data written to '{path}' "
        f"(x={x_larmor_freq_mhz} MHz, y={y_larmor_freq_mhz} MHz)."
    )
