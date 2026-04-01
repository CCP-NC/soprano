"""Contour export helpers for 2D NMR data.

This module owns all contour-grid serialization logic (SIMPSON/NPZ/CSV/JSON)
and is the separation-of-concerns boundary between data extraction and file I/O.

Public entrypoint:
    export_contour_data

Compatibility:
    NMRData2D.export_contour_data delegates to this function.
"""

import csv
import json
from typing import Any, Optional, Protocol, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from soprano.calculate.nmr.utils import ContourData, Peak2D


class _NMRData2DExportProtocol(Protocol):
    def get_contour_data(
        self,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        broadening_type: str = "lorentzian",
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
    ) -> "ContourData": ...

    def get_peaks(self) -> list["Peak2D"]: ...


def _log_info(nmr_data: Any, message: str) -> None:
    logger = getattr(nmr_data, "logger", None)
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)


def _log_warning(nmr_data: Any, message: str) -> None:
    logger = getattr(nmr_data, "logger", None)
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(message)


def export_contour_data(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    fmt: str = "simpson",
    x_broadening: Optional[float] = None,
    y_broadening: Optional[float] = None,
    broadening_type: str = "lorentzian",
    grid_size: int = 500,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    x_larmor_freq_mhz: Optional[float] = None,
    y_larmor_freq_mhz: Optional[float] = None,
) -> None:
    """Export contour data for an NMRData2D instance."""
    cd = nmr_data.get_contour_data(
        x_broadening=x_broadening,
        y_broadening=y_broadening,
        broadening_type=broadening_type,
        grid_size=grid_size,
        xlims=xlims,
        ylims=ylims,
    )

    fmt = fmt.lower().strip()

    if fmt == "simpson":
        _export_simpson(
            nmr_data,
            path,
            cd,
            x_larmor_freq_mhz=x_larmor_freq_mhz,
            y_larmor_freq_mhz=y_larmor_freq_mhz,
        )
    elif fmt == "npz":
        _export_npz(nmr_data, path, cd)
    elif fmt == "csv":
        _export_csv_grid(path, cd)
    elif fmt in ("json", "ssnake"):
        _export_json_ssnake(
            nmr_data,
            path,
            cd,
            x_larmor_freq_mhz=x_larmor_freq_mhz,
            y_larmor_freq_mhz=y_larmor_freq_mhz,
        )
    else:
        raise ValueError(
            f"Unknown export format '{fmt}'. "
            "Choose from 'simpson', 'npz', 'csv', 'json'."
        )
    _log_info(nmr_data, f"Exported contour data to '{path}' (format={fmt}).")


def _export_simpson(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    x_larmor_freq_mhz: Optional[float] = None,
    y_larmor_freq_mhz: Optional[float] = None,
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
            "spectrometer frequencies. Pass x_larmor_freq_mhz (and "
            "y_larmor_freq_mhz for heteronuclear spectra) to fix this."
        )

    with open(path, "w") as f:
        f.write("SIMP\n")
        f.write(f"NP={np_}\n")
        f.write(f"NI={ni}\n")
        f.write(f"SW={sw:.8g}\n")
        f.write(f"SW1={sw1:.8g}\n")
        f.write("TYPE=SPE\n")
        f.write("# Exported by Soprano NMRData2D.export_contour_data\n")
        f.write(f"# SW_unit={sw_unit}\n")
        if x_larmor_freq_mhz is not None:
            f.write(f"# SPECFREQ_x={x_larmor_freq_mhz:.6g} MHz  (direct dim)\n")
            f.write(f"# SPECFREQ_y={y_freq:.6g} MHz  (indirect dim)\n")
            f.write("# ssNake: Axes -> Edit axes, set carriers to these values\n")
        f.write(f"# x_broadening={cd.x_broadening:.6g} ppm\n")
        f.write(f"# y_broadening={cd.y_broadening:.6g} ppm\n")
        f.write(f"# broadening_type={cd.broadening_type}\n")
        f.write(f"# xlims_ppm={cd.xlims[0]:.6g} {cd.xlims[1]:.6g}\n")
        f.write(f"# ylims_ppm={cd.ylims[0]:.6g} {cd.ylims[1]:.6g}\n")
        f.write("DATA\n")
        for i in range(ni):
            for j in range(np_):
                f.write(f"{Z[i, j]:.8g} 0.0\n")
        f.write("END")

    peaks_path = path + ".peaks.csv"
    peaks = nmr_data.get_peaks()
    with open(peaks_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_ppm", "y_ppm", "xlabel", "ylabel", "correlation_strength"])
        for p in peaks:
            writer.writerow([p.x, p.y, p.xlabel, p.ylabel, p.correlation_strength])
    _log_info(nmr_data, f"Peak list written to '{peaks_path}'.")


def _export_npz(
    nmr_data: _NMRData2DExportProtocol, path: str, cd: "ContourData"
) -> None:
    """Write a NumPy compressed archive with the grid and metadata."""
    peaks = nmr_data.get_peaks()
    peak_x = np.array([p.x for p in peaks])
    peak_y = np.array([p.y for p in peaks])
    peak_strength = np.array([p.correlation_strength for p in peaks])
    peak_xlabels = np.array([p.xlabel for p in peaks])
    peak_ylabels = np.array([p.ylabel for p in peaks])

    np.savez_compressed(
        path,
        X=cd.X,
        Y=cd.Y,
        Z=cd.Z,
        peak_x=peak_x,
        peak_y=peak_y,
        peak_strength=peak_strength,
        peak_xlabels=peak_xlabels,
        peak_ylabels=peak_ylabels,
        x_broadening=cd.x_broadening,
        y_broadening=cd.y_broadening,
        broadening_type=np.bytes_(cd.broadening_type),
        xlims=np.array(cd.xlims),
        ylims=np.array(cd.ylims),
    )


def _export_csv_grid(path: str, cd: "ContourData") -> None:
    """Write a flat CSV with columns x, y, intensity."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_ppm", "y_ppm", "intensity"])
        ni, np_ = cd.Z.shape
        for i in range(ni):
            for j in range(np_):
                writer.writerow([cd.X[i, j], cd.Y[i, j], cd.Z[i, j]])


def _export_json_ssnake(
    nmr_data: _NMRData2DExportProtocol,
    path: str,
    cd: "ContourData",
    x_larmor_freq_mhz: Optional[float],
    y_larmor_freq_mhz: Optional[float],
) -> None:
    """Write an ssNake-native JSON file with Larmor frequencies embedded."""
    if x_larmor_freq_mhz is None:
        raise ValueError(
            "x_larmor_freq_mhz is required for 'json' export so that "
            "ssNake can display the ppm axis directly."
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

    struct = {
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
        },
    }
    with open(path, "w") as f:
        json.dump(struct, f)
    _log_info(
        nmr_data,
        f"ssNake JSON written to '{path}' (x={x_larmor_freq_mhz} MHz, y={y_freq_mhz} MHz)."
    )
