"""FITS Header WCS Parser for AstroGrid.

Parses World Coordinate System (WCS) information from FITS header dictionaries
and provides pixel-to-world and world-to-pixel coordinate transformations.
"""

import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class WCSAxisType(str, Enum):
    """Enumeration of standard WCS axis types."""
    RA = "RA"
    DEC = "DEC"
    FREQ = "FREQ"
    TIME = "TIME"
    VELO = "VELO"
    STOKES = "STOKES"
    ICRS = "ICRS"
    GAL = "GAL"
    EQJ = "EQJ"
    EKB = "EKB"
    OTHER = "OTHER"


class WCSAxis(BaseModel):
    """Represents a single WCS axis."""
    index: int = Field(..., description="1-based axis index")
    ctype: str = Field(..., description="CTYPEn keyword value")
    crval: float = Field(..., description="CRVALn keyword value")
    crpix: float = Field(..., description="CRPIXn keyword value")
    cdelt: float = Field(..., description="CDELTn keyword value")
    axis_type: WCSAxisType = Field(default=WCSAxisType.OTHER)


class WCSInfo(BaseModel):
    """Complete WCS information parsed from a FITS header."""
    naxis: int = Field(..., description="Number of axes")
    axes: List[WCSAxis] = Field(default_factory=list)
    cd_matrix: Optional[np.ndarray] = Field(default=None, description="CD matrix if present")
    pc_matrix: Optional[np.ndarray] = Field(default=None, description="PC matrix if present")
    has_cd: bool = Field(default=False)
    has_pc: bool = Field(default=False)


def _parse_axis_type(ctype: str) -> WCSAxisType:
    """Parse the axis type from a CTYPEn string."""
    ctype_upper = ctype.upper().strip()
    if ctype_upper.startswith("RA") or ctype_upper.startswith("GLON"):
        return WCSAxisType.RA
    if ctype_upper.startswith("DEC") or ctype_upper.startswith("GLAT"):
        return WCSAxisType.DEC
    if ctype_upper.startswith("FREQ"):
        return WCSAxisType.FREQ
    if ctype_upper.startswith("TIME"):
        return WCSAxisType.TIME
    if ctype_upper.startswith("VELO") or ctype_upper.startswith("VRAD"):
        return WCSAxisType.VELO
    if ctype_upper.startswith("STOK"):
        return WCSAxisType.STOKES
    if ctype_upper.startswith("ICRS"):
        return WCSAxisType.ICRS
    if ctype_upper.startswith("GAL"):
        return WCSAxisType.GAL
    if ctype_upper.startswith("EQJ"):
        return WCSAxisType.EQJ
    if ctype_upper.startswith("EKB"):
        return WCSAxisType.EKB
    return WCSAxisType.OTHER


def parse_wcs_header(header: Dict[str, Any]) -> WCSInfo:
    """Parse WCS information from a FITS header dictionary.

    Args:
        header: Dictionary representing FITS header keywords.

    Returns:
        WCSInfo object containing parsed WCS parameters.
    """
    naxis = int(header.get("NAXIS", 0))
    axes: List[WCSAxis] = []

    for i in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{i}", "OTHER"))
        crval = float(header.get(f"CRVAL{i}", 0.0))
        crpix = float(header.get(f"CRPIX{i}", 1.0))
        cdelt = float(header.get(f"CDELT{i}", 1.0))
        axis_type = _parse_axis_type(ctype)

        axes.append(WCSAxis(
            index=i,
            ctype=ctype,
            crval=crval,
            crpix=crpix,
            cdelt=cdelt,
            axis_type=axis_type,
        ))

    # Check for CD matrix
    has_cd = any(f"CD{i}{j}" in header for i in range(1, naxis + 1) for j in range(1, naxis + 1))
    cd_matrix = None
    if has_cd:
        cd_matrix = np.zeros((naxis, naxis))
        for i in range(1, naxis + 1):
            for j in range(1, naxis + 1):
                cd_matrix[i - 1, j - 1] = float(header.get(f"CD{i}{j}", 0.0))

    # Check for PC matrix
    has_pc = any(f"PC{i}{j}" in header for i in range(1, naxis + 1) for j in range(1, naxis + 1))
    pc_matrix = None
    if has_pc:
        pc_matrix = np.zeros((naxis, naxis))
        for i in range(1, naxis + 1):
            for j in range(1, naxis + 1):
                pc_matrix[i - 1, j - 1] = float(header.get(f"PC{i}{j}", 0.0))

    return WCSInfo(
        naxis=naxis,
        axes=axes,
        cd_matrix=cd_matrix,
        pc_matrix=pc_matrix,
        has_cd=has_cd,
        has_pc=has_pc,
    )


def pixel_to_world(wcs: WCSInfo, pixel_coords: np.ndarray) -> np.ndarray:
    """Transform pixel coordinates to world coordinates.

    Args:
        wcs: Parsed WCS information.
        pixel_coords: Array of shape (..., naxis) with pixel coordinates (1-based).

    Returns:
        Array of shape (..., naxis) with world coordinates.
    """
    pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
    naxis = wcs.naxis
    result = np.zeros_like(pixel_coords)

    # Compute linear transformation: (pixel - crpix)
    crpix = np.array([ax.crpix for ax in wcs.axes])
    shifted = pixel_coords - crpix

    # Apply linear transformation matrix
    if wcs.has_cd and wcs.cd_matrix is not None:
        linear = shifted @ wcs.cd_matrix.T
    elif wcs.has_pc and wcs.pc_matrix is not None:
        cdelt = np.array([ax.cdelt for ax in wcs.axes])
        linear = (shifted * cdelt) @ wcs.pc_matrix.T
    else:
        cdelt = np.array([ax.cdelt for ax in wcs.axes])
        linear = shifted * cdelt

    # Add reference values
    crval = np.array([ax.crval for ax in wcs.axes])
    result = linear + crval

    return result


def world_to_pixel(wcs: WCSInfo, world_coords: np.ndarray) -> np.ndarray:
    """Transform world coordinates to pixel coordinates.

    Args:
        wcs: Parsed WCS information.
        world_coords: Array of shape (..., naxis) with world coordinates.

    Returns:
        Array of shape (..., naxis) with pixel coordinates (1-based).
    """
    world_coords = np.asarray(world_coords, dtype=np.float64)
    naxis = wcs.naxis

    # Subtract reference values
    crval = np.array([ax.crval for ax in wcs.axes])
    shifted = world_coords - crval

    # Invert linear transformation
    if wcs.has_cd and wcs.cd_matrix is not None:
        inv_matrix = np.linalg.inv(wcs.cd_matrix)
        linear = shifted @ inv_matrix.T
    elif wcs.has_pc and wcs.pc_matrix is not None:
        cdelt = np.array([ax.cdelt for ax in wcs.axes])
        inv_pc = np.linalg.inv(wcs.pc_matrix)
        linear = (shifted @ inv_pc.T) / cdelt
    else:
        cdelt = np.array([ax.cdelt for ax in wcs.axes])
        linear = shifted / cdelt

    # Add reference pixel
    crpix = np.array([ax.crpix for ax in wcs.axes])
    result = linear + crpix

    return result