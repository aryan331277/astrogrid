"""Pydantic data schemas, telemetry records, and coordinate models for fits_header_wcs_parser."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import numpy as np


class WCSAxisType(str, Enum):
    """Enumeration of standard WCS axis types."""
    RA = "RA"
    DEC = "DEC"
    FREQ = "FREQ"
    STOKES = "STOKES"
    TIME = "TIME"
    VELOCITY = "VELOCITY"
    POSITION = "POSITION"
    ENERGY = "ENERGY"
    WAVELENGTH = "WAVELENGTH"
    CUSTOM = "CUSTOM"


class CoordinateSystem(str, Enum):
    """Enumeration of supported coordinate systems."""
    ICRS = "ICRS"
    GALACTIC = "GALACTIC"
    ECLIPTIC = "ECLIPTIC"
    SUPERGALACTIC = "SUPERGALACTIC"
    LOCAL = "LOCAL"
    CUSTOM = "CUSTOM"


class PixelCoordinate(BaseModel):
    """Represents a pixel coordinate in the image plane."""
    x: float = Field(..., description="Pixel column index (1-based)")
    y: float = Field(..., description="Pixel row index (1-based)")
    z: Optional[float] = Field(None, description="Optional third pixel dimension")

    @field_validator("x", "y")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("Pixel coordinates must be >= 1.0 (1-based indexing)")
        return v


class WorldCoordinate(BaseModel):
    """Represents a world coordinate in the physical/celestial frame."""
    values: List[float] = Field(..., description="World coordinate values")
    system: CoordinateSystem = Field(CoordinateSystem.ICRS, description="Coordinate system")
    units: Optional[List[str]] = Field(None, description="Units for each coordinate value")

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("World coordinate values cannot be empty")
        return v


class WCSAxis(BaseModel):
    """Describes a single WCS axis with its transformation parameters."""
    index: int = Field(..., ge=1, description="Axis index (1-based)")
    type: WCSAxisType = Field(WCSAxisType.CUSTOM, description="Type of WCS axis")
    reference_pixel: float = Field(..., description="Reference pixel value (CRPIX)")
    reference_value: float = Field(..., description="Reference world value (CRVAL)")
    pixel_scale: float = Field(..., description="Pixel scale factor (CDELT)")
    rotation: Optional[float] = Field(None, description="Rotation angle in degrees")
    units: Optional[str] = Field(None, description="Units of the world coordinate")
    name: Optional[str] = Field(None, description="Axis name from header")


class WCSMatrix(BaseModel):
    """WCS transformation matrix (PC or CD matrix)."""
    matrix: np.ndarray = Field(..., description="2D or 3D transformation matrix")
    is_cd: bool = Field(False, description="True if CD matrix, False if PC matrix")

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim not in (2, 3):
            raise ValueError("WCS matrix must be 2D or 3D")
        return v


class FITSHeaderWCS(BaseModel):
    """Complete WCS information parsed from a FITS header."""
    naxes: int = Field(..., ge=1, description="Number of axes in the image")
    axes: List[WCSAxis] = Field(..., description="List of WCS axes")
    matrix: Optional[WCSMatrix] = Field(None, description="WCS transformation matrix")
    coordinate_system: CoordinateSystem = Field(CoordinateSystem.ICRS, description="Primary coordinate system")
    crpix: Optional[List[float]] = Field(None, description="Reference pixel values")
    crval: Optional[List[float]] = Field(None, description="Reference world values")
    cdelt: Optional[List[float]] = Field(None, description="Pixel scale values")
    cunit: Optional[List[str]] = Field(None, description="Units for each axis")
    ctype: Optional[List[str]] = Field(None, description="Type strings for each axis")
    raw_header: Dict[str, Any] = Field(default_factory=dict, description="Raw header keywords")

    @field_validator("axes")
    @classmethod
    def validate_axes_count(cls, v: List[WCSAxis], info: Any) -> List[WCSAxis]:
        naxes = info.data.get("naxes")
        if naxes is not None and len(v) != naxes:
            raise ValueError(f"Number of axes ({len(v)}) must match naxes ({naxes})")
        return v


class WCSParseResult(BaseModel):
    """Result of parsing a FITS header for WCS information."""
    success: bool = Field(..., description="Whether parsing was successful")
    wcs: Optional[FITSHeaderWCS] = Field(None, description="Parsed WCS information")
    errors: List[str] = Field(default_factory=list, description="List of parse errors")
    warnings: List[str] = Field(default_factory=list, description="List of parse warnings")
    parse_time_ms: float = Field(0.0, ge=0.0, description="Time taken to parse in milliseconds")
    source_file: Optional[str] = Field(None, description="Source FITS file path")
    parsed_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of parsing")


class CoordinateTransformRequest(BaseModel):
    """Request to transform pixel coordinates to world coordinates."""
    pixel_coords: List[PixelCoordinate] = Field(..., description="Input pixel coordinates")
    wcs: FITSHeaderWCS = Field(..., description="WCS to use for transformation")
    target_system: Optional[CoordinateSystem] = Field(None, description="Target coordinate system")


class CoordinateTransformResult(BaseModel):
    """Result of a coordinate transformation."""
    world_coords: List[WorldCoordinate] = Field(..., description="Transformed world coordinates")
    source_system: CoordinateSystem = Field(..., description="Source coordinate system")
    target_system: CoordinateSystem = Field(..., description="Target coordinate system")
    transform_time_ms: float = Field(0.0, ge=0.0, description="Transformation time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Transformation errors")


class WCSTelemetryRecord(BaseModel):
    """Telemetry record for WCS parsing operations."""
    record_id: str = Field(..., description="Unique record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Record timestamp")
    operation: str = Field(..., description="Operation type (parse, transform, validate)")
    source_file: Optional[str] = Field(None, description="Source file identifier")
    naxes: Optional[int] = Field(None, ge=1, description="Number of axes processed")
    success: bool = Field(..., description="Whether the operation succeeded")
    duration_ms: float = Field(0.0, ge=0.0, description="Operation duration in milliseconds")
    error_count: int = Field(0, ge=0, description="Number of errors encountered")
    warning_count: int = Field(0, ge=0, description="Number of warnings encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WCSValidationResult(BaseModel):
    """Result of validating a WCS configuration."""
    is_valid: bool = Field(..., description="Whether the WCS is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")
    wcs: Optional[FITSHeaderWCS] = Field(None, description="The validated WCS")


class BatchWCSParseRequest(BaseModel):
    """Request to parse multiple FITS headers for WCS information."""
    files: List[str] = Field(..., description="List of FITS file paths")
    max_workers: int = Field(4, ge=1, le=64, description="Maximum parallel workers")
    timeout_seconds: float = Field(30.0, gt=0.0, description="Timeout per file in seconds")


class BatchWCSParseResult(BaseModel):
    """Result of batch WCS parsing."""
    total_files: int = Field(..., ge=0, description="Total number of files processed")
    successful: int = Field(0, ge=0, description="Number of successful parses")
    failed: int = Field(0, ge=0, description="Number of failed parses")
    results: List[WCSParseResult] = Field(default_factory=list, description="Individual parse results")
    total_time_ms: float = Field(0.0, ge=0.0, description="Total batch processing time in milliseconds")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Batch completion timestamp")