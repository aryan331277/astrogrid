"""
Pydantic data schemas, telemetry records, and coordinate models for aperture photometry.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict, Any
from enum import Enum
import numpy as np


class ApertureType(str, Enum):
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    RECTANGULAR = "rectangular"
    POLYGON = "polygon"


class CoordinateSystem(str, Enum):
    RA_DEC = "ra_dec"
    GALACTIC = "galactic"
    PIXEL = "pixel"


class ApertureGeometry(BaseModel):
    """Defines the shape and parameters of the photometric aperture."""
    aperture_type: ApertureType
    center_ra: Optional[float] = Field(None, description="Right ascension in degrees")
    center_dec: Optional[float] = Field(None, description="Declination in degrees")
    center_x: Optional[float] = Field(None, description="Pixel x coordinate")
    center_y: Optional[float] = Field(None, description="Pixel y coordinate")
    radius: Optional[float] = Field(None, description="Radius in pixels or degrees")
    semi_major: Optional[float] = Field(None, description="Semi-major axis for elliptical")
    semi_minor: Optional[float] = Field(None, description="Semi-minor axis for elliptical")
    position_angle: Optional[float] = Field(None, description="Position angle in degrees")
    width: Optional[float] = Field(None, description="Width for rectangular aperture")
    height: Optional[float] = Field(None, description="Height for rectangular aperture")
    vertices: Optional[List[List[float]]] = Field(None, description="Polygon vertices")

    @field_validator("aperture_type")
    @classmethod
    def validate_aperture_type(cls, v):
        if v not in ApertureType:
            raise ValueError(f"Invalid aperture type: {v}")
        return v


class BackgroundEstimation(BaseModel):
    """Parameters for background estimation and subtraction."""
    method: str = Field("annulus", description="Background estimation method")
    inner_radius: Optional[float] = Field(None, description="Inner radius of annulus")
    outer_radius: Optional[float] = Field(None, description="Outer radius of annulus")
    sigma_clip: Optional[float] = Field(3.0, description="Sigma clipping threshold")
    n_iterations: Optional[int] = Field(5, description="Number of sigma clipping iterations")
    background_value: Optional[float] = Field(None, description="Estimated background value")
    background_std: Optional[float] = Field(None, description="Standard deviation of background")


class ApertureCorrection(BaseModel):
    """Aperture correction parameters to account for PSF truncation."""
    correction_factor: float = Field(1.0, description="Multiplicative correction factor")
    correction_method: str = Field("psf_model", description="Method used for correction")
    psf_fwhm: Optional[float] = Field(None, description="PSF FWHM in pixels")
    aperture_efficiency: Optional[float] = Field(None, description="Fraction of flux captured")


class AperturePhotometryResult(BaseModel):
    """Result of aperture photometry measurement."""
    source_id: str = Field(..., description="Unique identifier for the source")
    aperture: ApertureGeometry
    background: Optional[BackgroundEstimation] = None
    correction: Optional[ApertureCorrection] = None
    raw_flux: float = Field(..., description="Raw summed flux within aperture")
    background_flux: float = Field(0.0, description="Estimated background flux")
    net_flux: float = Field(..., description="Net flux after background subtraction")
    flux_error: float = Field(..., description="Uncertainty in net flux")
    snr: float = Field(..., description="Signal-to-noise ratio")
    coordinate_system: CoordinateSystem = Field(CoordinateSystem.PIXEL)
    observation_time: Optional[str] = Field(None, description="ISO 8601 timestamp")
    instrument: Optional[str] = Field(None, description="Instrument name")
    filter_name: Optional[str] = Field(None, description="Filter band name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("net_flux")
    @classmethod
    def validate_net_flux(cls, v, values):
        if "raw_flux" in values and "background_flux" in values:
            expected = values["raw_flux"] - values["background_flux"]
            if abs(v - expected) > 1e-6:
                raise ValueError(f"Net flux {v} does not match raw_flux - background_flux = {expected}")
        return v

    @field_validator("snr")
    @classmethod
    def validate_snr(cls, v, values):
        if "net_flux" in values and "flux_error" in values and values["flux_error"] > 0:
            expected = values["net_flux"] / values["flux_error"]
            if abs(v - expected) > 1e-6:
                raise ValueError(f"SNR {v} does not match net_flux / flux_error = {expected}")
        return v


class AperturePhotometryRequest(BaseModel):
    """Request to perform aperture photometry on an image."""
    image_id: str = Field(..., description="Identifier for the input image")
    sources: List[ApertureGeometry] = Field(..., description="List of source apertures")
    background_method: str = Field("annulus", description="Background estimation method")
    apply_correction: bool = Field(True, description="Whether to apply aperture correction")
    coordinate_system: CoordinateSystem = Field(CoordinateSystem.PIXEL)
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")


class AperturePhotometryResponse(BaseModel):
    """Response containing photometry results."""
    request_id: str = Field(..., description="Identifier for the request")
    results: List[AperturePhotometryResult] = Field(..., description="Photometry results")
    status: str = Field("success", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class ApertureTelemetryRecord(BaseModel):
    """Telemetry record for monitoring aperture photometry operations."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    operation_id: str = Field(..., description="Unique operation identifier")
    image_id: str = Field(..., description="Image identifier")
    n_sources: int = Field(..., description="Number of sources processed")
    n_success: int = Field(..., description="Number of successful measurements")
    n_failed: int = Field(..., description="Number of failed measurements")
    avg_snr: Optional[float] = Field(None, description="Average SNR across sources")
    min_snr: Optional[float] = Field(None, description="Minimum SNR")
    max_snr: Optional[float] = Field(None, description="Maximum SNR")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage in MB")
    status: str = Field("completed", description="Operation status")
    error_details: Optional[str] = Field(None, description="Error details if any")


def compute_aperture_mask(
    geometry: ApertureGeometry,
    image_shape: tuple,
    center_offset: tuple = (0.0, 0.0)
) -> np.ndarray:
    """
    Compute a boolean mask for the given aperture geometry.
    
    Args:
        geometry: Aperture geometry definition
        image_shape: Shape of the image (height, width)
        center_offset: Offset to apply to center coordinates
        
    Returns:
        Boolean numpy array mask
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=bool)
    
    if geometry.aperture_type == ApertureType.CIRCULAR:
        if geometry.center_x is None or geometry.center_y is None or geometry.radius is None:
            raise ValueError("Circular aperture requires center_x, center_y, and radius")
        cx = geometry.center_x + center_offset[0]
        cy = geometry.center_y + center_offset[1]
        r = geometry.radius
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        
    elif geometry.aperture_type == ApertureType.ELLIPTICAL:
        if (geometry.center_x is None or geometry.center_y is None or 
            geometry.semi_major is None or geometry.semi_minor is None):
            raise ValueError("Elliptical aperture requires center_x, center_y, semi_major, semi_minor")
        cx = geometry.center_x + center_offset[0]
        cy = geometry.center_y + center_offset[1]
        a = geometry.semi_major
        b = geometry.semi_minor
        pa = np.radians(geometry.position_angle or 0.0)
        y, x = np.ogrid[:height, :width]
        dx = x - cx
        dy = y - cy
        # Rotate coordinates
        dx_rot = dx * np.cos(pa) + dy * np.sin(pa)
        dy_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        mask = (dx_rot / a) ** 2 + (dy_rot / b) ** 2 <= 1.0
        
    elif geometry.aperture_type == ApertureType.RECTANGULAR:
        if (geometry.center_x is None or geometry.center_y is None or
            geometry.width is None or geometry.height is None):
            raise ValueError("Rectangular