"""Pydantic data schemas, telemetry records, and coordinate models for psf_fitting_engine."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class PSFModelType(str, Enum):
    GAUSSIAN = "gaussian"
    MOG = "mog"
    KRON = "kron"


class PSFParameters(BaseModel):
    """Parameters for a PSF model."""
    model_type: PSFModelType = Field(..., description="Type of PSF model")
    fwhm: float = Field(..., gt=0, description="Full width at half maximum in pixels")
    ellipticity_g1: float = Field(0.0, ge=-1, le=1, description="Ellipticity component g1")
    ellipticity_g2: float = Field(0.0, ge=-1, le=1, description="Ellipticity component g2")
    center_x: float = Field(0.0, description="Center x coordinate in pixels")
    center_y: float = Field(0.0, description="Center y coordinate in pixels")
    flux: float = Field(0.0, description="Total flux of the source")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional model-specific parameters")

    @field_validator("fwhm")
    @classmethod
    def validate_fwhm(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("FWHM must be positive")
        return v


class PSFFitResult(BaseModel):
    """Result of a PSF fitting operation."""
    success: bool = Field(..., description="Whether the fit converged")
    parameters: PSFParameters = Field(..., description="Fitted PSF parameters")
    chi_squared: float = Field(..., ge=0, description="Chi-squared statistic of the fit")
    dof: int = Field(..., ge=0, description="Degrees of freedom")
    reduced_chi_squared: float = Field(..., ge=0, description="Reduced chi-squared")
    iterations: int = Field(0, ge=0, description="Number of iterations performed")
    covariance_matrix: Optional[List[List[float]]] = Field(None, description="Parameter covariance matrix")
    residuals: Optional[List[float]] = Field(None, description="Residuals from the fit")
    timestamp: float = Field(..., description="Unix timestamp of the fit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PSFModelConfig(BaseModel):
    """Configuration for PSF fitting engine."""
    model_type: PSFModelType = Field(PSFModelType.GAUSSIAN, description="Default PSF model type")
    max_iterations: int = Field(1000, gt=0, description="Maximum number of iterations")
    tolerance: float = Field(1e-6, gt=0, description="Convergence tolerance")
    initial_fwhm: float = Field(2.0, gt=0, description="Initial FWHM estimate in pixels")
    search_radius: float = Field(5.0, gt=0, description="Search radius for source detection in pixels")
    min_flux_threshold: float = Field(0.0, ge=0, description="Minimum flux threshold for source detection")
    background_order: int = Field(0, ge=0, le=4, description="Order of background polynomial")
    use_adaptive_grid: bool = Field(True, description="Whether to use adaptive grid for fitting")
    grid_size: int = Field(11, gt=0, description="Size of fitting grid")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration options")


class PSFSource(BaseModel):
    """A detected source with PSF fit results."""
    source_id: str = Field(..., description="Unique identifier for the source")
    ra: float = Field(..., ge=0, le=360, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90, le=90, description="Declination in degrees")
    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")
    flux: float = Field(..., description="Measured flux")
    fwhm: float = Field(..., gt=0, description="FWHM in pixels")
    g1: float = Field(0.0, description="Ellipticity g1")
    g2: float = Field(0.0, description="Ellipticity g2")
    fit_result: Optional[PSFFitResult] = Field(None, description="Detailed fit result")
    quality_flag: int = Field(0, ge=0, description="Quality flag (0=good, >0=issues)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PSFEngineTelemetry(BaseModel):
    """Telemetry record for PSF fitting engine operations."""
    engine_id: str = Field(..., description="Unique identifier for the engine instance")
    timestamp: float = Field(..., description="Unix timestamp")
    operation: str = Field(..., description="Type of operation performed")
    status: str = Field(..., description="Operation status (success, failure, warning)")
    duration_ms: float = Field(..., ge=0, description="Operation duration in milliseconds")
    sources_processed: int = Field(0, ge=0, description="Number of sources processed")
    fits_converged: int = Field(0, ge=0, description="Number of fits that converged")
    fits_failed: int = Field(0, ge=0, description="Number of fits that failed")
    avg_iteration_count: float = Field(0.0, ge=0, description="Average number of iterations")
    memory_usage_mb: float = Field(0.0, ge=0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(0.0, ge=0, le=100, description="CPU usage percentage")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional telemetry data")


class PSFGridCoordinate(BaseModel):
    """Coordinate in the PSF fitting grid."""
    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")
    ra: Optional[float] = Field(None, description="Right ascension in degrees")
    dec: Optional[float] = Field(None, description="Declination in degrees")
    weight: float = Field(1.0, ge=0, description="Weight for this grid point")
    is_valid: bool = Field(True, description="Whether this grid point is valid")


class PSFFitRequest(BaseModel):
    """Request to perform PSF fitting."""
    image_data: List[List[float]] = Field(..., description="2D image data as nested list")
    source_positions: List[PSFGridCoordinate] = Field(..., description="Positions of sources to fit")
    config: Optional[PSFModelConfig] = Field(None, description="PSF model configuration")
    background_model: Optional[List[float]] = Field(None, description="Background model coefficients")
    variance_map: Optional[List[List[float]]] = Field(None, description="Variance map for weighted fitting")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: float = Field(..., description="Request timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")


class PSFFitResponse(BaseModel):
    """Response from PSF fitting operation."""
    request_id: str = Field(..., description="Request identifier")
    success: bool = Field(..., description="Whether the operation succeeded")
    results: List[PSFFitResult] = Field(default_factory=list, description="Fit results for each source")
    sources: List[PSFSource] = Field(default_factory=list, description="Detected sources with fit parameters")
    total_duration_ms: float = Field(0.0, ge=0, description="Total operation duration in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    timestamp: float = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")