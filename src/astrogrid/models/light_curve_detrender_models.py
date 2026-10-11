"""Pydantic data schemas, telemetry records, and coordinate models for light_curve_detrender."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import numpy as np


class DetrendingMethod(str, Enum):
    """Enum for supported detrending algorithms."""
    LOWESS = "lowess"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"
    MEDIAN_FILTER = "median_filter"
    SAVITZKY_GOLAY = "savitzky_golay"


class DetrendingConfig(BaseModel):
    """Configuration parameters for the detrending algorithm."""
    method: DetrendingMethod = Field(default=DetrendingMethod.LOWESS, description="Detrending algorithm to use.")
    window_size: int = Field(default=50, ge=3, description="Window size for local regression or filter.")
    degree: int = Field(default=2, ge=1, le=5, description="Polynomial degree for local or global fit.")
    smoothing: float = Field(default=0.3, ge=0.0, le=1.0, description="Smoothing factor (0=raw, 1=smooth).")
    n_iterations: int = Field(default=1, ge=1, le=10, description="Number of iterative refinement steps.")
    robust: bool = Field(default=True, description="Whether to use robust (weighted) fitting.")

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        if v < 3:
            raise ValueError("window_size must be at least 3")
        return v


class LightCurveData(BaseModel):
    """Raw light curve observation data."""
    source_id: str = Field(..., description="Unique identifier for the astronomical source.")
    times: List[float] = Field(..., min_length=2, description="Observation timestamps (MJD or seconds).")
    fluxes: List[float] = Field(..., min_length=2, description="Measured flux values.")
    flux_errors: Optional[List[float]] = Field(None, description="Flux measurement uncertainties.")
    band: str = Field(default="broadband", description="Photometric band or filter name.")
    instrument: str = Field(default="unknown", description="Instrument or telescope name.")
    observation_start: Optional[datetime] = Field(None, description="Start time of observation campaign.")
    observation_end: Optional[datetime] = Field(None, description="End time of observation campaign.")

    @field_validator("times", "fluxes")
    @classmethod
    def validate_lengths_match(cls, v: List[float], info: Any) -> List[float]:
        return v

    def validate_consistency(self) -> bool:
        """Check that times and fluxes have matching lengths."""
        return len(self.times) == len(self.fluxes)


class DetrendedLightCurve(BaseModel):
    """Result of applying a detrending algorithm to a light curve."""
    source_id: str = Field(..., description="Source identifier matching input.")
    times: List[float] = Field(..., description="Original observation times.")
    original_fluxes: List[float] = Field(..., description="Original measured fluxes.")
    detrended_fluxes: List[float] = Field(..., description="Fluxes after removing baseline trend.")
    baseline: List[float] = Field(..., description="Estimated baseline/trend component.")
    residuals: List[float] = Field(..., description="Residuals (original - baseline).")
    method: DetrendingMethod = Field(..., description="Method used for detrending.")
    config: DetrendingConfig = Field(..., description="Configuration used.")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality metric of detrending fit.")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of processing.")

    @field_validator("detrended_fluxes", "baseline", "residuals")
    @classmethod
    def validate_array_lengths(cls, v: List[float], info: Any) -> List[float]:
        return v


class DetrendingTelemetry(BaseModel):
    """Telemetry record for monitoring detrending pipeline performance."""
    run_id: str = Field(..., description="Unique run identifier.")
    source_id: str = Field(..., description="Source being processed.")
    method: DetrendingMethod = Field(..., description="Detrending method applied.")
    n_points: int = Field(..., ge=2, description="Number of data points processed.")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds.")
    memory_peak_mb: float = Field(default=0.0, ge=0.0, description="Peak memory usage in MB.")
    success: bool = Field(default=True, description="Whether processing completed successfully.")
    error_message: Optional[str] = Field(None, description="Error details if failed.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Telemetry record timestamp.")


class DetrendingRequest(BaseModel):
    """API request model for detrending a light curve."""
    source_id: str = Field(..., description="Source identifier.")
    times: List[float] = Field(..., min_length=2, description="Observation times.")
    fluxes: List[float] = Field(..., min_length=2, description="Flux measurements.")
    flux_errors: Optional[List[float]] = Field(None, description="Flux uncertainties.")
    config: Optional[DetrendingConfig] = Field(None, description="Optional detrending configuration.")
    band: str = Field(default="broadband", description="Photometric band.")
    instrument: str = Field(default="unknown", description="Instrument name.")


class DetrendingResponse(BaseModel):
    """API response model containing detrending results."""
    source_id: str = Field(..., description="Source identifier.")
    success: bool = Field(default=True, description="Processing success flag.")
    result: Optional[DetrendedLightCurve] = Field(None, description="Detrending result if successful.")
    error: Optional[str] = Field(None, description="Error message if failed.")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Total processing time.")
    request_id: str = Field(default="", description="Correlation ID for request tracking.")


class BatchDetrendingRequest(BaseModel):
    """Batch request for detrending multiple light curves."""
    requests: List[DetrendingRequest] = Field(..., min_length=1, description="List of detrending requests.")
    parallel: bool = Field(default=True, description="Whether to process in parallel.")
    max_workers: int = Field(default=4, ge=1, le=32, description="Maximum parallel workers.")


class BatchDetrendingResponse(BaseModel):
    """Batch response containing results for multiple light curves."""
    results: List[DetrendingResponse] = Field(..., description="Individual detrending results.")
    total_processed: int = Field(..., ge=0, description="Total number processed.")
    total_failed: int = Field(default=0, ge=0, description="Total number failed.")
    total_time_ms: float = Field(default=0.0, ge=0.0, description="Total batch processing time.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch completion timestamp.")