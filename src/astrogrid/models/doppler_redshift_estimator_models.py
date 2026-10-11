"""
Pydantic data schemas, telemetry records, and coordinate models for the Doppler Redshift Estimator module.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Literal
import math


class SpectralLine(BaseModel):
    """
    Represents a single observed spectral line.
    """
    rest_wavelength_nm: float = Field(..., gt=0, description="Rest wavelength in nanometers")
    observed_wavelength_nm: float = Field(..., gt=0, description="Observed wavelength in nanometers")
    line_strength: float = Field(1.0, ge=0, le=1, description="Relative strength of the spectral line")
    line_id: Optional[str] = Field(None, description="Identifier for the spectral line (e.g., 'H-alpha')")

    @field_validator("rest_wavelength_nm", "observed_wavelength_nm")
    @classmethod
    def validate_wavelengths(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Wavelengths must be finite numbers")
        return v


class PhotometricBand(BaseModel):
    """
    Represents a photometric band measurement.
    """
    band_name: str = Field(..., description="Name of the photometric band (e.g., 'u', 'g', 'r')")
    flux_density: float = Field(..., gt=0, description="Flux density in Jy or AB magnitude")
    flux_error: float = Field(0.0, ge=0, description="Uncertainty in flux density")
    effective_wavelength_nm: float = Field(..., gt=0, description="Effective wavelength of the band in nm")


class RedshiftEstimate(BaseModel):
    """
    Represents a computed redshift estimate with uncertainty.
    """
    redshift: float = Field(..., description="Dimensionless redshift z")
    uncertainty: float = Field(0.0, ge=0, description="1-sigma uncertainty on redshift")
    method: Literal["spectral_line", "photometric", "hybrid"] = Field(
        "spectral_line", description="Method used for estimation"
    )
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence score of the estimate")
    velocity_kms: Optional[float] = Field(None, description="Recession velocity in km/s (non-relativistic approx)")

    @field_validator("redshift")
    @classmethod
    def validate_redshift(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Redshift must be finite")
        if v < -1.0:
            raise ValueError("Redshift cannot be less than -1.0")
        return v

    def compute_velocity_kms(self) -> float:
        """Compute non-relativistic recession velocity in km/s."""
        c_kms = 299792.458
        return self.redshift * c_kms


class DopplerRedshiftInput(BaseModel):
    """
    Input data for the Doppler redshift estimator.
    """
    source_id: str = Field(..., description="Unique identifier for the astronomical source")
    spectral_lines: List[SpectralLine] = Field(default_factory=list, description="List of observed spectral lines")
    photometric_bands: List[PhotometricBand] = Field(default_factory=list, description="Photometric measurements")
    reference_frame: Literal["CMB", "Galactic", "Local"] = Field("CMB", description="Reference frame for velocity")

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not v or len(v) < 1:
            raise ValueError("Source ID must be non-empty")
        return v


class DopplerRedshiftOutput(BaseModel):
    """
    Output result from the Doppler redshift estimator.
    """
    source_id: str = Field(..., description="Unique identifier for the astronomical source")
    estimate: RedshiftEstimate = Field(..., description="The redshift estimate")
    contributing_lines: List[str] = Field(default_factory=list, description="IDs of spectral lines used")
    contributing_bands: List[str] = Field(default_factory=list, description="Names of photometric bands used")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp of computation")


class RedshiftTelemetryRecord(BaseModel):
    """
    Telemetry record for monitoring the redshift estimator service.
    """
    record_id: str = Field(..., description="Unique record identifier")
    source_id: str = Field(..., description="Source being processed")
    status: Literal["success", "failure", "timeout"] = Field("success", description="Processing status")
    processing_time_ms: float = Field(0.0, ge=0, description="Time taken to process in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if status is failure")
    redshift_value: Optional[float] = Field(None, description="Computed redshift if successful")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp of the record")


class CoordinateSystem(BaseModel):
    """
    Celestial coordinates for a source.
    """
    ra_deg: float = Field(..., ge=0, le=360, description="Right ascension in degrees")
    dec_deg: float = Field(..., ge=-90, le=90, description="Declination in degrees")
    distance_mpc: Optional[float] = Field(None, gt=0, description="Distance in megaparsecs")
    coordinate_system: Literal["ICRS", "FK5", "FK4"] = Field("ICRS", description="Coordinate system")

    @field_validator("ra_deg")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("RA must be finite")
        return v % 360.0

    @field_validator("dec_deg")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Dec must be finite")
        if v < -90 or v > 90:
            raise ValueError("Dec must be between -90 and 90 degrees")
        return v