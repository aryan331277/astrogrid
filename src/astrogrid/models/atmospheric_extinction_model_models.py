"""
Pydantic data schemas, telemetry records, and coordinate models for atmospheric extinction.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from enum import Enum
from datetime import datetime
import math


class ExtinctionModelType(str, Enum):
    """
    Supported atmospheric extinction model types.
    """
    STANDARD = "standard"
    CUSTOM = "custom"
    NONE = "none"


class ExtinctionCoefficients(BaseModel):
    """
    Atmospheric extinction coefficients (a + b * airmass).
    """
    a: float = Field(..., description="Extinction coefficient at zenith (mag/airmass)")
    b: float = Field(..., description="Differential extinction coefficient (mag/airmass)")
    wavelength_nm: float = Field(..., description="Wavelength in nanometers")
    band: Optional[str] = Field(None, description="Photometric band name (e.g., 'V', 'R')")

    @field_validator("a", "b")
    @classmethod
    def validate_coefficients(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Extinction coefficients must be non-negative")
        return v

    @field_validator("wavelength_nm")
    @classmethod
    def validate_wavelength(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Wavelength must be positive")
        return v


class AirmassModel(str, Enum):
    """
    Airmass calculation model.
    """
    KENNEDY = "kennedy"
    YOUNG = "young"
    PIERCE = "pierce"
    SIMPLE = "simple"


class AtmosphericState(BaseModel):
    """
    Atmospheric conditions affecting extinction.
    """
    pressure_hpa: float = Field(..., description="Surface pressure in hectopascals")
    temperature_k: float = Field(..., description="Temperature in Kelvin")
    humidity_pct: float = Field(0.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    o3_column_du: float = Field(0.0, ge=0.0, description="Ozone column in Dobson units")
    aerosol_tau: float = Field(0.0, ge=0.0, description="Aerosol optical depth at 550nm")

    @field_validator("pressure_hpa")
    @classmethod
    def validate_pressure(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Pressure must be positive")
        return v

    @field_validator("temperature_k")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Temperature must be positive")
        return v


class ExtinctionObservation(BaseModel):
    """
    A single extinction observation record.
    """
    timestamp: datetime = Field(..., description="Observation timestamp (UTC)")
    source_ra_deg: float = Field(..., ge=-180.0, le=180.0, description="Source RA in degrees")
    source_dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Source Dec in degrees")
    observer_lat_deg: float = Field(..., ge=-90.0, le=90.0, description="Observer latitude in degrees")
    observer_lon_deg: float = Field(..., ge=-180.0, le=180.0, description="Observer longitude in degrees")
    observer_alt_m: float = Field(0.0, description="Observer altitude in meters")
    observed_mag: float = Field(..., description="Observed magnitude")
    reference_mag: float = Field(..., description="Reference (top-of-atmosphere) magnitude")
    wavelength_nm: float = Field(..., gt=0.0, description="Wavelength in nanometers")
    band: Optional[str] = Field(None, description="Photometric band")
    airmass: Optional[float] = Field(None, gt=0.0, description="Computed airmass")
    extinction_mag: Optional[float] = Field(None, description="Measured extinction in magnitudes")
    model_type: ExtinctionModelType = Field(ExtinctionModelType.STANDARD, description="Model used")
    airmass_model: AirmassModel = Field(AirmassModel.KENNEDY, description="Airmass model used")
    atmospheric_state: Optional[AtmosphericState] = Field(None, description="Atmospheric conditions")
    quality_flag: int = Field(0, ge=0, le=255, description="Quality flag bitmask")
    notes: Optional[str] = Field(None, description="Free-form notes")

    @field_validator("source_ra_deg")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if v < -180.0 or v > 180.0:
            raise ValueError("RA must be in [-180, 180] degrees")
        return v

    @field_validator("source_dec_deg")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if v < -90.0 or v > 90.0:
            raise ValueError("Dec must be in [-90, 90] degrees")
        return v


class ExtinctionModelConfig(BaseModel):
    """
    Configuration for an atmospheric extinction model.
    """
    model_type: ExtinctionModelType = Field(ExtinctionModelType.STANDARD, description="Model type")
    airmass_model: AirmassModel = Field(AirmassModel.KENNEDY, description="Airmass calculation model")
    coefficients: List[ExtinctionCoefficients] = Field(
        default_factory=list, description="Extinction coefficients per band/wavelength"
    )
    default_atmosphere: Optional[AtmosphericState] = Field(
        None, description="Default atmospheric state if not provided per observation"
    )
    max_airmass: float = Field(3.0, gt=0.0, description="Maximum airmass for valid observations")
    min_elevation_deg: float = Field(10.0, ge=0.0, le=90.0, description="Minimum elevation angle in degrees")
    description: Optional[str] = Field(None, description="Model description")

    @field_validator("coefficients")
    @classmethod
    def validate_coefficients_list(cls, v: List[ExtinctionCoefficients]) -> List[ExtinctionCoefficients]:
        if len(v) > 100:
            raise ValueError("Maximum 100 coefficients allowed")
        return v


class ExtinctionResult(BaseModel):
    """
    Result of an extinction calculation.
    """
    airmass: float = Field(..., gt=0.0, description="Computed airmass")
    extinction_mag: float = Field(..., description="Total extinction in magnitudes")
    transmission: float = Field(..., ge=0.0, le=1.0, description="Atmospheric transmission (0-1)")
    zenith_angle_deg: float = Field(..., ge=0.0, le=180.0, description="Zenith angle in degrees")
    elevation_deg: float = Field(..., ge=-90.0, le=90.0, description="Elevation angle in degrees")
    wavelength_nm: float = Field(..., gt=0.0, description="Wavelength in nanometers")
    band: Optional[str] = Field(None, description="Photometric band")
    model_type: ExtinctionModelType = Field(..., description="Model type used")
    airmass_model: AirmassModel = Field(..., description="Airmass model used")
    valid: bool = Field(True, description="Whether the result is physically valid")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")
    computed_at: datetime = Field(default_factory=datetime.utcnow, description="Computation timestamp")

    @field_validator("transmission")
    @classmethod
    def validate_transmission(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Transmission must be in [0, 1]")
        return v


class ExtinctionTelemetry(BaseModel):
    """
    Telemetry record for monitoring extinction model performance.
    """
    timestamp: datetime = Field(..., description="Telemetry timestamp (UTC)")
    model_id: str = Field(..., description="Unique model identifier")
    model_type: ExtinctionModelType = Field(..., description="Model type")
    observations_count: int = Field(0, ge=0, description="Number of observations processed")
    mean_residual_mag: float = Field(0.0, description="Mean residual in magnitudes")
    rms_residual_mag: float = Field(0.0, ge=0.0, description="RMS residual in magnitudes")
    max_airmass_observed: float = Field(0.0, ge=0.0, description="Maximum airmass observed")
    min_elevation_observed: float = Field(90.0, ge=-90.0, le=90.0, description="Minimum elevation observed")
    status: str = Field("ok", description="Model status (ok, degraded, error)")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"ok", "degraded