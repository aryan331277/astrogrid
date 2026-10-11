"""Pydantic data schemas, telemetry records, and coordinate models for the Arnett supernova model."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from enum import Enum
import math


class OpacityModel(str, Enum):
    """
    Defines the opacity treatment in the Arnett model.
    """
    GREY = "grey"
    FREQUENCY_DEPENDENT = "frequency_dependent"
    COMPOSITE = "composite"


class SupernovaType(str, Enum):
    """
    Classification of supernova types.
    """
    TYPE_Ia = "Ia"
    TYPE_Ib = "Ib"
    TYPE_Ic = "Ic"
    TYPE_II = "II"
    TYPE_II_P = "II-P"
    TYPE_II_L = "II-L"
    TYPE_IIb = "IIb"


class ArnettModelParameters(BaseModel):
    """
    Physical parameters for the Arnett supernova model.
    """
    ejecta_mass: float = Field(..., gt=0, description="Total ejecta mass in solar masses")
    nickel_mass: float = Field(..., ge=0, description="Mass of Ni-56 in solar masses")
    opacity: float = Field(..., gt=0, description="Grey opacity in cm^2/g")
    diffusion_timescale: float = Field(..., gt=0, description="Diffusion timescale in days")
    decay_timescale: float = Field(..., gt=0, description="Ni-56 decay timescale in days")
    initial_velocity: float = Field(..., gt=0, description="Initial ejecta velocity in km/s")
    peak_luminosity: Optional[float] = Field(None, ge=0, description="Peak luminosity in erg/s")
    peak_time: Optional[float] = Field(None, ge=0, description="Time to peak in days")

    @field_validator("ejecta_mass", "nickel_mass", "opacity", "diffusion_timescale", "decay_timescale", "initial_velocity")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class ArnettModelResult(BaseModel):
    """
    Results from the Arnett model calculation.
    """
    time_days: List[float] = Field(..., description="Time array in days")
    luminosity_erg_s: List[float] = Field(..., description="Luminosity in erg/s")
    internal_energy_erg: List[float] = Field(..., description="Internal energy in erg")
    diffusion_timescale_days: List[float] = Field(..., description="Diffusion timescale in days")
    peak_luminosity_erg_s: float = Field(..., ge=0, description="Peak luminosity in erg/s")
    peak_time_days: float = Field(..., ge=0, description="Time to peak in days")
    total_energy_erg: float = Field(..., ge=0, description="Total radiated energy in erg")


class SupernovaObservation(BaseModel):
    """
    Telemetry record for a supernova observation.
    """
    observation_id: str = Field(..., description="Unique observation identifier")
    supernova_id: str = Field(..., description="Supernova identifier")
    timestamp: float = Field(..., description="Unix timestamp of observation")
    redshift: float = Field(..., ge=0, description="Cosmological redshift")
    distance_mpc: float = Field(..., gt=0, description="Luminosity distance in Mpc")
    ra_degrees: float = Field(..., ge=0, le=360, description="Right ascension in degrees")
    dec_degrees: float = Field(..., ge=-90, le=90, description="Declination in degrees")
    magnitude: Optional[float] = Field(None, description="Apparent magnitude")
    band: Optional[str] = Field(None, description="Photometric band")
    uncertainty: Optional[float] = Field(None, ge=0, description="Measurement uncertainty")
    detector_id: Optional[str] = Field(None, description="Detector identifier")
    exposure_time_s: Optional[float] = Field(None, gt=0, description="Exposure time in seconds")

    @field_validator("ra_degrees")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if v < 0 or v > 360:
            raise ValueError("RA must be between 0 and 360 degrees")
        return v

    @field_validator("dec_degrees")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if v < -90 or v > 90:
            raise ValueError("Dec must be between -90 and 90 degrees")
        return v


class ArnettModelConfig(BaseModel):
    """
    Configuration for running the Arnett model.
    """
    opacity_model: OpacityModel = Field(default=OpacityModel.GREY, description="Opacity treatment")
    supernova_type: SupernovaType = Field(default=SupernovaType.TYPE_Ia, description="Supernova type")
    num_time_steps: int = Field(default=100, gt=0, description="Number of time steps")
    max_time_days: float = Field(default=100.0, gt=0, description="Maximum simulation time in days")
    time_step_days: Optional[float] = Field(None, gt=0, description="Fixed time step in days")
    use_analytic_solution: bool = Field(default=True, description="Use analytic Arnett solution")
    include_decay_heat: bool = Field(default=True, description="Include Ni-56 decay heating")
    include_diffusion: bool = Field(default=True, description="Include diffusion losses")

    @field_validator("num_time_steps")
    @classmethod
    def validate_num_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Number of time steps must be positive")
        return v


class ArnettModelTelemetry(BaseModel):
    """
    Telemetry record for Arnett model execution.
    """
    run_id: str = Field(..., description="Unique run identifier")
    start_time: float = Field(..., description="Unix timestamp of run start")
    end_time: Optional[float] = Field(None, description="Unix timestamp of run end")
    status: str = Field(default="running", description="Run status: running, completed, failed")
    config: ArnettModelConfig = Field(..., description="Model configuration")
    parameters: ArnettModelParameters = Field(..., description="Model parameters")
    result: Optional[ArnettModelResult] = Field(None, description="Model result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    computation_time_s: Optional[float] = Field(None, ge=0, description="Computation time in seconds")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Peak memory usage in MB")


class CoordinateSystem(str, Enum):
    """
    Supported coordinate systems.
    """
    ICRS = "icrs"
    GALACTIC = "galactic"
    ECLIPTIC = "ecliptic"


class CelestialCoordinates(BaseModel):
    """
    Celestial coordinates for a supernova.
    """
    ra_degrees: float = Field(..., ge=0, le=360, description="Right ascension in degrees")
    dec_degrees: float = Field(..., ge=-90, le=90, description="Declination in degrees")
    coordinate_system: CoordinateSystem = Field(default=CoordinateSystem.ICRS, description="Coordinate system")
    epoch: Optional[float] = Field(None, description="Epoch (e.g., 2000.0 for J2000)")
    uncertainty_arcsec: Optional[float] = Field(None, ge=0, description="Positional uncertainty in arcseconds")

    @field_validator("ra_degrees")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if v < 0 or v > 360:
            raise ValueError("RA must be between 0 and 360 degrees")
        return v

    @field_validator("dec_degrees")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if v < -90 or v > 90:
            raise ValueError("Dec must be between -90 and 90 degrees")
        return v

    def to_rad(self) -> tuple:
        """Convert coordinates to radians."""
        return math.radians(self.ra_degrees), math.radians(self.dec_degrees)

    def angular_distance_degrees(self, other: "CelestialCoordinates") -> float:
        """Calculate angular distance to another coordinate in degrees."""
        ra1, dec1 = self.to_rad()
        ra2, dec2 = other.to_rad()
        cos_dist = (math.sin(dec1) * math.sin(dec2) +
                    math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
        cos_dist = max(-1.0, min(1.0, cos_dist))
        return math.degrees(math.acos(cos_dist))