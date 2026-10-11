"""Pydantic data schemas, telemetry records, and coordinate models for radial_velocity_solver."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
from datetime import datetime
from enum import Enum
import numpy as np


class CoordinateSystem(str, Enum):
    ICRS = "ICRS"
    GCRS = "GCRS"
    BCRS = "BCRS"


class RadialVelocityInput(BaseModel):
    """Input parameters for radial velocity computation."""
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")
    timestamp: datetime = Field(..., description="Observation timestamp in UTC")
    observer_lat: float = Field(..., ge=-90.0, le=90.0, description="Observer latitude in degrees")
    observer_lon: float = Field(..., ge=-180.0, le=180.0, description="Observer longitude in degrees")
    observer_alt: float = Field(default=0.0, description="Observer altitude in meters")
    coordinate_system: CoordinateSystem = Field(default=CoordinateSystem.ICRS)

    @field_validator("ra")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if v < 0 or v > 360:
            raise ValueError("RA must be between 0 and 360 degrees")
        return v

    @field_validator("dec")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if v < -90 or v > 90:
            raise ValueError("Dec must be between -90 and 90 degrees")
        return v


class RadialVelocityResult(BaseModel):
    """Result of radial velocity computation."""
    radial_velocity: float = Field(..., description="Radial velocity in m/s")
    barycentric_correction: float = Field(..., description="Barycentric correction in m/s")
    observer_velocity: float = Field(..., description="Observer velocity component in m/s")
    relativistic_correction: float = Field(..., description="Relativistic correction in m/s")
    timestamp: datetime = Field(..., description="Computation timestamp")
    coordinate_system: CoordinateSystem = Field(..., description="Coordinate system used")
    uncertainty: Optional[float] = Field(default=None, description="Velocity uncertainty in m/s")


class RadialVelocityTelemetry(BaseModel):
    """Telemetry record for radial velocity solver operations."""
    solver_id: str = Field(..., description="Unique solver identifier")
    input: RadialVelocityInput = Field(..., description="Input parameters")
    result: Optional[RadialVelocityResult] = Field(default=None, description="Computation result")
    status: str = Field(default="pending", description="Processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation time")
    processed_at: Optional[datetime] = Field(default=None, description="Processing completion time")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")


class ObserverPosition(BaseModel):
    """Observer position in various coordinate systems."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    alt: float = Field(default=0.0, description="Altitude in meters")
    x_icrs: Optional[float] = Field(default=None, description="X coordinate in ICRS (m)")
    y_icrs: Optional[float] = Field(default=None, description="Y coordinate in ICRS (m)")
    z_icrs: Optional[float] = Field(default=None, description="Z coordinate in ICRS (m)")
    velocity_x: Optional[float] = Field(default=None, description="Velocity X component (m/s)")
    velocity_y: Optional[float] = Field(default=None, description="Velocity Y component (m/s)")
    velocity_z: Optional[float] = Field(default=None, description="Velocity Z component (m/s)")


class RadialVelocityBatchRequest(BaseModel):
    """Batch request for multiple radial velocity computations."""
    inputs: List[RadialVelocityInput] = Field(..., min_length=1, description="List of input parameters")
    priority: int = Field(default=0, ge=0, le=10, description="Processing priority (0-10)")
    request_id: Optional[str] = Field(default=None, description="Optional request identifier")


class RadialVelocityBatchResponse(BaseModel):
    """Batch response with multiple radial velocity results."""
    request_id: str = Field(..., description="Request identifier")
    results: List[RadialVelocityResult] = Field(..., description="List of results")
    errors: List[Optional[str]] = Field(default_factory=list, description="Error messages for failed computations")
    total_count: int = Field(..., description="Total number of inputs")
    success_count: int = Field(..., description="Number of successful computations")
    failed_count: int = Field(..., description="Number of failed computations")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing completion time")