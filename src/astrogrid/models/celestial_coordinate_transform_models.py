"""Pydantic data schemas, telemetry records, and coordinate models for celestial_coordinate_transform."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List, Union
from datetime import datetime
from enum import Enum
import numpy as np


class CoordinateSystem(str, Enum):
    ICRF = "ICRF"
    ITRF = "ITRF"
    GCRS = "GCRS"


class CelestialCoordinate(BaseModel):
    """Represents a celestial coordinate in a specified reference frame."""
    system: CoordinateSystem = Field(..., description="Reference frame of the coordinate")
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")
    distance: Optional[float] = Field(None, ge=0.0, description="Distance in parsecs (if applicable)")
    timestamp: datetime = Field(..., description="Observation timestamp in UTC")

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


class TerrestrialCoordinate(BaseModel):
    """Represents a terrestrial coordinate in ITRF."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Geodetic latitude in degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Geodetic longitude in degrees")
    altitude: float = Field(0.0, description="Altitude above ellipsoid in meters")
    timestamp: datetime = Field(..., description="Observation timestamp in UTC")

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if v < -90 or v > 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if v < -180 or v > 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v


class TransformationRequest(BaseModel):
    """Request to transform coordinates between reference frames."""
    source_coordinate: Union[CelestialCoordinate, TerrestrialCoordinate] = Field(
        ..., description="Source coordinate to transform"
    )
    target_system: CoordinateSystem = Field(..., description="Target reference frame")
    include_uncertainty: bool = Field(False, description="Whether to include uncertainty estimates")


class TransformationResult(BaseModel):
    """Result of a coordinate transformation."""
    source_system: CoordinateSystem = Field(..., description="Source reference frame")
    target_system: CoordinateSystem = Field(..., description="Target reference frame")
    transformed_coordinate: Union[CelestialCoordinate, TerrestrialCoordinate] = Field(
        ..., description="Transformed coordinate"
    )
    rotation_matrix: Optional[List[List[float]]] = Field(
        None, description="3x3 rotation matrix used in transformation"
    )
    uncertainty: Optional[float] = Field(None, ge=0.0, description="Uncertainty in arcseconds")
    timestamp: datetime = Field(..., description="Transformation timestamp")


class TelemetryRecord(BaseModel):
    """Telemetry record for coordinate transformation operations."""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Operation timestamp")
    source_system: CoordinateSystem = Field(..., description="Source reference frame")
    target_system: CoordinateSystem = Field(..., description="Target reference frame")
    success: bool = Field(..., description="Whether the transformation succeeded")
    processing_time_ms: float = Field(0.0, ge=0.0, description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if transformation failed")
    input_coordinate: Optional[Union[CelestialCoordinate, TerrestrialCoordinate]] = Field(
        None, description="Input coordinate for debugging"
    )
    output_coordinate: Optional[Union[CelestialCoordinate, TerrestrialCoordinate]] = Field(
        None, description="Output coordinate for debugging"
    )


class BatchTransformationRequest(BaseModel):
    """Request to transform multiple coordinates."""
    coordinates: List[Union[CelestialCoordinate, TerrestrialCoordinate]] = Field(
        ..., min_length=1, description="List of coordinates to transform"
    )
    target_system: CoordinateSystem = Field(..., description="Target reference frame")
    include_uncertainty: bool = Field(False, description="Whether to include uncertainty estimates")


class BatchTransformationResult(BaseModel):
    """Result of a batch coordinate transformation."""
    results: List[TransformationResult] = Field(..., description="List of transformation results")
    total_count: int = Field(..., ge=0, description="Total number of transformations")
    success_count: int = Field(..., ge=0, description="Number of successful transformations")
    failure_count: int = Field(..., ge=0, description="Number of failed transformations")
    timestamp: datetime = Field(..., description="Batch operation timestamp")