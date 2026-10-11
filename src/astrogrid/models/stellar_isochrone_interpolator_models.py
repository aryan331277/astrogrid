"""
Pydantic data schemas, telemetry records, and coordinate models for stellar_isochrone_interpolator.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class IsochroneParameterSpace(BaseModel):
    """
    Defines the continuous parameter space for stellar isochrone interpolation.
    """
    age_range: tuple = Field(..., description="Min and max age in years")
    metallicity_range: tuple = Field(..., description="Min and max metallicity [Fe/H]")
    mass_range: tuple = Field(..., description="Min and max stellar mass in solar masses")
    grid_resolution: Dict[str, int] = Field(
        default={"age": 100, "metallicity": 50, "mass": 100},
        description="Number of grid points along each parameter axis"
    )

    @field_validator("age_range", "metallicity_range", "mass_range")
    @classmethod
    def validate_range(cls, v: tuple) -> tuple:
        if len(v) != 2:
            raise ValueError("Range must be a tuple of two values")
        if v[0] >= v[1]:
            raise ValueError("Min value must be less than max value")
        return v


class StellarObservable(BaseModel):
    """
    Represents a set of observable properties for a stellar evolutionary track point.
    """
    bolometric_luminosity: float = Field(..., ge=0, description="Luminosity in solar units")
    effective_temperature: float = Field(..., ge=0, description="Effective temperature in Kelvin")
    absolute_magnitude: Optional[float] = Field(None, description="Absolute magnitude")
    surface_gravity: Optional[float] = Field(None, description="log(g) in cgs")
    radius: Optional[float] = Field(None, ge=0, description="Radius in solar units")

    @field_validator("effective_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 1000 or v > 100000:
            raise ValueError("Effective temperature must be between 1000 and 100000 K")
        return v


class IsochroneTrackPoint(BaseModel):
    """
    A single point along a stellar evolutionary track at a specific age.
    """
    age: float = Field(..., ge=0, description="Age in years")
    mass: float = Field(..., ge=0, description="Stellar mass in solar masses")
    observables: StellarObservable
    evolutionary_stage: str = Field(..., description="Current evolutionary stage")


class IsochroneGrid(BaseModel):
    """
    A discrete grid of isochrone tracks for a given set of parameters.
    """
    parameter_space: IsochroneParameterSpace
    tracks: List[List[IsochroneTrackPoint]] = Field(
        ..., description="Nested list: [metallicity_idx][mass_idx] -> list of track points"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InterpolationResult(BaseModel):
    """
    Result of interpolating stellar observables at a specific parameter point.
    """
    age: float
    metallicity: float
    mass: float
    observables: StellarObservable
    interpolation_method: str = Field(..., description="Method used for interpolation")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in interpolation")
    nearest_track_indices: Optional[Dict[str, int]] = Field(
        None, description="Indices of nearest grid points used"
    )


class InterpolationTelemetry(BaseModel):
    """
    Telemetry record for an interpolation operation.
    """
    timestamp: float = Field(..., description="Unix timestamp of operation")
    operation_id: str = Field(..., description="Unique identifier for this operation")
    input_parameters: Dict[str, float] = Field(..., description="Input parameter values")
    output_result: Optional[InterpolationResult] = Field(None, description="Interpolation result")
    computation_time_ms: float = Field(..., ge=0, description="Time taken in milliseconds")
    success: bool = Field(..., description="Whether the operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    grid_version: Optional[str] = Field(None, description="Version of the isochrone grid used")


class CoordinateSystem(Enum):
    """
    Supported coordinate systems for stellar positions.
    """
    ICRS = "icrs"
    GALACTIC = "galactic"
    ECLIPTIC = "ecliptic"


class StellarPosition(BaseModel):
    """
    Position of a star in a specified coordinate system.
    """
    coordinate_system: CoordinateSystem = Field(default=CoordinateSystem.ICRS)
    ra: Optional[float] = Field(None, description="Right ascension in degrees (ICRS)")
    dec: Optional[float] = Field(None, description="Declination in degrees (ICRS)")
    galactic_l: Optional[float] = Field(None, description="Galactic longitude in degrees")
    galactic_b: Optional[float] = Field(None, description="Galactic latitude in degrees")
    ecliptic_lon: Optional[float] = Field(None, description="Ecliptic longitude in degrees")
    ecliptic_lat: Optional[float] = Field(None, description="Ecliptic latitude in degrees")
    parallax: Optional[float] = Field(None, ge=0, description="Parallax in arcseconds")
    distance_pc: Optional[float] = Field(None, ge=0, description="Distance in parsecs")

    @field_validator("ra")
    @classmethod
    def validate_ra(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 360):
            raise ValueError("Right ascension must be between 0 and 360 degrees")
        return v

    @field_validator("dec", "galactic_b", "ecliptic_lat")
    @classmethod
    def validate_lat(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < -90 or v > 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("galactic_l", "ecliptic_lon")
    @classmethod
    def validate_lon(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 360):
            raise ValueError("Longitude must be between 0 and 360 degrees")
        return v


class IsochroneQuery(BaseModel):
    """
    Query parameters for interpolating stellar isochrones.
    """
    age: float = Field(..., ge=0, description="Age in years")
    metallicity: float = Field(..., description="Metallicity [Fe/H]")
    mass: float = Field(..., ge=0, description="Stellar mass in solar masses")
    interpolation_method: str = Field(
        default="trilinear",
        description="Interpolation method: 'trilinear', 'nearest', 'cubic'"
    )
    include_uncertainty: bool = Field(default=False, description="Include uncertainty estimates")
    coordinate_system: Optional[CoordinateSystem] = Field(
        None, description="Coordinate system for positional data if available"
    )

    @field_validator("interpolation_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = {"trilinear", "nearest", "cubic"}
        if v not in valid_methods:
            raise ValueError(f"Interpolation method must be one of {valid_methods}")
        return v


class IsochroneResponse(BaseModel):
    """
    Response containing interpolated isochrone data.
    """
    query: IsochroneQuery
    result: InterpolationResult
    telemetry: InterpolationTelemetry
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")