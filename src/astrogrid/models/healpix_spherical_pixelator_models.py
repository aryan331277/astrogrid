"""
Pydantic data schemas, telemetry records, and coordinate models for the
healpix_spherical_pixelator module.

This module defines the data contracts for the HealPix spherical pixelation
pipeline, including input validation, coordinate transformations, and
telemetry records for observability.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
import math


class HealPixInputSchema(BaseModel):
    """
    Input schema for the HealPix spherical pixelator module.
    Validates celestial coordinates and pixelization parameters.
    """
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees [0, 360)")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees [-90, 90]")
    nside: int = Field(..., ge=1, description="HealPix Nside parameter (power of 2)")
    order: str = Field("NESTED", description="Pixel ordering scheme: NESTED or RING")

    @field_validator("nside")
    @classmethod
    def validate_nside_power_of_two(cls, v: int) -> int:
        """Ensure nside is a power of 2."""
        if v < 1 or (v & (v - 1)) != 0:
            raise ValueError(f"nside must be a power of 2, got {v}")
        return v

    @field_validator("order")
    @classmethod
    def validate_order(cls, v: str) -> str:
        """Validate pixel ordering scheme."""
        v_upper = v.upper()
        if v_upper not in ("NESTED", "RING"):
            raise ValueError(f"order must be 'NESTED' or 'RING', got '{v}'")
        return v_upper

    @field_validator("ra")
    @classmethod
    def validate_ra_range(cls, v: float) -> float:
        """Normalize RA to [0, 360)."""
        return v % 360.0


class HealPixCoordinate(BaseModel):
    """
    Represents a point on the celestial sphere with both degree and
    radian representations.
    """
    ra_deg: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")

    @property
    def ra_rad(self) -> float:
        """Right ascension in radians."""
        return math.radians(self.ra_deg)

    @property
    def dec_rad(self) -> float:
        """Declination in radians."""
        return math.radians(self.dec_deg)

    @property
    def unit_vector(self) -> List[float]:
        """
        Convert spherical coordinates to a unit vector in Cartesian space.
        Returns [x, y, z] where:
          x = cos(dec) * cos(ra)
          y = cos(dec) * sin(ra)
          z = sin(dec)
        """
        ra = self.ra_rad
        dec = self.dec_rad
        x = math.cos(dec) * math.cos(ra)
        y = math.cos(dec) * math.sin(ra)
        z = math.sin(dec)
        return [x, y, z]

    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float) -> "HealPixCoordinate":
        """
        Create a HealPixCoordinate from a Cartesian unit vector.
        """
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-15:
            raise ValueError("Cannot convert zero vector to spherical coordinates")
        x, y, z = x / norm, y / norm, z / norm
        ra = math.atan2(y, x)
        if ra < 0:
            ra += 2 * math.pi
        dec = math.asin(max(-1.0, min(1.0, z)))
        return cls(ra_deg=math.degrees(ra), dec_deg=math.degrees(dec))


class HealPixPixel(BaseModel):
    """
    Represents a single HealPix pixel on the sphere.
    """
    pixel_index: int = Field(..., ge=0, description="Linear pixel index")
    nside: int = Field(..., ge=1, description="Nside parameter used")
    order: str = Field("NESTED", description="Pixel ordering: NESTED or RING")
    center_ra_deg: float = Field(..., ge=0.0, le=360.0, description="Center RA in degrees")
    center_dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Center Dec in degrees")
    area_sr: float = Field(..., ge=0.0, description="Pixel area in steradians")

    @field_validator("order")
    @classmethod
    def validate_order(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in ("NESTED", "RING"):
            raise ValueError(f"order must be 'NESTED' or 'RING', got '{v}'")
        return v_upper

    @property
    def n_pix(self) -> int:
        """Total number of pixels at this nside."""
        return 12 * self.nside * self.nside

    @property
    def center_coordinate(self) -> HealPixCoordinate:
        """Return the center as a HealPixCoordinate."""
        return HealPixCoordinate(ra_deg=self.center_ra_deg, dec_deg=self.center_dec_deg)


class HealPixPixelizationResult(BaseModel):
    """
    Result of pixelizing a set of celestial coordinates.
    """
    input_coordinates: List[HealPixCoordinate] = Field(..., description="Input coordinates")
    pixels: List[HealPixPixel] = Field(..., description="Resulting pixels")
    nside: int = Field(..., ge=1, description="Nside used for pixelization")
    order: str = Field("NESTED", description="Pixel ordering scheme")
    success: bool = Field(True, description="Whether pixelization succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class HealPixTelemetryRecord(BaseModel):
    """
    Telemetry record for observability of the HealPix pixelator pipeline.
    """
    timestamp: float = Field(..., description="Unix timestamp of the event")
    operation: str = Field(..., description="Operation name (e.g., 'pixelize', 'validate')")
    nside: int = Field(..., ge=1, description="Nside parameter")
    n_input_points: int = Field(..., ge=0, description="Number of input points")
    n_output_pixels: int = Field(..., ge=0, description="Number of output pixels")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    success: bool = Field(True, description="Whether the operation succeeded")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class HealPixBatchInput(BaseModel):
    """
    Batch input for pixelizing multiple celestial coordinates.
    """
    coordinates: List[HealPixInputSchema] = Field(..., min_length=1, description="List of input coordinates")
    nside: int = Field(..., ge=1, description="Nside parameter")
    order: str = Field("NESTED", description="Pixel ordering: NESTED or RING")

    @field_validator("nside")
    @classmethod
    def validate_nside_power_of_two(cls, v: int) -> int:
        if v < 1 or (v & (v - 1)) != 0:
            raise ValueError(f"nside must be a power of 2, got {v}")
        return v

    @field_validator("order")
    @classmethod
    def validate_order(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in ("NESTED", "RING"):
            raise ValueError(f"order must be 'NESTED' or 'RING', got '{v}'")
        return v_upper


class HealPixBatchResult(BaseModel):
    """
    Batch result for pixelizing multiple celestial coordinates.
    """
    results: List[HealPixPixelizationResult] = Field(..., description="Individual results")
    total_input_points: int = Field(..., ge=0, description="Total input points processed")
    total_output_pixels: int = Field(..., ge=0, description="Total unique pixels")
    success: bool = Field(True, description="Overall success flag")
    error_message: Optional[str] = Field(None, description="Error message if failed")