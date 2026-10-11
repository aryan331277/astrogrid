"""
WCS Tangent Point Solver for AstroGrid.

Implements the tangent-plane (gnomonic) projection for World Coordinate Systems,
converting between celestial coordinates (RA, Dec) and tangent-plane coordinates (x, y).
"""

import math
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class WcsTangentSolverInput(BaseModel):
    """
    Input schema for the WCS Tangent Point Solver.
    
    Defines the celestial sphere parameters and the tangent point.
    """
    ra: float = Field(..., description="Right Ascension in radians")
    dec: float = Field(..., description="Declination in radians")
    ra0: float = Field(..., description="Tangent point Right Ascension in radians")
    dec0: float = Field(..., description="Tangent point Declination in radians")
    rotation: float = Field(0.0, description="Rotation angle theta in radians")

    @field_validator("ra", "dec", "ra0", "dec0", "rotation")
    @classmethod
    def validate_angle(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Angle must be finite")
        return v


class WcsTangentSolverOutput(BaseModel):
    """
    Output schema for the WCS Tangent Point Solver.
    
    Returns tangent-plane coordinates (x, y) in radians.
    """
    x: float = Field(..., description="Tangent-plane x coordinate in radians")
    y: float = Field(..., description="Tangent-plane y coordinate in radians")
    valid: bool = Field(True, description="Whether the projection is valid (not on the far side of the sphere)")


def _to_cartesian(ra: float, dec: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates (RA, Dec) to Cartesian unit vector.
    
    Returns (x, y, z) where:
        x = cos(dec) * cos(ra)
        y = cos(dec) * sin(ra)
        z = sin(dec)
    """
    cd = math.cos(dec)
    return (cd * math.cos(ra), cd * math.sin(ra), math.sin(dec))


def _from_cartesian(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Convert Cartesian unit vector to spherical coordinates (RA, Dec).
    
    Returns (ra, dec) in radians.
    """
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-15:
        return (0.0, 0.0)
    dec = math.asin(max(-1.0, min(1.0, z / r)))
    ra = math.atan2(y, x)
    return (ra, dec)


def solve_tangent_point(
    ra: float,
    dec: float,
    ra0: float,
    dec0: float,
    rotation: float = 0.0,
) -> Tuple[float, float, bool]:
    """
    Solve for tangent-plane coordinates (x, y) given celestial coordinates.
    
    Uses the gnomonic (tangent-plane) projection:
    1. Convert (ra, dec) and (ra0, dec0) to Cartesian unit vectors.
    2. Compute the angle between them.
    3. Project onto the tangent plane at (ra0, dec0).
    4. Apply rotation.
    
    Args:
        ra: Right Ascension of the point in radians.
        dec: Declination of the point in radians.
        ra0: Tangent point Right Ascension in radians.
        dec0: Tangent point Declination in radians.
        rotation: Rotation angle theta in radians.
        
    Returns:
        Tuple of (x, y, valid) where x, y are tangent-plane coordinates in radians,
        and valid is False if the point is on the far side of the sphere.
    """
    # Convert to Cartesian
    x1, y1, z1 = _to_cartesian(ra, dec)
    x0, y0, z0 = _to_cartesian(ra0, dec0)
    
    # Compute the dot product (cosine of the angular separation)
    cos_sep = x1 * x0 + y1 * y0 + z1 * z0
    
    # Check if the point is on the far side of the sphere
    if cos_sep < 1e-10:
        return (0.0, 0.0, False)
    
    # Compute the component of v1 perpendicular to v0
    px = x1 - cos_sep * x0
    py = y1 - cos_sep * y0
    pz = z1 - cos_sep * z0
    
    # The magnitude of the perpendicular component
    perp_mag = math.sqrt(px * px + py * py + pz * pz)
    
    if perp_mag < 1e-15:
        # Point is at the tangent point
        return (0.0, 0.0, True)
    
    # Compute a basis for the tangent plane
    # e1 = (0, 0, 1) x v0 normalized (if v0 is not near the pole)
    e1x = -z0
    e1y = x0
    e1z = 0.0
    
    e1_mag = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    
    if e1_mag < 1e-15:
        # v0 is near the pole, use a different basis
        e1x = 1.0
        e1y = 0.0
        e1z = 0.0
        e1_mag = 1.0
    else:
        e1x /= e1_mag
        e1y /= e1_mag
        e1z /= e1_mag
    
    # e2 = v0 x e1
    e2x = y0 * e1z - z0 * e1y
    e2y = z0 * e1x - x0 * e1z
    e2z = x0 * e1y - y0 * e1x
    
    # Project the perpendicular component onto e1 and e2
    u = px * e1x + py * e1y + pz * e1z
    v = px * e2x + py * e2y + pz * e2z
    
    # The tangent-plane coordinates
    x_tan = math.atan2(u, cos_sep)
    y_tan = math.atan2(v, cos_sep)
    
    # Apply rotation
    if abs(rotation) > 1e-15:
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        x_rot = x_tan * cos_r - y_tan * sin_r
        y_rot = x_tan * sin_r + y_tan * cos_r
    else:
        x_rot = x_tan
        y_rot = y_tan
    
    return (x_rot, y_rot, True)


def solve_inverse_tangent_point(
    x: float,
    y: float,
    ra0: float,
    dec0: float,
    rotation: float = 0.0,
) -> Tuple[float, float]:
    """
    Solve for celestial coordinates (RA, Dec) given tangent-plane coordinates.
    
    Args:
        x: Tangent-plane x coordinate in radians.
        y: Tangent-plane y coordinate in radians.
        ra0: Tangent point Right Ascension in radians.
        dec0: Tangent point Declination in radians.
        rotation: Rotation angle theta in radians.
        
    Returns:
        Tuple of (ra, dec) in radians.
    """
    # Apply inverse rotation
    if abs(rotation) > 1e-15:
        cos_r = math.cos(-rotation)
        sin_r = math.sin(-rotation)
        x_unrot = x * cos_r - y * sin_r
        y_unrot = x * sin_r + y * cos_r
    else:
        x_unrot = x
        y_unrot = y
    
    # Compute the angular separation
    cos_sep = 1.0 / math.sqrt(1.0 + x_unrot * x_unrot + y_unrot * y_unrot)
    
    # Compute the tangent point Cartesian vector
    x0, y0, z0 = _to_cartesian(ra0, dec0)
    
    # Compute a basis for the tangent plane
    e1x = -z0
    e1y = x0
    e1z = 0.0
    
    e1_mag = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    
    if e1_mag < 1e-15:
        e1x = 1.0
        e1y = 0.0
        e1z = 0.0
        e1_mag = 1.0
    else:
        e1x /= e1_mag
        e1y /= e1_mag
        e1z /= e1_mag
    
    # e2 = v0 x e1
    e2x = y0 * e1z - z0 * e1y
    e2y = z0 * e1x - x0 * e1z
    e2z = x0 * e1y - y0 * e1x
    
    # Reconstruct the Cartesian vector
    # v = cos_sep * v0 + sin_sep * (u * e1 + v * e2)
    # where u = x_unrot / tan_sep, v = y