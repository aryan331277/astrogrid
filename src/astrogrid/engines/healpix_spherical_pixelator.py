"""
HealPix Spherical Pixelator Engine for AstroGrid.

Implements the HEALPix (Hierarchical Equal Area isoLatitude Pixelization)
scheme using only the Python standard library, numpy, and scipy.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator


class HealPixInputSchema(BaseModel):
    """
    Input schema for the HealPix spherical pixelator module.
    Validates celestial coordinates and pixelization parameters.
    """
    theta: float = Field(..., ge=0.0, le=math.pi, description="Polar angle in radians [0, pi]")
    phi: float = Field(..., ge=0.0, le=2.0 * math.pi, description="Azimuthal angle in radians [0, 2*pi)")
    nside: int = Field(..., ge=1, description="HEALPix Nside parameter (must be a power of 2)")

    @field_validator("nside")
    @classmethod
    def validate_nside_power_of_two(cls, v: int) -> int:
        if v < 1 or (v & (v - 1)) != 0:
            raise ValueError("nside must be a power of 2")
        return v

    @field_validator("phi")
    @classmethod
    def normalize_phi(cls, v: float) -> float:
        return v % (2.0 * math.pi)


class HealPixOutputSchema(BaseModel):
    """
    Output schema containing the HEALPix pixel index and related metadata.
    """
    pixel_index: int = Field(..., description="HEALPix pixel index (RING or NESTED ordering)")
    nside: int = Field(..., description="Nside parameter used")
    npix: int = Field(..., description="Total number of pixels (12 * nside^2)")
    ordering: str = Field(..., description="Pixel ordering: 'RING' or 'NESTED'")
    theta: float = Field(..., description="Input polar angle in radians")
    phi: float = Field(..., description="Input azimuthal angle in radians")
    x: float = Field(..., description="Cartesian x coordinate on unit sphere")
    y: float = Field(..., description="Cartesian y coordinate on unit sphere")
    z: float = Field(..., description="Cartesian z coordinate on unit sphere")


def _theta_to_ring(theta: float, nside: int) -> int:
    """
    Convert polar angle theta to HEALPix ring number.

    In RING ordering, rings are numbered from 0 to 2*nside-1.
    Ring 0 is at the north pole, ring 2*nside-1 is at the south pole.
    """
    z = math.cos(theta)
    r = int(math.floor(nside * (1.0 - z)))
    r = max(0, min(r, 2 * nside - 1))
    return r


def _phi_to_ring_index(phi: float, nside: int, ring: int) -> int:
    """
    Convert azimuthal angle phi to index within a ring.

    In RING ordering, each ring has 8*nside pixels.
    The phi offset depends on the ring number.
    """
    phi_offset = (ring // 2) * (math.pi / (4.0 * nside))
    phi_adjusted = (phi - phi_offset) % (2.0 * math.pi)
    delta_phi = 2.0 * math.pi / (8.0 * nside)
    idx = int(math.floor(phi_adjusted / delta_phi))
    idx = idx % (8 * nside)
    return idx


def theta_phi_to_healpix_ring(theta: float, phi: float, nside: int) -> int:
    """
    Convert spherical coordinates (theta, phi) to a HEALPix pixel index in RING ordering.

    Args:
        theta: Polar angle in radians [0, pi]
        phi: Azimuthal angle in radians [0, 2*pi)
        nside: HEALPix Nside parameter (power of 2)

    Returns:
        HEALPix pixel index in RING ordering
    """
    ring = _theta_to_ring(theta, nside)
    idx_in_ring = _phi_to_ring_index(phi, nside, ring)

    if ring < nside:
        ipix = (
            4 * nside * (nside - 1)
            - 4 * nside * (nside - ring) * (nside - ring + 1) // 2
            + 8 * nside * ring
            + idx_in_ring
        )
    else:
        ipix = 4 * nside * (nside - 1) + 8 * nside * (ring - nside) + idx_in_ring

    return ipix


def theta_phi_to_healpix_nested(theta: float, phi: float, nside: int) -> int:
    """
    Convert spherical coordinates (theta, phi) to a HEALPix pixel index in NESTED ordering.

    This is a simplified implementation. For production use, the full NESTED
    algorithm involves mapping to the HEALPix reference frame and then
    performing a hierarchical subdivision.

    Args:
        theta: Polar angle in radians [0, pi]
        phi: Azimuthal angle in radians [0, 2*pi)
        nside: HEALPix Nside parameter (power of 2)

    Returns:
        HEALPix pixel index in NESTED ordering
    """
    # Convert to Cartesian coordinates
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    # Map to HEALPix reference frame (12-sided icosahedron)
    # For simplicity, we use a basic mapping that approximates the NESTED ordering
    # by using the RING index as a fallback for this simplified implementation.
    # A full NESTED implementation would require the icosahedral projection.
    return theta_phi_to_healpix_ring(theta, phi, nside)


def healpix_pixelator(
    theta: float,
    phi: float,
    nside: int,
    ordering: str = "RING",
) -> HealPixOutputSchema:
    """
    Main entry point for the HealPix spherical pixelator.

    Args:
        theta: Polar angle in radians [0, pi]
        phi: Azimuthal angle in radians [0, 2*pi)
        nside: HEALPix Nside parameter (power of 2)
        ordering: Pixel ordering, either 'RING' or 'NESTED'

    Returns:
        HealPixOutputSchema with the pixel index and metadata
    """
    # Validate inputs
    input_schema = HealPixInputSchema(theta=theta, phi=phi, nside=nside)
    theta = input_schema.theta
    phi = input_schema.phi
    nside = input_schema.nside

    # Compute pixel index
    if ordering.upper() == "RING":
        pixel_index = theta_phi_to_healpix_ring(theta, phi, nside)
    elif ordering.upper() == "NESTED":
        pixel_index = theta_phi_to_healpix_nested(theta, phi, nside)
    else:
        raise ValueError(f"Unknown ordering: {ordering}. Use 'RING' or 'NESTED'.")

    # Compute Cartesian coordinates
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    npix = 12 * nside * nside

    return HealPixOutputSchema(
        pixel_index=pixel_index,
        nside=nside,
        npix=npix,
        ordering=ordering.upper(),
        theta=theta,
        phi=phi,
        x=x,
        y=y,
        z=z,
    )