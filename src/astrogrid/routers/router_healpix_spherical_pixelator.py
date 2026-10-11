"""FastAPI router for healpix_spherical_pixelator module."""

from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.astrogrid.engines.healpix_spherical_pixelator import (
    HealPixInputSchema,
    HealPixOutputSchema,
    _theta_to_ring,
    _phi_to_ring_index,
)

router = APIRouter(prefix='/api/healpix_spherical_pixelator', tags=['healpix_spherical_pixelator'])


class HealPixRequest(BaseModel):
    """Request model for HEALPix pixelation."""
    theta: List[float] = Field(..., description="Colatitude in radians [0, pi]")
    phi: List[float] = Field(..., description="Longitude in radians [0, 2*pi)")
    nside: int = Field(..., description="HEALPix nside parameter (must be power of 2)")


class HealPixResponse(BaseModel):
    """Response model for HEALPix pixelation."""
    pixel_indices: List[int]
    nside: int
    count: int


@router.post("/pixelate", response_model=HealPixResponse)
def pixelate_points(request: HealPixRequest):
    """Convert spherical coordinates to HEALPix pixel indices."""
    # Validate input
    input_schema = HealPixInputSchema(
        theta=request.theta,
        phi=request.phi,
        nside=request.nside
    )
    
    try:
        input_schema.validate_nside_power_of_two()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Normalize phi to [0, 2*pi)
    phi_normalized = input_schema.normalize_phi()
    
    theta_arr = np.array(request.theta)
    phi_arr = np.array(phi_normalized)
    nside = request.nside
    
    # Compute pixel indices
    pixel_indices = []
    for i in range(len(theta_arr)):
        ring = _theta_to_ring(theta_arr[i], nside)
        ring_index = _phi_to_ring_index(phi_arr[i], nside, ring)
        # Compute base pixel index for the ring
        # Number of pixels in rings before this one
        if ring < nside:
            # Northern hemisphere: rings 0 to nside-1
            # Each ring has 4*(ring+1) pixels
            base = 4 * ring * (ring + 1)
        else:
            # Equatorial and southern hemisphere
            # Rings nside to 3*nside-1 have 4*nside pixels each
            # Southern hemisphere rings 3*nside to 4*nside-1
            if ring < 3 * nside:
                base = 4 * nside * nside + 4 * nside * (ring - nside)
            else:
                # Southern hemisphere
                south_ring = ring - 3 * nside
                base = 12 * nside * nside - 4 * (nside - south_ring) * (nside - south_ring - 1)
        
        pixel = base + ring_index
        pixel_indices.append(int(pixel))
    
    return HealPixResponse(
        pixel_indices=pixel_indices,
        nside=nside,
        count=len(pixel_indices)
    )


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "module": "healpix_spherical_pixelator"}