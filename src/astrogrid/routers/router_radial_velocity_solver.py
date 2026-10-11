"""FastAPI router for radial velocity solver endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

from ..engines.radial_velocity_solver import (
    RadialVelocitySolver,
    RadialVelocityInput,
    RadialVelocityResult,
)

router = APIRouter(prefix='/api/radial_velocity_solver', tags=['radial_velocity_solver'])


class RadialVelocityRequest(BaseModel):
    """Request model for radial velocity calculation."""
    ra: float = Field(..., description="Right ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")
    vx: float = Field(..., description="Velocity component in x-direction (km/s)")
    vy: float = Field(..., description="Velocity component in y-direction (km/s)")
    vz: float = Field(..., description="Velocity component in z-direction (km/s)")
    observer_ra: Optional[float] = Field(None, description="Observer right ascension in degrees")
    observer_dec: Optional[float] = Field(None, description="Observer declination in degrees")


class RadialVelocityResponse(BaseModel):
    """Response model for radial velocity calculation."""
    radial_velocity: float = Field(..., description="Radial velocity in km/s")
    transverse_velocity: float = Field(..., description="Transverse velocity in km/s")
    total_velocity: float = Field(..., description="Total velocity magnitude in km/s")
    ra: float = Field(..., description="Right ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")


@router.post("/solve", response_model=RadialVelocityResponse)
def solve_radial_velocity(request: RadialVelocityRequest) -> RadialVelocityResponse:
    """
    Solve for radial velocity given celestial coordinates and velocity vector.
    
    Args:
        request: Radial velocity request with coordinates and velocity components.
        
    Returns:
        Radial velocity response with computed velocities.
    """
    try:
        # Create input object
        input_data = RadialVelocityInput(
            ra=request.ra,
            dec=request.dec,
            vx=request.vx,
            vy=request.vy,
            vz=request.vz
        )
        
        # Validate the velocity vector
        input_data.validate_velocity_vector()
        
        # Create solver and compute
        solver = RadialVelocitySolver()
        result = solver.solve(input_data)
        
        return RadialVelocityResponse(
            radial_velocity=result.radial_velocity,
            transverse_velocity=result.transverse_velocity,
            total_velocity=result.total_velocity,
            ra=result.ra,
            dec=result.dec
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/health")
def health_check() -> dict:
    """Health check endpoint for the radial velocity solver service."""
    return {
        "status": "healthy",
        "service": "radial_velocity_solver"
    }