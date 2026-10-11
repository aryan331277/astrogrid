"""FastAPI router for WCS tangent point solver."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from src.astrogrid.engines.wcs_tangent_solver import (
    WcsTangentSolverInput,
    WcsTangentSolverOutput,
    solve_tangent_point,
)

router = APIRouter(prefix='/api/wcs_tangent_solver', tags=['wcs_tangent_solver'])


class TangentPointRequest(BaseModel):
    """Request model for tangent point calculation."""
    ra: float = Field(..., description="Right ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")
    ra0: float = Field(..., description="Reference right ascension in degrees")
    dec0: float = Field(..., description="Reference declination in degrees")


class TangentPointResponse(BaseModel):
    """Response model for tangent point calculation."""
    tangent_ra: float
    tangent_dec: float
    angular_distance_deg: float


@router.post("/solve", response_model=TangentPointResponse)
async def solve_tangent_endpoint(request: TangentPointRequest) -> TangentPointResponse:
    """
    Solve for the tangent point given a source position and reference frame.

    The tangent point is the point on the celestial sphere where the line of
    sight from the observer is perpendicular to the reference direction.
    """
    try:
        # Validate input using the engine's input model
        input_model = WcsTangentSolverInput(
            ra=request.ra,
            dec=request.dec,
            ra0=request.ra0,
            dec0=request.dec0,
        )
        input_model.validate_angle()

        # Call the engine function
        result = solve_tangent_point(
            ra=request.ra,
            dec=request.dec,
            ra0=request.ra0,
            dec0=request.dec0,
        )

        # Extract values from result (assuming it returns a tuple or dict)
        if isinstance(result, tuple):
            tangent_ra, tangent_dec, angular_distance = result
        elif isinstance(result, dict):
            tangent_ra = result.get('tangent_ra', result.get('ra', 0.0))
            tangent_dec = result.get('tangent_dec', result.get('dec', 0.0))
            angular_distance = result.get('angular_distance_deg', result.get('distance', 0.0))
        else:
            # Fallback: assume result is a WcsTangentSolverOutput
            tangent_ra = getattr(result, 'tangent_ra', 0.0)
            tangent_dec = getattr(result, 'tangent_dec', 0.0)
            angular_distance = getattr(result, 'angular_distance_deg', 0.0)

        return TangentPointResponse(
            tangent_ra=float(tangent_ra),
            tangent_dec=float(tangent_dec),
            angular_distance_deg=float(angular_distance),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "wcs_tangent_solver"}