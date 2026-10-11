"""FastAPI router for cosmic_ray_lacosmic module."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

from src.astrogrid.engines.cosmic_ray_lacosmic import calculate_astronomical_metric

router = APIRouter(prefix='/api/cosmic_ray_lacosmic', tags=['cosmic_ray_lacosmic'])


class CosmicRayLacosmicRequest(BaseModel):
    """Request model for cosmic ray LA-COSMIC calculations."""
    energy_gev: float = Field(..., gt=0, description="Energy in GeV")
    mass_amu: float = Field(..., gt=0, description="Mass in atomic mass units")
    charge: int = Field(..., ge=1, description="Charge number")
    ra_deg: float = Field(..., ge=0, le=360, description="Right ascension in degrees")
    dec_deg: float = Field(..., ge=-90, le=90, description="Declination in degrees")
    galactic_longitude: Optional[float] = Field(None, ge=0, le=360, description="Galactic longitude in degrees")
    galactic_latitude: Optional[float] = Field(None, ge=-90, le=90, description="Galactic latitude in degrees")
    observation_time: Optional[str] = Field(None, description="ISO 8601 observation timestamp")


class CosmicRayLacosmicResponse(BaseModel):
    """Response model for cosmic ray LA-COSMIC calculations."""
    metric_value: float
    metric_name: str
    input_echo: CosmicRayLacosmicRequest
    status: str = "success"


@router.post("/calculate", response_model=CosmicRayLacosmicResponse)
async def calculate_endpoint(request: CosmicRayLacosmicRequest) -> CosmicRayLacosmicResponse:
    """
    Calculate astronomical metric for cosmic ray LA-COSMIC data.
    
    Args:
        request: Request containing cosmic ray parameters
        
    Returns:
        Response with calculated metric value
    """
    try:
        # Prepare parameters for the engine function
        params = {
            "energy_gev": request.energy_gev,
            "mass_amu": request.mass_amu,
            "charge": request.charge,
            "ra_deg": request.ra_deg,
            "dec_deg": request.dec_deg,
        }
        
        if request.galactic_longitude is not None:
            params["galactic_longitude"] = request.galactic_longitude
        if request.galactic_latitude is not None:
            params["galactic_latitude"] = request.galactic_latitude
        if request.observation_time is not None:
            params["observation_time"] = request.observation_time
        
        # Call the engine function
        result = calculate_astronomical_metric(**params)
        
        # Extract metric value and name from result
        if isinstance(result, dict):
            metric_value = float(result.get("value", result.get("metric", 0.0)))
            metric_name = str(result.get("name", result.get("metric_name", "astronomical_metric")))
        elif isinstance(result, (int, float, np.floating, np.integer)):
            metric_value = float(result)
            metric_name = "astronomical_metric"
        else:
            metric_value = float(result) if result is not None else 0.0
            metric_name = "astronomical_metric"
        
        return CosmicRayLacosmicResponse(
            metric_value=metric_value,
            metric_name=metric_name,
            input_echo=request
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for cosmic_ray_lacosmic router."""
    return {
        "status": "healthy",
        "module": "cosmic_ray_lacosmic",
        "router": "router_cosmic_ray_lacosmic"
    }