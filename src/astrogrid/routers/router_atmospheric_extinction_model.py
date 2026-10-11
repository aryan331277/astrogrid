"""FastAPI router for atmospheric extinction model endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.astrogrid.engines.atmospheric_extinction_model import (
    AtmosphericExtinctionEngine,
    ExtinctionModelType,
    ExtinctionParameters,
    ExtinctionResult,
)

router = APIRouter(prefix='/api/atmospheric_extinction_model', tags=['atmospheric_extinction_model'])


class ExtinctionRequest(BaseModel):
    """Request body for computing atmospheric extinction."""
    altitude_deg: float = Field(..., ge=0.0, le=90.0, description="Altitude angle in degrees")
    wavelength_um: float = Field(..., gt=0.0, description="Wavelength in micrometers")
    model_type: str = Field(default="standard", description="Extinction model type")
    airmass: Optional[float] = Field(None, gt=0.0, description="Optional pre-computed airmass")


class ExtinctionResponse(BaseModel):
    """Response body for extinction computation."""
    airmass: float
    extinction_coefficient: float
    total_extinction: float
    model_type: str
    wavelength_um: float
    altitude_deg: float


@router.post("/compute", response_model=ExtinctionResponse)
async def compute_extinction(request: ExtinctionRequest) -> ExtinctionResponse:
    """Compute atmospheric extinction for given parameters."""
    try:
        model_type = ExtinctionModelType(request.model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")

    engine = AtmosphericExtinctionEngine()
    params = ExtinctionParameters(
        altitude_deg=request.altitude_deg,
        wavelength_um=request.wavelength_um,
        model_type=model_type,
    )

    if request.airmass is not None:
        airmass = request.airmass
    else:
        airmass = engine.compute_airmass(request.altitude_deg)

    result = engine.compute_extinction(params, airmass=airmass)

    return ExtinctionResponse(
        airmass=result.airmass,
        extinction_coefficient=result.extinction_coefficient,
        total_extinction=result.total_extinction,
        model_type=result.model_type.value if hasattr(result.model_type, 'value') else str(result.model_type),
        wavelength_um=request.wavelength_um,
        altitude_deg=request.altitude_deg,
    )


@router.get("/airmass")
async def get_airmass(altitude_deg: float) -> dict:
    """Compute airmass for a given altitude angle."""
    if not (0.0 <= altitude_deg <= 90.0):
        raise HTTPException(status_code=400, detail="Altitude must be between 0 and 90 degrees")
    engine = AtmosphericExtinctionEngine()
    airmass = engine.compute_airmass(altitude_deg)
    return {"altitude_deg": altitude_deg, "airmass": airmass}