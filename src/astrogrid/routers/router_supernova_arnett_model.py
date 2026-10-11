"""FastAPI router for the Supernova Arnett Model."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from src.astrogrid.engines.supernova_arnett_model import (
    ArnettModelParams,
    SupernovaArnettModel,
)

router = APIRouter(prefix='/api/supernova_arnett_model', tags=['supernova_arnett_model'])


class ArnettModelRequest(BaseModel):
    """Request body for computing the Arnett model."""
    ejecta_mass: float = Field(..., gt=0, description="Ejecta mass in solar masses")
    nickel_mass: float = Field(..., ge=0, description="Nickel-56 mass in solar masses")
    kinetic_energy: float = Field(..., gt=0, description="Kinetic energy in ergs")
    opacity: float = Field(..., gt=0, description="Opacity in cm^2/g")
    time_days: list[float] = Field(..., description="Times in days at which to evaluate luminosity")


class ArnettModelResponse(BaseModel):
    """Response containing the computed Arnett model results."""
    times_days: list[float]
    luminosities_erg_s: list[float]
    peak_luminosity_erg_s: float
    peak_time_days: float


@router.post("/compute", response_model=ArnettModelResponse)
def compute_arnett_model(request: ArnettModelRequest) -> ArnettModelResponse:
    """Compute the Arnett model light curve for the given parameters."""
    try:
        params = ArnettModelParams(
            ejecta_mass=request.ejecta_mass,
            nickel_mass=request.nickel_mass,
            kinetic_energy=request.kinetic_energy,
            opacity=request.opacity,
        )
        model = SupernovaArnettModel(params)
        result = model.get_result()

        times = np.array(request.time_days)
        luminosities = []
        for t in times:
            lum = model.get_luminosity_at_t(float(t))
            luminosities.append(float(lum))

        # Find peak luminosity and time
        if luminosities:
            peak_idx = int(np.argmax(luminosities))
            peak_lum = luminosities[peak_idx]
            peak_time = float(times[peak_idx])
        else:
            peak_lum = 0.0
            peak_time = 0.0

        return ArnettModelResponse(
            times_days=[float(t) for t in times],
            luminosities_erg_s=luminosities,
            peak_luminosity_erg_s=peak_lum,
            peak_time_days=peak_time,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to compute Arnett model: {str(e)}")


@router.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "module": "supernova_arnett_model"}