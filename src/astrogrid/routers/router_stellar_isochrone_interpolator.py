"""FastAPI router for stellar isochrone interpolation endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.astrogrid.engines.stellar_isochrone_interpolator import (
    IsochroneParameterSpace,
    StellarObservables,
    StellarIsochroneInterpolator,
)

router = APIRouter(prefix='/api/stellar_isochrone_interpolator', tags=['stellar_isochrone_interpolator'])


class IsochroneFitRequest(BaseModel):
    """Request body for fitting an isochrone interpolator."""
    parameter_space: dict = Field(..., description="Parameter space definition (e.g., age, metallicity ranges)")
    observables: List[dict] = Field(..., description="List of stellar observables for fitting")


class IsochroneInterpolateRequest(BaseModel):
    """Request body for interpolating stellar parameters."""
    age: float = Field(..., description="Stellar age in years")
    metallicity: float = Field(..., description="Metallicity [Fe/H]")
    mass: float = Field(..., description="Stellar mass in solar masses")


class IsochroneInterpolateResponse(BaseModel):
    """Response for isochrone interpolation."""
    teff: float = Field(..., description="Effective temperature in K")
    luminosity: float = Field(..., description="Luminosity in solar units")
    radius: float = Field(..., description="Radius in solar radii")
    surface_gravity: float = Field(..., description="Surface gravity log(g) in cgs")


class IsochroneBatchInterpolateRequest(BaseModel):
    """Request body for batch interpolation."""
    parameters: List[IsochroneInterpolateRequest] = Field(..., description="List of parameter sets")


class IsochroneBatchInterpolateResponse(BaseModel):
    """Response for batch interpolation."""
    results: List[IsochroneInterpolateResponse] = Field(..., description="List of interpolation results")


class IsochroneFitResponse(BaseModel):
    """Response for isochrone fitting."""
    success: bool = Field(..., description="Whether the fit was successful")
    message: str = Field(..., description="Status message")


@router.post("/fit", response_model=IsochroneFitResponse)
async def fit_isochrone(request: IsochroneFitRequest) -> IsochroneFitResponse:
    """Fit a stellar isochrone interpolator from given parameter space and observables."""
    try:
        param_space = IsochroneParameterSpace(**request.parameter_space)
        obs_list = [StellarObservables(**obs) for obs in request.observables]
        interpolator = StellarIsochroneInterpolator(param_space)
        interpolator.fit(obs_list)
        return IsochroneFitResponse(success=True, message="Isochrone interpolator fitted successfully")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fit isochrone: {str(e)}")


@router.post("/interpolate", response_model=IsochroneInterpolateResponse)
async def interpolate_isochrone(request: IsochroneInterpolateRequest) -> IsochroneInterpolateResponse:
    """Interpolate stellar observables for given parameters."""
    try:
        interpolator = StellarIsochroneInterpolator(IsochroneParameterSpace())
        result = interpolator.interpolate(request.age, request.metallicity, request.mass)
        return IsochroneInterpolateResponse(
            teff=result.get('teff', 0.0),
            luminosity=result.get('luminosity', 0.0),
            radius=result.get('radius', 0.0),
            surface_gravity=result.get('surface_gravity', 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to interpolate isochrone: {str(e)}")


@router.post("/interpolate_batch", response_model=IsochroneBatchInterpolateResponse)
async def interpolate_batch_isochrone(request: IsochroneBatchInterpolateRequest) -> IsochroneBatchInterpolateResponse:
    """Batch interpolate stellar observables for multiple parameter sets."""
    try:
        interpolator = StellarIsochroneInterpolator(IsochroneParameterSpace())
        params = [(p.age, p.metallicity, p.mass) for p in request.parameters]
        results = interpolator.interpolate_batch(params)
        responses = [
            IsochroneInterpolateResponse(
                teff=r.get('teff', 0.0),
                luminosity=r.get('luminosity', 0.0),
                radius=r.get('radius', 0.0),
                surface_gravity=r.get('surface_gravity', 0.0),
            )
            for r in results
        ]
        return IsochroneBatchInterpolateResponse(results=responses)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to batch interpolate isochrone: {str(e)}")