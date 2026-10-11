"""FastAPI router for light curve detrending operations."""

from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..engines.light_curve_detrender import calculate_astronomical_metric

router = APIRouter(prefix='/api/light_curve_detrender', tags=['light_curve_detrender'])


class DetrendRequest(BaseModel):
    """Request model for light curve detrending."""
    times: List[float] = Field(..., description="Time values of the light curve")
    fluxes: List[float] = Field(..., description="Flux values of the light curve")
    window_size: int = Field(10, ge=1, description="Window size for detrending")
    method: str = Field("median", description="Detrending method: 'median' or 'mean'")


class DetrendResponse(BaseModel):
    """Response model for light curve detrending."""
    times: List[float]
    original_fluxes: List[float]
    detrended_fluxes: List[float]
    trend: List[float]
    metric: float


@router.post("/detrend", response_model=DetrendResponse)
async def detrend_light_curve(request: DetrendRequest) -> DetrendResponse:
    """
    Detrend a light curve by removing low-frequency variations.
    
    Args:
        request: DetrendRequest containing times, fluxes, window_size, and method
        
    Returns:
        DetrendResponse with original and detrended fluxes, trend, and metric
    """
    times = np.array(request.times)
    fluxes = np.array(request.fluxes)
    
    if len(times) != len(fluxes):
        raise HTTPException(status_code=400, detail="Times and fluxes must have the same length")
    
    if len(times) < 2:
        raise HTTPException(status_code=400, detail="At least 2 data points required")
    
    if request.window_size > len(times):
        raise HTTPException(status_code=400, detail="Window size cannot exceed number of data points")
    
    # Compute trend using rolling window
    half_window = request.window_size // 2
    trend = np.zeros_like(fluxes)
    
    for i in range(len(fluxes)):
        start = max(0, i - half_window)
        end = min(len(fluxes), i + half_window + 1)
        
        if request.method == "median":
            trend[i] = np.median(fluxes[start:end])
        elif request.method == "mean":
            trend[i] = np.mean(fluxes[start:end])
        else:
            raise HTTPException(status_code=400, detail="Method must be 'median' or 'mean'")
    
    # Detrend by dividing out the trend (preserves relative variations)
    detrended_fluxes = fluxes / trend
    
    # Calculate astronomical metric
    try:
        metric = calculate_astronomical_metric(times, detrended_fluxes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate metric: {str(e)}")
    
    return DetrendResponse(
        times=request.times,
        original_fluxes=request.fluxes,
        detrended_fluxes=detrended_fluxes.tolist(),
        trend=trend.tolist(),
        metric=float(metric)
    )


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for the light curve detrender service."""
    return {"status": "healthy", "service": "light_curve_detrender"}