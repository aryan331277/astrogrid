"""FastAPI router for the transit BLS detector module."""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..engines.transit_bls_detector import (
    BLSInputData,
    BLSResult,
    _compute_bls_likelihood,
    _optimize_phase,
)

router = APIRouter(prefix='/api/transit_bls_detector', tags=['transit_bls_detector'])


class BLSRequest(BaseModel):
    """Request body for BLS transit detection."""
    time: List[float] = Field(..., description="Time stamps (e.g., days)")
    flux: List[float] = Field(..., description="Flux measurements")
    flux_err: List[float] = Field(..., description="Flux uncertainties")
    period_min: float = Field(0.5, description="Minimum period to search (days)")
    period_max: float = Field(30.0, description="Maximum period to search (days)")
    duration: float = Field(0.05, description="Transit duration (days)")
    n_periods: int = Field(1000, description="Number of period grid points")


class BLSResponse(BaseModel):
    """Response body for BLS transit detection."""
    best_period: float
    best_phase: float
    best_depth: float
    best_likelihood: float
    periods: List[float]
    likelihoods: List[float]


@router.post("/detect", response_model=BLSResponse)
def detect_transit_bls(req: BLSRequest) -> BLSResponse:
    """Run a Box Least Squares search for a transit signal."""
    if len(req.time) != len(req.flux) or len(req.time) != len(req.flux_err):
        raise HTTPException(status_code=400, detail="time, flux, and flux_err must have the same length")
    if len(req.time) < 10:
        raise HTTPException(status_code=400, detail="At least 10 data points required")
    if req.period_min <= 0 or req.period_max <= req.period_min:
        raise HTTPException(status_code=400, detail="Invalid period range")
    if req.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")

    time = np.array(req.time, dtype=np.float64)
    flux = np.array(req.flux, dtype=np.float64)
    flux_err = np.array(req.flux_err, dtype=np.float64)

    # Build period grid
    periods = np.linspace(req.period_min, req.period_max, req.n_periods)

    # Compute likelihood for each period
    likelihoods = np.zeros(len(periods))
    best_idx = 0
    best_lik = -np.inf

    for i, p in enumerate(periods):
        # Optimize phase for this period
        try:
            phase, depth, lik = _optimize_phase(time, flux, flux_err, p, req.duration)
        except Exception:
            phase, depth, lik = 0.0, 0.0, -np.inf

        likelihoods[i] = lik
        if lik > best_lik:
            best_lik = lik
            best_idx = i

    best_period = float(periods[best_idx])
    best_phase, best_depth, best_likelihood = _optimize_phase(
        time, flux, flux_err, best_period, req.duration
    )

    return BLSResponse(
        best_period=best_period,
        best_phase=float(best_phase),
        best_depth=float(best_depth),
        best_likelihood=float(best_likelihood),
        periods=[float(p) for p in periods],
        likelihoods=[float(l) for l in likelihoods],
    )


@router.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "module": "transit_bls_detector"}