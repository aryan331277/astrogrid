"""FastAPI router for Doppler redshift estimation."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.astrogrid.engines.doppler_redshift_estimator import (
    DopplerRedshiftEstimator,
    SpectralLine,
    PhotometricBand,
    RedshiftEstimate,
)

router = APIRouter(prefix='/api/doppler_redshift_estimator', tags=['doppler_redshift_estimator'])


class SpectralLineInput(BaseModel):
    """Input model for a spectral line."""
    rest_wavelength: float = Field(..., description="Rest wavelength in Angstroms")
    observed_wavelength: float = Field(..., description="Observed wavelength in Angstroms")
    line_id: Optional[str] = Field(None, description="Identifier for the spectral line")


class PhotometricBandInput(BaseModel):
    """Input model for a photometric band."""
    band_name: str = Field(..., description="Name of the photometric band")
    flux: float = Field(..., description="Measured flux in the band")
    flux_error: float = Field(0.0, description="Flux measurement error")


class RedshiftEstimateRequest(BaseModel):
    """Request model for redshift estimation."""
    spectral_lines: List[SpectralLineInput] = Field(
        default_factory=list, description="List of spectral lines for estimation"
    )
    photometric_bands: List[PhotometricBandInput] = Field(
        default_factory=list, description="List of photometric bands for estimation"
    )
    method: str = Field(
        "spectral",
        description="Estimation method: 'spectral', 'photometric', or 'combined'"
    )


class RedshiftEstimateResponse(BaseModel):
    """Response model for redshift estimation."""
    redshift: float = Field(..., description="Estimated redshift value")
    redshift_error: float = Field(0.0, description="Redshift uncertainty")
    method_used: str = Field(..., description="Method used for estimation")
    confidence: float = Field(0.0, description="Confidence score (0-1)")
    details: dict = Field(default_factory=dict, description="Additional details")


@router.post("/estimate", response_model=RedshiftEstimateResponse)
async def estimate_redshift(request: RedshiftEstimateRequest) -> RedshiftEstimateResponse:
    """
    Estimate redshift from spectral lines and/or photometric bands.
    
    Args:
        request: RedshiftEstimateRequest containing spectral lines, photometric bands, and method
        
    Returns:
        RedshiftEstimateResponse with estimated redshift and metadata
    """
    try:
        estimator = DopplerRedshiftEstimator()
        
        # Convert input models to engine objects
        spectral_lines = []
        for line in request.spectral_lines:
            sl = SpectralLine(
                rest_wavelength=line.rest_wavelength,
                observed_wavelength=line.observed_wavelength,
                line_id=line.line_id
            )
            spectral_lines.append(sl)
        
        photometric_bands = []
        for band in request.photometric_bands:
            pb = PhotometricBand(
                band_name=band.band_name,
                flux=band.flux,
                flux_error=band.flux_error
            )
            photometric_bands.append(pb)
        
        # Perform estimation
        result = estimator.estimate_from_spectral(
            spectral_lines=spectral_lines,
            photometric_bands=photometric_bands,
            method=request.method
        )
        
        return RedshiftEstimateResponse(
            redshift=result.redshift,
            redshift_error=result.redshift_error,
            method_used=result.method_used,
            confidence=result.confidence,
            details=result.details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redshift estimation failed: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "doppler_redshift_estimator"}