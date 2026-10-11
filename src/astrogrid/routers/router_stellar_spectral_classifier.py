"""FastAPI router for stellar spectral classification endpoints."""

from typing import List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.astrogrid.engines.stellar_spectral_classifier import (
    StellarSpectralClassifier,
    SpectralClassificationResult,
)

router = APIRouter(prefix='/api/stellar_spectral_classifier', tags=['stellar_spectral_classifier'])


class SpectrumRequest(BaseModel):
    """Request model for spectral classification."""
    wavelengths: List[float] = Field(..., description="Array of wavelength values in Angstroms")
    fluxes: List[float] = Field(..., description="Array of flux values corresponding to wavelengths")
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio of the spectrum")


class ClassificationResponse(BaseModel):
    """Response model for spectral classification."""
    spectral_type: str
    confidence: float
    temperature_k: float
    log_g: float
    metallicity: float
    details: dict


@router.post("/classify", response_model=ClassificationResponse)
async def classify_spectrum(request: SpectrumRequest) -> ClassificationResponse:
    """Classify a stellar spectrum and return its spectral type and parameters."""
    if len(request.wavelengths) != len(request.fluxes):
        raise HTTPException(status_code=400, detail="Wavelengths and fluxes must have the same length")
    if len(request.wavelengths) < 10:
        raise HTTPException(status_code=400, detail="Spectrum must have at least 10 data points")

    try:
        wavelengths = np.array(request.wavelengths, dtype=np.float64)
        fluxes = np.array(request.fluxes, dtype=np.float64)

        classifier = StellarSpectralClassifier()
        result: SpectralClassificationResult = classifier.classify(wavelengths, fluxes, snr=request.snr)

        return ClassificationResponse(
            spectral_type=result.spectral_type,
            confidence=result.confidence,
            temperature_k=result.temperature_k,
            log_g=result.log_g,
            metallicity=result.metallicity,
            details=result.details if hasattr(result, 'details') else {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for the stellar spectral classifier service."""
    return {"status": "healthy", "service": "stellar_spectral_classifier"}