"""FastAPI router for PSF Fitting Engine."""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..engines.psf_fitting_engine import PSFFittingEngine, PSFFitResult, PSFModelType

router = APIRouter(prefix='/api/psf_fitting_engine', tags=['psf_fitting_engine'])


class PSFFitRequest(BaseModel):
    """Request model for PSF fitting."""
    image: List[List[float]] = Field(..., description="2D image data as nested list")
    model_type: str = Field("gaussian", description="PSF model type: gaussian, mog, kron")
    center: Optional[List[float]] = Field(None, description="Initial center [x, y]")
    fwhm: Optional[float] = Field(None, description="Initial FWHM estimate")


class PSFFitResponse(BaseModel):
    """Response model for PSF fitting."""
    success: bool
    model_type: str
    center: List[float]
    fwhm: float
    flux: float
    residuals: float
    message: str = ""


@router.post("/fit", response_model=PSFFitResponse)
async def fit_psf(request: PSFFitRequest) -> PSFFitResponse:
    """Fit a PSF model to the provided image data."""
    try:
        import numpy as np

        image = np.array(request.image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError("Image must be 2D")

        engine = PSFFittingEngine()

        # Determine model type
        model_type_str = request.model_type.lower()
        if model_type_str == "gaussian":
            model_type = PSFModelType.GAUSSIAN
        elif model_type_str == "mog":
            model_type = PSFModelType.MOG
        elif model_type_str == "kron":
            model_type = PSFModelType.KRON
        else:
            raise ValueError(f"Unsupported model type: {request.model_type}")

        # Prepare initial parameters
        center = request.center if request.center else [image.shape[1] / 2.0, image.shape[0] / 2.0]
        fwhm = request.fwhm if request.fwhm else 3.0

        # Perform fitting
        result: PSFFitResult = engine.fit(
            image=image,
            model_type=model_type,
            center=center,
            fwhm=fwhm
        )

        return PSFFitResponse(
            success=True,
            model_type=model_type_str,
            center=[result.center[0], result.center[1]],
            fwhm=result.fwhm,
            flux=result.flux,
            residuals=result.residuals,
            message="PSF fit completed successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PSF fitting failed: {str(e)}")


@router.get("/models")
async def list_models() -> dict:
    """List available PSF model types."""
    return {
        "models": [
            {"name": "gaussian", "description": "Single Gaussian PSF model"},
            {"name": "mog", "description": "Mixture of Gaussians PSF model"},
            {"name": "kron", "description": "Kron PSF model"}
        ]
    }


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "psf_fitting_engine"}