"""FastAPI router for aperture photometry endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..engines.aperture_photometry import (
    ApertureType,
    ApertureConfig,
    ApertureResult,
    _create_circular_mask,
)

router = APIRouter(prefix='/api/aperture_photometry', tags=['aperture_photometry'])


class ApertureRequest(BaseModel):
    """Request model for aperture photometry."""
    image: List[List[float]] = Field(..., description="2D image data as nested list")
    center_x: float = Field(..., description="X coordinate of aperture center")
    center_y: float = Field(..., description="Y coordinate of aperture center")
    radius: float = Field(..., gt=0, description="Aperture radius in pixels")
    aperture_type: str = Field("circular", description="Type of aperture: circular or annular")
    inner_radius: Optional[float] = Field(None, ge=0, description="Inner radius for annular aperture")


class ApertureResponse(BaseModel):
    """Response model for aperture photometry."""
    flux: float
    error: float
    n_pixels: int
    center_x: float
    center_y: float
    radius: float
    aperture_type: str


@router.post("/measure", response_model=ApertureResponse)
def measure_aperture(request: ApertureRequest) -> ApertureResponse:
    """Perform aperture photometry on the provided image."""
    try:
        image = request.image
        if not image or not image[0]:
            raise HTTPException(status_code=400, detail="Image data is empty")

        height = len(image)
        width = len(image[0])

        # Validate center coordinates
        if request.center_x < 0 or request.center_x >= width:
            raise HTTPException(status_code=400, detail="center_x out of bounds")
        if request.center_y < 0 or request.center_y >= height:
            raise HTTPException(status_code=400, detail="center_y out of bounds")

        # Create mask based on aperture type
        if request.aperture_type == "circular":
            mask = _create_circular_mask((height, width), request.center_x, request.center_y, request.radius)
        elif request.aperture_type == "annular":
            if request.inner_radius is None:
                raise HTTPException(status_code=400, detail="inner_radius required for annular aperture")
            if request.inner_radius >= request.radius:
                raise HTTPException(status_code=400, detail="inner_radius must be less than radius")
            outer_mask = _create_circular_mask((height, width), request.center_x, request.center_y, request.radius)
            inner_mask = _create_circular_mask((height, width), request.center_x, request.center_y, request.inner_radius)
            mask = [row[i] and not inner_row[i] for row, inner_row in zip(outer_mask, inner_mask)]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown aperture type: {request.aperture_type}")

        # Compute flux and error
        flux = 0.0
        n_pixels = 0
        for y in range(height):
            for x in range(width):
                if mask[y][x]:
                    flux += image[y][x]
                    n_pixels += 1

        if n_pixels == 0:
            raise HTTPException(status_code=400, detail="No pixels within aperture")

        # Estimate error as standard deviation of pixel values within aperture
        pixel_values = []
        for y in range(height):
            for x in range(width):
                if mask[y][x]:
                    pixel_values.append(image[y][x])

        mean_val = sum(pixel_values) / len(pixel_values)
        variance = sum((v - mean_val) ** 2 for v in pixel_values) / len(pixel_values)
        error = (variance ** 0.5) * (n_pixels ** 0.5)

        return ApertureResponse(
            flux=flux,
            error=error,
            n_pixels=n_pixels,
            center_x=request.center_x,
            center_y=request.center_y,
            radius=request.radius,
            aperture_type=request.aperture_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Aperture photometry failed: {str(e)}")


@router.get("/aperture-types")
def get_aperture_types() -> dict:
    """Return available aperture types."""
    return {
        "types": [
            {"name": "circular", "description": "Circular aperture"},
            {"name": "annular", "description": "Annular (ring) aperture"},
        ]
    }