"""FastAPI router for FITS header WCS parsing endpoints."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.astrogrid.engines.fits_header_wcs_parser import (
    WCSAxis,
    WCSAxisType,
    WCSInfo,
    parse_wcs_header,
    pixel_to_world,
    world_to_pixel,
)

router = APIRouter(prefix="/api/fits_header_wcs_parser", tags=["fits_header_wcs_parser"])


class HeaderRequest(BaseModel):
    """Request model for parsing a FITS header."""
    header: Dict[str, Any] = Field(..., description="FITS header as a dictionary")


class PixelToWorldRequest(BaseModel):
    """Request model for converting pixel coordinates to world coordinates."""
    header: Dict[str, Any] = Field(..., description="FITS header as a dictionary")
    pixel_coords: List[float] = Field(..., description="Pixel coordinates [x, y]")


class WorldToPixelRequest(BaseModel):
    """Request model for converting world coordinates to pixel coordinates."""
    header: Dict[str, Any] = Field(..., description="FITS header as a dictionary")
    world_coords: List[float] = Field(..., description="World coordinates [ra, dec]")


class WCSAxisResponse(BaseModel):
    """Response model for a single WCS axis."""
    index: int
    ctype: str
    axis_type: str
    crval: float
    crpix: float
    cdelt: float
    cunit: str


class WCSInfoResponse(BaseModel):
    """Response model for full WCS information."""
    naxis: int
    axes: List[WCSAxisResponse]
    raw_header: Dict[str, Any]


class CoordinateConversionResponse(BaseModel):
    """Response model for coordinate conversion."""
    input_coords: List[float]
    output_coords: List[float]
    success: bool


@router.post("/parse", response_model=WCSInfoResponse)
async def parse_wcs_endpoint(request: HeaderRequest) -> WCSInfoResponse:
    """Parse a FITS header and return WCS information."""
    try:
        wcs_info: WCSInfo = parse_wcs_header(request.header)
        axes: List[WCSAxisResponse] = []
        for axis in wcs_info.axes:
            axes.append(
                WCSAxisResponse(
                    index=axis.index,
                    ctype=axis.ctype,
                    axis_type=axis.axis_type.value if hasattr(axis.axis_type, "value") else str(axis.axis_type),
                    crval=axis.crval,
                    crpix=axis.crpix,
                    cdelt=axis.cdelt,
                    cunit=axis.cunit,
                )
            )
        return WCSInfoResponse(
            naxis=wcs_info.naxis,
            axes=axes,
            raw_header=request.header,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse WCS header: {str(e)}")


@router.post("/pixel-to-world", response_model=CoordinateConversionResponse)
async def pixel_to_world_endpoint(request: PixelToWorldRequest) -> CoordinateConversionResponse:
    """Convert pixel coordinates to world coordinates."""
    try:
        wcs_info: WCSInfo = parse_wcs_header(request.header)
        pixel_coords = np.array(request.pixel_coords, dtype=np.float64)
        world_coords = pixel_to_world(wcs_info, pixel_coords)
        return CoordinateConversionResponse(
            input_coords=request.pixel_coords,
            output_coords=[float(c) for c in world_coords],
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert pixel to world: {str(e)}")


@router.post("/world-to-pixel", response_model=CoordinateConversionResponse)
async def world_to_pixel_endpoint(request: WorldToPixelRequest) -> CoordinateConversionResponse:
    """Convert world coordinates to pixel coordinates."""
    try:
        wcs_info: WCSInfo = parse_wcs_header(request.header)
        world_coords = np.array(request.world_coords, dtype=np.float64)
        pixel_coords = world_to_pixel(wcs_info, world_coords)
        return CoordinateConversionResponse(
            input_coords=request.world_coords,
            output_coords=[float(c) for c in pixel_coords],
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert world to pixel: {str(e)}")