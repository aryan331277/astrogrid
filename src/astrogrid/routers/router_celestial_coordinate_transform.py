"""FastAPI router for celestial coordinate transformation endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.astrogrid.engines.celestial_coordinate_transform import (
    CelestialCoordinate,
    TransformResult,
)

router = APIRouter(prefix='/api/celestial_coordinate_transform', tags=['celestial_coordinate_transform'])


class CoordinateInput(BaseModel):
    """Input celestial coordinate in any supported system."""
    ra_hours: Optional[float] = Field(None, description="Right ascension in hours (0-24)")
    dec_degrees: Optional[float] = Field(None, description="Declination in degrees (-90 to 90)")
    galactic_l: Optional[float] = Field(None, description="Galactic longitude in degrees (0-360)")
    galactic_b: Optional[float] = Field(None, description="Galactic latitude in degrees (-90 to 90)")
    epoch: float = Field(2000.0, description="Epoch of the coordinate (e.g., 2000.0 for J2000)")


class TransformRequest(BaseModel):
    """Request body for coordinate transformation."""
    source_system: str = Field("equatorial", description="Source coordinate system: equatorial, galactic")
    target_system: str = Field("equatorial", description="Target coordinate system: equatorial, galactic")
    coordinate: CoordinateInput
    apply_precession: bool = Field(False, description="Whether to apply precession correction")
    apply_nutation: bool = Field(False, description="Whether to apply nutation correction")
    observation_date: Optional[str] = Field(None, description="ISO 8601 date string for precession/nutation")


class TransformResponse(BaseModel):
    """Response containing the transformed coordinate."""
    ra_hours: Optional[float] = None
    dec_degrees: Optional[float] = None
    galactic_l: Optional[float] = None
    galactic_b: Optional[float] = None
    epoch: float
    system: str


@router.post("/transform", response_model=TransformResponse)
async def transform_coordinate(request: TransformRequest) -> TransformResponse:
    """Transform a celestial coordinate from one system to another."""
    coord = request.coordinate

    # Validate input based on source system
    if request.source_system == "equatorial":
        if coord.ra_hours is None or coord.dec_degrees is None:
            raise HTTPException(status_code=400, detail="ra_hours and dec_degrees required for equatorial source")
        if not (0 <= coord.ra_hours < 24):
            raise HTTPException(status_code=400, detail="ra_hours must be in [0, 24)")
        if not (-90 <= coord.dec_degrees <= 90):
            raise HTTPException(status_code=400, detail="dec_degrees must be in [-90, 90]")
    elif request.source_system == "galactic":
        if coord.galactic_l is None or coord.galactic_b is None:
            raise HTTPException(status_code=400, detail="galactic_l and galactic_b required for galactic source")
        if not (0 <= coord.galactic_l < 360):
            raise HTTPException(status_code=400, detail="galactic_l must be in [0, 360)")
        if not (-90 <= coord.galactic_b <= 90):
            raise HTTPException(status_code=400, detail="galactic_b must be in [-90, 90]")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported source system: {request.source_system}")

    # Validate target system
    if request.target_system not in ("equatorial", "galactic"):
        raise HTTPException(status_code=400, detail=f"Unsupported target system: {request.target_system}")

    # Parse observation date if provided
    obs_dt = None
    if request.observation_date:
        try:
            obs_dt = datetime.fromisoformat(request.observation_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid observation_date format, use ISO 8601")

    # Build the source CelestialCoordinate
    if request.source_system == "equatorial":
        source_coord = CelestialCoordinate(
            ra_hours=coord.ra_hours,
            dec_degrees=coord.dec_degrees,
            epoch=coord.epoch,
        )
    else:
        source_coord = CelestialCoordinate(
            galactic_l=coord.galactic_l,
            galactic_b=coord.galactic_b,
            epoch=coord.epoch,
        )

    # Perform transformation
    try:
        result: TransformResult = source_coord.transform_to(
            target_system=request.target_system,
            apply_precession=request.apply_precession,
            apply_nutation=request.apply_nutation,
            observation_date=obs_dt,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(exc)}")

    # Build response
    response = TransformResponse(
        epoch=result.epoch,
        system=request.target_system,
    )
    if request.target_system == "equatorial":
        response.ra_hours = result.ra_hours
        response.dec_degrees = result.dec_degrees
    else:
        response.galactic_l = result.galactic_l
        response.galactic_b = result.galactic_b

    return response


@router.get("/systems")
async def list_supported_systems() -> dict:
    """Return the list of supported coordinate systems."""
    return {
        "systems": [
            {"name": "equatorial", "description": "Equatorial (RA/Dec) coordinate system"},
            {"name": "galactic", "description": "Galactic (l/b) coordinate system"},
        ]
    }