"""Service for stellar isochrone interpolation with data lake sync, Kafka stream, and caching."""

import time
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx

logger = logging.getLogger(__name__)


class IsochroneRequest(BaseModel):
    """Request model for isochrone interpolation."""
    age: float = Field(..., gt=0, description="Stellar age in Gyr")
    metallicity: float = Field(..., ge=-2.0, le=0.5, description="Metallicity [Fe/H]")
    mass_range: tuple = Field((0.1, 100.0), description="Mass range in solar masses")
    n_points: int = Field(100, ge=10, le=1000, description="Number of interpolation points")


class IsochroneResponse(BaseModel):
    """Response model for isochrone interpolation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)


class CacheEntry(BaseModel):
    """Cache entry for isochrone results."""
    key: str
    data: Dict[str, Any]
    created_at: float
    ttl: int = 3600


class StellarIsochroneService:
    """Service for stellar isochrone interpolation with caching and data lake sync."""

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        self._interpolator = None
        self._kafka_producer = None
        self._data_lake_client = None
        self._http_client = httpx.AsyncClient(timeout=30.0)

    def _get_cache_key(self, request: IsochroneRequest) -> str:
        """Generate a unique cache key for the request."""
        key_data = {
            "age": request.age,
            "metallicity": request.metallicity,
            "mass_range": list(request.mass_range),
            "n_points": request.n_points
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache if valid."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.created_at < entry.ttl:
                self._cache.move_to_end(key)
                return entry.data
            else:
                del self._cache[key]
        return None

    def _put_to_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in cache, evicting oldest if necessary."""
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = CacheEntry(
            key=key,
            data=data,
            created_at=time.time(),
            ttl=self._cache_ttl
        )

    async def _initialize_interpolator(self):
        """Initialize the isochrone interpolator engine."""
        if self._interpolator is None:
            try:
                from src.astrogrid.engine.stellar_isochrone_interpolator import StellarIsochroneInterpolator
                self._interpolator = StellarIsochroneInterpolator()
                logger.info("Stellar isochrone interpolator initialized")
            except ImportError as e:
                logger.error(f"Failed to import interpolator: {e}")
                raise

    async def _sync_to_data_lake(self, request: IsochroneRequest, result: Dict[str, Any]) -> None:
        """Synchronize results to the data lake."""
        try:
            record = {
                "request": request.dict(),
                "result": result,
                "timestamp": time.time(),
                "service": "stellar_isochrone_interpolator"
            }
            # Simulate data lake write
            logger.debug(f"Syncing to data lake: {record['request']['age']} Gyr, {record['request']['metallicity']} [Fe/H]")
        except Exception as e:
            logger.error(f"Data lake sync failed: {e}")

    async def _publish_to_kafka(self, request: IsochroneRequest, result: Dict[str, Any]) -> None:
        """Publish results to Kafka stream."""
        try:
            message = {
                "topic": "stellar_isochrone_results",
                "key": self._get_cache_key(request),
                "value": {
                    "request": request.dict(),
                    "result": result,
                    "timestamp": time.time()
                }
            }
            # Simulate Kafka publish
            logger.debug(f"Published to Kafka: {message['key']}")
        except Exception as e:
            logger.error(f"Kafka publish failed: {e}")

    async def interpolate(self, request: IsochroneRequest) -> IsochroneResponse:
        """Perform isochrone interpolation with caching."""
        cache_key = self._get_cache_key(request)

        # Check cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return IsochroneResponse(
                success=True,
                data=cached_data,
                cache_hit=True
            )

        # Initialize interpolator if needed
        await self._initialize_interpolator()

        try:
            # Perform interpolation
            masses = np.linspace(request.mass_range[0], request.mass_range[1], request.n_points)
            result = self._interpolator.interpolate(
                age=request.age,
                metallicity=request.metallicity,
                masses=masses
            )

            response_data = {
                "masses": result.get("masses", masses.tolist()),
                "luminosities": result.get("luminosities", []),
                "temperatures": result.get("temperatures", []),
                "radii": result.get("radii", []),
                "ages": result.get("ages", [request.age])
            }

            # Cache the result
            self._put_to_cache(cache_key, response_data)

            # Background tasks for sync
            await self._sync_to_data_lake(request, response_data)
            await self._publish_to_kafka(request, response_data)

            return IsochroneResponse(
                success=True,
                data=response_data,
                cache_hit=False
            )

        except Exception as e:
            logger.error(f"Isochrone interpolation failed: {e}")
            return IsochroneResponse(
                success=False,
                error=str(e),
                cache_hit=False
            )

    async def interpolate_batch(self, requests: List[IsochroneRequest]) -> List[IsochroneResponse]:
        """Perform batch isochrone interpolation."""
        responses = []
        for request in requests:
            response = await self.interpolate(request)
            responses.append(response)
        return responses

    async def close(self):
        """Close service resources."""
        await self._http_client.aclose()
        logger.info("Stellar isochrone service closed")


# Create service instance
service = StellarIsochroneService()

# FastAPI app
app = FastAPI(title="Stellar Isochrone Interpolator Service")


@app.post("/api/v1/isochrone/interpolate", response_model=IsochroneResponse)
async def interpolate_endpoint(request: IsochroneRequest, background_tasks: BackgroundTasks):
    """Endpoint for isochrone interpolation."""
    response = await service.interpolate(request)
    return response


@app.post("/api/v1/isochrone/interpolate-batch", response_model=List[IsochroneResponse])
async def interpolate_batch_endpoint(requests: List[IsochroneRequest]):
    """Endpoint for batch isochrone interpolation."""
    responses = await service.interpolate_batch(requests)
    return responses


@app.get("/api/v1/isochrone/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stellar_isochrone_interpolator"}


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await service.close()