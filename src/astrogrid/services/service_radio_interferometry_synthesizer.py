"""
Radio Interferometry Synthesizer Service
Data lake synchronization, Kafka stream, and caching service for radio_interferometry_synthesizer.
"""

import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    PENDING = "pending"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VisibilityRecord:
    """Represents a single visibility data record."""
    timestamp: float
    baseline_id: str
    frequency: float
    uvw: np.ndarray
    complex_value: complex
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "baseline_id": self.baseline_id,
            "frequency": self.frequency,
            "uvw": self.uvw.tolist(),
            "complex_value": [self.complex_value.real, self.complex_value.imag],
            "weight": self.weight,
        }


class SynthesisRequest(BaseModel):
    """Request model for synthesis operation."""
    dataset_id: str
    frequency_range: List[float] = Field(..., min_length=2)
    beam_pattern: str = "gaussian"
    resolution: int = 256
    weight_scheme: str = "natural"


class SynthesisResponse(BaseModel):
    """Response model for synthesis operation."""
    dataset_id: str
    status: SyncStatus
    image_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = {}


class RadioInterferometrySynthesizerService:
    """
    Service for radio interferometry synthesis with data lake sync,
    Kafka streaming, and caching capabilities.
    """

    def __init__(self, data_lake_url: str = "http://datalake:8080",
                 kafka_broker: str = "kafka:9092",
                 cache_ttl: int = 3600):
        self.data_lake_url = data_lake_url
        self.kafka_broker = kafka_broker
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._sync_status: Dict[str, SyncStatus] = {}
        self._http_client = httpx.AsyncClient(timeout=30.0)

    def _get_cache_key(self, dataset_id: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for a synthesis request."""
        key_str = f"{dataset_id}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[cache_key]) < self.cache_ttl

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None

    def _store_in_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store data in cache with timestamp."""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = time.time()

    def _evict_expired_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        expired_keys = [
            key for key, ts in self._cache_timestamps.items()
            if (now - ts) >= self.cache_ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    async def _fetch_visibilities(self, dataset_id: str) -> List[VisibilityRecord]:
        """Fetch visibility data from the data lake."""
        url = f"{self.data_lake_url}/datasets/{dataset_id}/visibilities"
        try:
            response = await self._http_client.get(url)
            response.raise_for_status()
            data = response.json()
            records = []
            for item in data.get("visibilities", []):
                record = VisibilityRecord(
                    timestamp=item["timestamp"],
                    baseline_id=item["baseline_id"],
                    frequency=item["frequency"],
                    uvw=np.array(item["uvw"]),
                    complex_value=complex(*item["complex_value"]),
                    weight=item["weight"],
                )
                records.append(record)
            return records
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch visibilities for {dataset_id}: {e}")
            raise

    async def _publish_to_kafka(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish a message to Kafka stream."""
        url = f"{self.kafka_broker}/topics/{topic}/records"
        try:
            await self._http_client.post(url, json=message)
        except httpx.HTTPError as e:
            logger.warning(f"Failed to publish to Kafka topic {topic}: {e}")

    def _compute_synthesis(self, records: List[VisibilityRecord],
                           request: SynthesisRequest) -> np.ndarray:
        """
        Compute the synthesized image from visibility data.
        Uses a simplified CLEAN-like algorithm.
        """
        resolution = request.resolution
        image = np.zeros((resolution, resolution), dtype=np.float64)

        if len(records) == 0:
            return image

        uv_coords = np.array([r.uvw for r in records])
        weights = np.array([r.weight for r in records])
        complex_values = np.array([r.complex_value for r in records])

        max_uv = np.max(np.abs(uv_coords), axis=0)
        if max_uv[0] == 0 or max_uv[1] == 0:
            return image

        for i in range(resolution):
            for j in range(resolution):
                u = (i - resolution / 2) / max_uv[0]
                v = (j - resolution / 2) / max_uv[1]
                phase = 2 * np.pi * (u * uv_coords[:, 0] + v * uv_coords[:, 1])
                weighted_sum = np.sum(weights * complex_values * np.exp(1j * phase))
                image[i, j] = np.abs(weighted_sum)

        if request.beam_pattern == "gaussian":
            sigma = resolution / 8.0
            y, x = np.mgrid[0:resolution, 0:resolution]
            center = resolution / 2.0
            beam = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
            image = image * beam

        return image

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """Main synthesis endpoint."""
        self._evict_expired_cache()

        params = {
            "frequency_range": request.frequency_range,
            "beam_pattern": request.beam_pattern,
            "resolution": request.resolution,
            "weight_scheme": request.weight_scheme,
        }
        cache_key = self._get_cache_key(request.dataset_id, params)

        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info(f"Cache hit for {request.dataset_id}")
            return SynthesisResponse(
                dataset_id=request.dataset_id,
                status=SyncStatus.COMPLETED,
                image_data=cached["image_data"],
                metadata=cached["metadata"],
            )

        self._sync_status[request.dataset_id] = SyncStatus.SYNCING
        try:
            records = await self._fetch_visibilities(request.dataset_id)
        except Exception as e:
            self._sync_status[request.dataset_id] = SyncStatus.FAILED
            raise HTTPException(status_code=500, detail=f"Data fetch failed: {str(e)}")

        image = self._compute_synthesis(records, request)

        metadata = {
            "num_visibilities": len(records),
            "frequency_range": request.frequency_range,
            "beam_pattern": request.beam_pattern,
            "resolution": request.resolution,
            "weight_scheme": request.weight_scheme,
            "computed_at": time.time(),
        }

        self._store_in_cache(cache_key, {"image_data": image, "metadata": metadata})