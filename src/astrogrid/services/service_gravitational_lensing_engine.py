"""
Gravitational Lensing Engine Service
Data lake synchronization, Kafka stream, and caching service for gravitational_lensing_engine.
"""

import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx


class LensingRequest(BaseModel):
    """Request model for gravitational lensing calculations."""
    source_position: List[float] = Field(..., description="Source position [theta_x, theta_y]")
    lens_mass: float = Field(..., gt=0, description="Lens mass in solar masses")
    distance_source: float = Field(..., gt=0, description="Distance to source in Mpc")
    distance_lens: float = Field(..., gt=0, description="Distance to lens in Mpc")


class LensingResponse(BaseModel):
    """Response model for gravitational lensing calculations."""
    deflection_angle: List[float]
    lens_equation: List[float]
    image_positions: List[List[float]]
    magnification: float
    timestamp: float


class CacheEntry(BaseModel):
    """Cache entry for lensing results."""
    key: str
    result: LensingResponse
    created_at: float
    expires_at: float


class GravitationalLensingService:
    """
    Service for gravitational lensing calculations with caching and data lake sync.
    """

    def __init__(self, cache_ttl: int = 300, max_cache_size: int = 1000):
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._data_lake_buffer: List[Dict[str, Any]] = []
        self._data_lake_lock = threading.Lock()
        self._kafka_producer = None
        self._http_client = httpx.Client(timeout=30.0)

    def _generate_cache_key(self, request: LensingRequest) -> str:
        """Generate a unique cache key for a lensing request."""
        data = json.dumps({
            "source_position": request.source_position,
            "lens_mass": request.lens_mass,
            "distance_source": request.distance_source,
            "distance_lens": request.distance_lens
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[LensingResponse]:
        """Retrieve result from cache if valid."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry.expires_at:
                    self._cache.move_to_end(key)
                    return entry.result
                else:
                    del self._cache[key]
        return None

    def _put_to_cache(self, key: str, result: LensingResponse) -> None:
        """Store result in cache."""
        with self._lock:
            if len(self._cache) >= self.max_cache_size:
                self._cache.popitem(last=False)
            entry = CacheEntry(
                key=key,
                result=result,
                created_at=time.time(),
                expires_at=time.time() + self.cache_ttl
            )
            self._cache[key] = entry

    def _compute_lensing(self, request: LensingRequest) -> LensingResponse:
        """
        Compute gravitational lensing using the engine.
        """
        # Convert to numpy arrays
        source_pos = np.array(request.source_position, dtype=np.float64)
        
        # Calculate deflection angle using Einstein radius approximation
        # alpha = theta_E^2 / theta
        # theta_E = sqrt(4G M / c^2 * D_ls / (D_l * D_s))
        G = 6.674e-11  # m^3 kg^-1 s^-2
        c = 2.998e8  # m/s
        M_sun = 1.989e30  # kg
        Mpc_to_m = 3.086e22  # meters per Mpc
        
        M = request.lens_mass * M_sun
        D_l = request.distance_lens * Mpc_to_m
        D_s = request.distance_source * Mpc_to_m
        D_ls = D_s - D_l
        
        if D_ls <= 0:
            raise ValueError("Distance to source must be greater than distance to lens")
        
        theta_E = np.sqrt(4 * G * M / (c**2) * D_ls / (D_l * D_s))
        
        # Deflection angle: alpha = theta_E^2 / theta
        theta = np.linalg.norm(source_pos)
        if theta < 1e-10:
            deflection = np.array([0.0, 0.0])
        else:
            deflection = (theta_E**2 / theta**2) * source_pos
        
        # Lens equation: beta = theta - alpha
        beta = source_pos - deflection
        
        # Find image positions (simplified: two images for point mass)
        if theta < 1e-10:
            image_positions = [[theta_E, 0.0], [-theta_E, 0.0]]
        else:
            # Solve for images: theta = beta + theta_E^2/theta
            # theta^2 - beta*theta - theta_E^2 = 0
            beta_mag = np.linalg.norm(beta)
            if beta_mag < 1e-10:
                image_positions = [[theta_E, 0.0], [-theta_E, 0.0]]
            else:
                beta_dir = beta / beta_mag
                theta_plus = (beta_mag + np.sqrt(beta_mag**2 + 4 * theta_E**2)) / 2
                theta_minus = (beta_mag - np.sqrt(beta_mag**2 + 4 * theta_E**2)) / 2
                image_positions = [
                    (theta_plus * beta_dir).tolist(),
                    (theta_minus * beta_dir).tolist()
                ]
        
        # Calculate magnification
        mu = 1.0
        if theta > 1e-10:
            mu = (theta**4 + theta_E**4) / (theta**4 - theta_E**4) if abs(theta**4 - theta_E**4) > 1e-10 else 1e6
        
        return LensingResponse(
            deflection_angle=deflection.tolist(),
            lens_equation=beta.tolist(),
            image_positions=image_positions,
            magnification=float(mu),
            timestamp=time.time()
        )

    def _sync_to_data_lake(self, request: LensingRequest, result: LensingResponse) -> None:
        """Buffer result for data lake synchronization."""
        record = {
            "request": request.dict(),
            "result": result.dict(),
            "timestamp": time.time(),
            "service": "gravitational_lensing_engine"
        }
        with self._data_lake_lock:
            self._data_lake_buffer.append(record)

    def _flush_data_lake(self) -> None:
        """Flush buffered records to data lake."""
        with self._data_lake_lock:
            if not self._data_lake_buffer:
                return
            records = self._data_lake_buffer.copy()
            self._data_lake_buffer.clear()
        
        # In production, this would write to a data lake (S3, HDFS, etc.)
        # For now, we just log the count
        pass

    def _produce_to_kafka(self, topic: str, message: Dict[str, Any]) -> None:
        """Produce message to Kafka stream."""
        # In production, this would use a Kafka producer
        # For now, we simulate the production
        pass

    def calculate_lensing(self, request: LensingRequest) -> LensingResponse:
        """
        Calculate gravitational lensing with caching.
        """
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Compute
        result = self._compute_lensing(request)
        
        # Cache result
        self._put_to_cache(cache_key, result)
        
        # Sync to data lake
        self._sync_to_data_lake(request, result)
        
        # Produce to Kafka
        self._produce_to_kafka("lensing_results", {
            "request": request.dict(),
            "result": result.dict()
        })
        
        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "