"""Service for supernova Arnett model: data lake sync, Kafka stream, and caching."""

import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArnettModelRequest(BaseModel):
    """Request model for Arnett model computation."""
    ejecta_mass: float = Field(..., gt=0, description="Ejecta mass in solar masses")
    nickel_mass: float = Field(..., ge=0, description="Nickel-56 mass in solar masses")
    kinetic_energy: float = Field(..., gt=0, description="Kinetic energy in ergs")
    opacity: float = Field(..., gt=0, description="Opacity in cm^2/g")
    time_points: List[float] = Field(default_factory=lambda: [1.0, 10.0, 100.0], description="Time points in days")


class ArnettModelResponse(BaseModel):
    """Response model for Arnett model computation."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)


class SupernovaArnettModelService:
    """Service for supernova Arnett model with caching and data lake sync."""

    def __init__(self, cache_size: int = 1000, sync_interval: float = 60.0):
        self._cache: OrderedDict[str, ArnettModelResponse] = OrderedDict()
        self._cache_size = cache_size
        self._sync_interval = sync_interval
        self._last_sync: float = 0.0
        self._pending_sync: List[Dict[str, Any]] = []
        self._lock = None  # Simplified; in production use threading.Lock

    def _compute_cache_key(self, request: ArnettModelRequest) -> str:
        """Compute a deterministic cache key from request parameters."""
        data = {
            "ejecta_mass": request.ejecta_mass,
            "nickel_mass": request.nickel_mass,
            "kinetic_energy": request.kinetic_energy,
            "opacity": request.opacity,
            "time_points": request.time_points,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[ArnettModelResponse]:
        """Retrieve result from cache if present."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _put_to_cache(self, key: str, response: ArnettModelResponse) -> None:
        """Store result in cache, evicting oldest if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = response
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _compute_model(self, request: ArnettModelRequest) -> Dict[str, Any]:
        """Compute Arnett model using standard physics formulas."""
        M_ej = request.ejecta_mass * 1.989e33  # solar mass to grams
        M_Ni = request.nickel_mass * 1.989e33
        E_kin = request.kinetic_energy
        kappa = request.opacity

        # Arnett's rule: peak luminosity and time
        # t_peak = (2 * kappa * M_ej / (13.8 * E_kin)) ** 0.5
        t_peak = np.sqrt(2.0 * kappa * M_ej / (13.8 * E_kin))

        # Peak luminosity
        # L_peak = 1.3e43 * (M_Ni / M_sun) * (E_kin / 1e51)^(-0.5) erg/s
        M_sun = 1.989e33
        L_peak = 1.3e43 * (M_Ni / M_sun) * (E_kin / 1e51) ** (-0.5)

        # Compute luminosity at each time point
        luminosities = []
        for t in request.time_points:
            if t <= 0:
                luminosities.append(0.0)
            elif t < t_peak:
                # Rising phase: L ~ t^2
                L = L_peak * (t / t_peak) ** 2
            else:
                # Declining phase: L ~ exp(-t/tau)
                tau = 1.48 * t_peak  # approximate decay timescale
                L = L_peak * np.exp(-(t - t_peak) / tau)
            luminosities.append(float(L))

        return {
            "t_peak_days": float(t_peak),
            "L_peak_erg_s": float(L_peak),
            "luminosities_erg_s": luminosities,
            "time_points_days": request.time_points,
        }

    def _sync_to_data_lake(self) -> None:
        """Synchronize pending results to the data lake (simulated)."""
        if not self._pending_sync:
            return
        logger.info(f"Syncing {len(self._pending_sync)} records to data lake")
        # In production, this would write to a data lake via httpx
        self._pending_sync.clear()
        self._last_sync = time.time()

    def _maybe_sync(self) -> None:
        """Trigger sync if interval has elapsed."""
        if time.time() - self._last_sync >= self._sync_interval:
            self._sync_to_data_lake()

    def compute(self, request: ArnettModelRequest) -> ArnettModelResponse:
        """Compute Arnett model with caching and data lake sync."""
        key = self._compute_cache_key(request)

        # Check cache
        cached = self._get_from_cache(key)
        if cached is not None:
            return ArnettModelResponse(
                success=True,
                result=cached.result,
                cache_hit=True,
            )

        # Compute
        try:
            result = self._compute_model(request)
            response = ArnettModelResponse(
                success=True,
                result=result,
                cache_hit=False,
            )
            self._put_to_cache(key, response)

            # Queue for data lake sync
            self._pending_sync.append({
                "key": key,
                "request": request.dict(),
                "result": result,
                "timestamp": time.time(),
            })
            self._maybe_sync()

            return response
        except Exception as e:
            logger.error(f"Error computing Arnett model: {e}")
            return ArnettModelResponse(
                success=False,
                error=str(e),
                cache_hit=False,
            )

    def get_cached(self, request: ArnettModelRequest) -> Optional[ArnettModelResponse]:
        """Retrieve cached result without computing."""
        key = self._compute_cache_key(request)
        return self._get_from_cache(key)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "pending_sync": len(self._pending_sync),
            "last_sync": self._last_sync,
        }