"""Data lake synchronization, Kafka stream, and caching service for doppler_redshift_estimator."""

import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx

logger = logging.getLogger(__name__)

# Simulated Kafka producer/consumer using in-memory structures
class InMemoryKafka:
    """In-memory simulation of Kafka for stream processing."""
    
    def __init__(self):
        self.topics: Dict[str, List[Dict[str, Any]]] = {}
    
    def produce(self, topic: str, message: Dict[str, Any]):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(message)
        logger.debug(f"Produced message to topic {topic}")
    
    def consume(self, topic: str, max_messages: int = 100) -> List[Dict[str, Any]]:
        if topic not in self.topics:
            return []
        messages = self.topics[topic][:max_messages]
        self.topics[topic] = self.topics[topic][max_messages:]
        return messages


class RedshiftEstimateRequest(BaseModel):
    """Request model for redshift estimation."""
    observed_wavelengths: List[float] = Field(..., description="Observed wavelengths in Angstroms")
    rest_wavelengths: List[float] = Field(..., description="Rest frame wavelengths in Angstroms")
    source_id: str = Field(..., description="Unique identifier for the source")


class RedshiftEstimateResponse(BaseModel):
    """Response model for redshift estimation."""
    source_id: str
    redshift: float
    velocity: float
    confidence: float
    timestamp: float


class DopplerRedshiftService:
    """Service for Doppler redshift estimation with caching and streaming."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache: OrderedDict[str, RedshiftEstimateResponse] = OrderedDict()
        self.cache_size = cache_size
        self.kafka = InMemoryKafka()
        self._lock = False  # Simple flag for thread safety simulation
    
    def _get_cache_key(self, request: RedshiftEstimateRequest) -> str:
        """Generate a cache key from the request."""
        data = json.dumps({
            "observed": request.observed_wavelengths,
            "rest": request.rest_wavelengths,
            "source": request.source_id
        }, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
    
    def _cache_get(self, key: str) -> Optional[RedshiftEstimateResponse]:
        """Get item from cache, moving it to end for LRU."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def _cache_set(self, key: str, value: RedshiftEstimateResponse):
        """Set item in cache, evicting oldest if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
    
    def estimate_redshift(self, request: RedshiftEstimateRequest) -> RedshiftEstimateResponse:
        """Estimate redshift from spectral line shifts."""
        cache_key = self._get_cache_key(request)
        cached = self._cache_get(cache_key)
        if cached:
            logger.debug(f"Cache hit for source {request.source_id}")
            return cached
        
        # Validate inputs
        if len(request.observed_wavelengths) != len(request.rest_wavelengths):
            raise ValueError("Observed and rest wavelength arrays must have same length")
        
        if len(request.observed_wavelengths) == 0:
            raise ValueError("Wavelength arrays cannot be empty")
        
        # Calculate redshift for each line pair
        redshifts = []
        for obs, rest in zip(request.observed_wavelengths, request.rest_wavelengths):
            if rest <= 0:
                raise ValueError("Rest wavelength must be positive")
            z = (obs - rest) / rest
            redshifts.append(z)
        
        redshifts = np.array(redshifts)
        
        # Use median for robustness against outliers
        median_z = float(np.median(redshifts))
        
        # Calculate confidence based on variance
        if len(redshifts) > 1:
            std_z = float(np.std(redshifts))
            confidence = max(0.0, min(1.0, 1.0 - std_z / (abs(median_z) + 1e-10)))
        else:
            confidence = 0.5
        
        # Calculate velocity using relativistic Doppler formula
        c = 299792458.0  # speed of light in m/s
        beta = (median_z ** 2 + 2 * median_z) / (median_z ** 2 + 2 * median_z + 1)
        velocity = beta * c
        
        response = RedshiftEstimateResponse(
            source_id=request.source_id,
            redshift=median_z,
            velocity=velocity,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self._cache_set(cache_key, response)
        
        # Produce to Kafka stream
        self.kafka.produce("redshift_estimates", {
            "source_id": request.source_id,
            "redshift": median_z,
            "velocity": velocity,
            "confidence": confidence,
            "timestamp": response.timestamp
        })
        
        logger.info(f"Estimated redshift {median_z:.6f} for source {request.source_id}")
        return response
    
    def sync_to_data_lake(self, batch_size: int = 100) -> int:
        """Synchronize cached estimates to data lake (simulated)."""
        messages = self.kafka.consume("redshift_estimates", max_messages=batch_size)
        if not messages:
            return 0
        
        # Simulate writing to data lake
        logger.info(f"Synchronizing {len(messages)} records to data lake")
        return len(messages)


# FastAPI application
app = FastAPI(title="Doppler Redshift Estimator Service")
service = DopplerRedshiftService()


@app.post("/estimate", response_model=RedshiftEstimateResponse)
async def estimate_redshift(request: RedshiftEstimateRequest):
    """Estimate redshift from spectral line shifts."""
    try:
        return service.estimate_redshift(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error estimating redshift: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/sync")
async def sync_data_lake(batch_size: int = 100):
    """Synchronize estimates to data lake."""
    count = service.sync_to_data_lake(batch_size)
    return {"synchronized": count}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "cache_size": len(service.cache)}