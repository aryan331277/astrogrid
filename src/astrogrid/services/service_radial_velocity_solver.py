"""
Radial Velocity Solver Service
Data lake synchronization, Kafka stream, and caching service for radial_velocity_solver.
"""

import time
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RadialVelocityRequest(BaseModel):
    """Request model for radial velocity calculation."""
    ra: float = Field(..., description="Right ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")
    velocity_x: float = Field(..., description="Velocity component in X (km/s)")
    velocity_y: float = Field(..., description="Velocity component in Y (km/s)")
    velocity_z: float = Field(..., description="Velocity component in Z (km/s)")
    observer_ra: Optional[float] = Field(None, description="Observer RA in degrees")
    observer_dec: Optional[float] = Field(None, description="Observer Dec in degrees")


class RadialVelocityResponse(BaseModel):
    """Response model for radial velocity calculation."""
    radial_velocity: float = Field(..., description="Radial velocity in km/s")
    transverse_velocity: float = Field(..., description="Transverse velocity in km/s")
    total_velocity: float = Field(..., description="Total velocity in km/s")
    timestamp: float = Field(..., description="Unix timestamp of calculation")


class CacheEntry(BaseModel):
    """Cache entry for radial velocity results."""
    key: str
    result: RadialVelocityResponse
    created_at: float
    expires_at: float


class RadialVelocityCache:
    """LRU cache for radial velocity results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def _make_key(self, request: RadialVelocityRequest) -> str:
        """Generate a cache key from request parameters."""
        data = {
            "ra": request.ra,
            "dec": request.dec,
            "vx": request.velocity_x,
            "vy": request.velocity_y,
            "vz": request.velocity_z,
            "ora": request.observer_ra,
            "odec": request.observer_dec,
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get(self, request: RadialVelocityRequest) -> Optional[RadialVelocityResponse]:
        """Get cached result if available and not expired."""
        key = self._make_key(request)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry.expires_at:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return entry.result
            else:
                # Expired, remove
                del self._cache[key]
        return None
    
    def put(self, request: RadialVelocityRequest, result: RadialVelocityResponse) -> None:
        """Store result in cache."""
        key = self._make_key(request)
        now = time.time()
        entry = CacheEntry(
            key=key,
            result=result,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)
        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()


class KafkaStreamSimulator:
    """Simulated Kafka stream for publishing radial velocity events."""
    
    def __init__(self, topic: str = "radial-velocity-events"):
        self.topic = topic
        self._messages: List[Dict[str, Any]] = []
    
    def publish(self, event: Dict[str, Any]) -> None:
        """Publish an event to the simulated Kafka stream."""
        event["timestamp"] = time.time()
        event["topic"] = self.topic
        self._messages.append(event)
        logger.debug("Published to %s: %s", self.topic, event)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieve all published messages."""
        return list(self._messages)
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()


class DataLakeSyncService:
    """Simulated data lake synchronization service."""
    
    def __init__(self):
        self._synced_records: List[Dict[str, Any]] = []
        self._last_sync_time: float = 0.0
    
    def sync(self, record: Dict[str, Any]) -> None:
        """Synchronize a record to the data lake."""
        self._synced_records.append(record)
        self._last_sync_time = time.time()
        logger.debug("Synced record to data lake: %s", record)
    
    def get_synced_records(self) -> List[Dict[str, Any]]:
        """Retrieve all synced records."""
        return list(self._synced_records)
    
    def get_last_sync_time(self) -> float:
        """Get the last sync timestamp."""
        return self._last_sync_time


class RadialVelocitySolverService:
    """Main service for radial velocity calculations with caching, streaming, and sync."""
    
    def __init__(self):
        self.cache = RadialVelocityCache(max_size=1000, ttl_seconds=300.0)
        self.kafka_stream = KafkaStreamSimulator(topic="radial-velocity-events")
        self.data_lake = DataLakeSyncService()
    
    def _deg_to_rad(self, degrees: float) -> float:
        """Convert degrees to radians."""
        return degrees * np.pi / 180.0
    
    def _compute_radial_velocity(self, request: RadialVelocityRequest) -> RadialVelocityResponse:
        """Compute radial velocity from input parameters."""
        ra_rad = self._deg_to_rad(request.ra)
        dec_rad = self._deg_to_rad(request.dec)
        
        # Unit vector pointing to the source
        src_x = np.cos(dec_rad) * np.cos(ra_rad)
        src_y = np.cos(dec_rad) * np.sin(ra_rad)
        src_z = np.sin(dec_rad)
        
        # Velocity vector
        vel = np.array([request.velocity_x, request.velocity_y, request.velocity_z])
        
        # Radial velocity is the dot product of velocity and source direction
        radial_velocity = float(np.dot(vel, np.array([src_x, src_y, src_z])))
        
        # Total velocity magnitude
        total_velocity = float(np.linalg.norm(vel))
        
        # Transverse velocity
        transverse_velocity = float(np.sqrt(max(0.0, total_velocity ** 2 - radial_velocity ** 2)))
        
        return RadialVelocityResponse(
            radial_velocity=radial_velocity,
            transverse_velocity=transverse_velocity,
            total_velocity=total_velocity,
            timestamp=time.time(),
        )
    
    def solve(self, request: RadialVelocityRequest) -> RadialVelocityResponse:
        """Solve radial velocity with caching, streaming, and data lake sync."""
        # Check cache first
        cached = self.cache.get(request)
        if cached is not None:
            logger.debug("Cache hit for radial velocity request")
            return cached
        
        # Compute result
        result = self._compute_radial_velocity(request)
        
        # Store in cache
        self.cache.put(request, result)
        
        # Publish to Kafka stream
        event = {
            "ra": request.ra,
            "dec": request.dec,
            "velocity_x": request.velocity_x,
            "velocity_y": request.velocity_y,
            "velocity_z": request.velocity_z,
            "radial_velocity": result.radial_velocity,
            "transverse_velocity": result.transverse_velocity,
            "total_velocity": result.total_velocity,
        }
        self.kafka_stream.publish(event)
        
        # Sync to data lake
        self.data_lake.sync(event)
        
        return result


# Create the FastAPI application
app = FastAPI(title="Radial Velocity Solver Service", version="1.0.0")
service = RadialVelocitySolverService()


@app.post("/radial-velocity/solve", response_model=RadialVelocityResponse)
async def solve_radial_velocity(request: RadialVelocityRequest) -> RadialVelocityResponse:
    """Solve radial velocity for the given input."""
    try:
        return service.solve(request)
    except Exception as e:
        logger.error