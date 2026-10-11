"""Service for NFW dark matter halo data lake synchronization, Kafka stream, and caching."""

import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
import numpy as np

# Simulated Kafka producer/consumer using in-memory structures
class SimulatedKafkaProducer:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()
    
    def send(self, topic: str, message: Dict[str, Any]):
        with self.lock:
            self.messages.append({
                'topic': topic,
                'message': message,
                'timestamp': time.time()
            })

class SimulatedKafkaConsumer:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()
    
    def consume(self, topic: str) -> List[Dict[str, Any]]:
        with self.lock:
            return [m for m in self.messages if m['topic'] == topic]

# In-memory cache
class HaloCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()

# Data lake sync service
class NFWDarkMatterHaloService:
    def __init__(self):
        self.cache = HaloCache()
        self.kafka_producer = SimulatedKafkaProducer()
        self.kafka_consumer = SimulatedKafkaConsumer()
        self.data_lake = {}
        self.lock = threading.Lock()
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a cache key from parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def compute_halo_density(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute NFW halo density with caching."""
        cache_key = self._generate_cache_key(params)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Compute using engine (simulated)
        result = self._compute_nfw_density(params)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        # Send to Kafka
        self.kafka_producer.send('nfw_halo_results', {
            'params': params,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def _compute_nfw_density(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute NFW density profile."""
        r = params.get('r', 1.0)
        r_s = params.get('r_s', 10.0)
        rho_0 = params.get('rho_0', 1e-22)
        
        # NFW density profile: rho(r) = rho_0 / ((r/r_s) * (1 + r/r_s)^2)
        x = r / r_s
        if x < 1e-10:
            rho = rho_0 / (x * (1 + x)**2) if x > 0 else float('inf')
        else:
            rho = rho_0 / (x * (1 + x)**2)
        
        return {
            'r': r,
            'r_s': r_s,
            'rho_0': rho_0,
            'density': rho,
            'timestamp': time.time()
        }
    
    def sync_to_data_lake(self, results: List[Dict[str, Any]]):
        """Synchronize results to data lake."""
        with self.lock:
            for result in results:
                key = self._generate_cache_key(result)
                self.data_lake[key] = result
    
    def get_from_data_lake(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from data lake."""
        with self.lock:
            return self.data_lake.get(key)

# FastAPI application
app = FastAPI(title="NFW Dark Matter Halo Service")
service = NFWDarkMatterHaloService()

class HaloRequest(BaseModel):
    r: float = Field(..., description="Radius in kpc")
    r_s: float = Field(..., description="Scale radius in kpc")
    rho_0: float = Field(..., description="Characteristic density in g/cm^3")

class HaloResponse(BaseModel):
    r: float
    r_s: float
    rho_0: float
    density: float
    timestamp: float

@app.post("/halo/density", response_model=HaloResponse)
async def compute_halo_density(request: HaloRequest):
    """Compute NFW halo density."""
    params = request.dict()
    result = service.compute_halo_density(params)
    return HaloResponse(**result)

@app.get("/halo/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return {
        'cache_size': len(service.cache.cache),
        'data_lake_size': len(service.data_lake)
    }

@app.post("/halo/cache/clear")
async def clear_cache():
    """Clear the cache."""
    service.cache.clear()
    return {'status': 'cleared'}

@app.get("/halo/data-lake/{key}")
async def get_data_lake_item(key: str):
    """Get item from data lake."""
    item = service.get_from_data_lake(key)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item