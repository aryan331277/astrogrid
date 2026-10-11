"""
Service for Fast Radio Burst Detector: data lake sync, Kafka stream, and caching.
"""
import time
import json
import logging
from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import httpx

logger = logging.getLogger(__name__)

# Simulated Kafka producer/consumer using in-memory structures
class SimulatedKafkaProducer:
    def __init__(self, topic: str):
        self.topic = topic
        self.messages = deque(maxlen=1000)
    
    def send(self, message: Dict[str, Any]):
        self.messages.append({
            'topic': self.topic,
            'message': message,
            'timestamp': time.time()
        })
        logger.debug(f"Sent message to {self.topic}: {message}")

class SimulatedKafkaConsumer:
    def __init__(self, topic: str):
        self.topic = topic
        self.messages = deque(maxlen=1000)
    
    def poll(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        if self.messages:
            return self.messages.popleft()
        return None

class FRBDetectionCache:
    """LRU cache for FRB detection results."""
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.order: deque = deque()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Dict[str, Any]):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.order.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.order.clear()

class DataLakeSyncService:
    """Handles synchronization with data lake."""
    def __init__(self, base_url: str = "http://datalake.local"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
    
    def upload_detection(self, detection_id: str, data: Dict[str, Any]) -> bool:
        try:
            response = self.client.post(
                f"{self.base_url}/api/v1/frb-detections/{detection_id}",
                json=data
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to upload detection {detection_id}: {e}")
            return False
    
    def fetch_detections(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/frb-detections",
                params={'start_time': start_time, 'end_time': end_time}
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Failed to fetch detections: {e}")
            return []
    
    def close(self):
        self.client.close()

class FRBDetectorService:
    """Main service for FRB detector with Kafka, caching, and data lake sync."""
    
    def __init__(
        self,
        kafka_topic: str = "frb-detections",
        cache_size: int = 1000,
        datalake_url: str = "http://datalake.local"
    ):
        self.kafka_producer = SimulatedKafkaProducer(kafka_topic)
        self.kafka_consumer = SimulatedKafkaConsumer(kafka_topic)
        self.cache = FRBDetectionCache(max_size=cache_size)
        self.data_lake = DataLakeSyncService(datalake_url)
        self._running = False
    
    def start(self):
        self._running = True
        logger.info("FRB Detector Service started")
    
    def stop(self):
        self._running = False
        self.data_lake.close()
        logger.info("FRB Detector Service stopped")
    
    def process_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Process a new FRB detection."""
        detection_id = detection.get('id', str(time.time()))
        
        # Check cache first
        cached = self.cache.get(detection_id)
        if cached:
            logger.debug(f"Cache hit for detection {detection_id}")
            return cached
        
        # Process and store
        result = {
            'id': detection_id,
            'timestamp': detection.get('timestamp', time.time()),
            'frequency_mhz': detection.get('frequency_mhz', 1400.0),
            'duration_ms': detection.get('duration_ms', 5.0),
            'snr': detection.get('snr', 10.0),
            'ra_deg': detection.get('ra_deg', 0.0),
            'dec_deg': detection.get('dec_deg', 0.0),
            'status': 'processed'
        }
        
        # Cache the result
        self.cache.put(detection_id, result)
        
        # Send to Kafka
        self.kafka_producer.send(result)
        
        # Upload to data lake
        self.data_lake.upload_detection(detection_id, result)
        
        return result
    
    def get_detection(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a detection by ID."""
        cached = self.cache.get(detection_id)
        if cached:
            return cached
        
        # Try fetching from data lake
        detections = self.data_lake.fetch_detections(0, time.time())
        for det in detections:
            if det.get('id') == detection_id:
                self.cache.put(detection_id, det)
                return det
        
        return None
    
    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent detections from cache."""
        items = list(self.cache.cache.values())
        items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return items[:limit]
    
    def poll_kafka(self) -> Optional[Dict[str, Any]]:
        """Poll for new messages from Kafka."""
        return self.kafka_consumer.poll(timeout=1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'cache_size': len(self.cache.cache),
            'cache_max_size': self.cache.max_size,
            'kafka_messages_pending': len(self.kafka_producer.messages),
            'running': self._running
        }

# Global service instance
_service_instance: Optional[FRBDetectorService] = None

def get_service() -> FRBDetectorService:
    global _service_instance
    if _service_instance is None:
        _service_instance = FRBDetectorService()
        _service_instance.start()
    return _service_instance

def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(title="FRB Detector Service", version="1.0.0")
    
    @app.on_event("startup")
    async def startup():
        get_service()
    
    @app.on_event("shutdown")
    async def shutdown():
        if _service_instance:
            _service_instance.stop()
    
    @app.post("/api/v1/frb-detections")
    async def create_detection(detection: Dict[str, Any]):
        service = get_service()
        result = service.process_detection(detection)
        return result
    
    @app.get("/api/v1/frb-detections/{detection_id}")
    async def get_detection(detection_id: str):
        service = get_service()
        result = service.get_detection(detection_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Detection not found")
        return result
    
    @app.get("/api/v1/frb-detections/recent")
    async def get_recent_detections(limit: int = 10):
        service = get_service()
        return service.get_recent_detections(limit)
    
    @app.get("/api/v1/stats")
    async def get_stats():
        service = get_service()
        return service.get_stats()
    
    return app

app = create_app()