"""
Service layer for Stokes Polarimetry Engine.
Handles data lake synchronization, Kafka stream simulation, and caching.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StokesData(BaseModel):
    """Pydantic model for Stokes parameter data."""
    timestamp: float
    frequency_hz: float
    i: float  # Total intensity
    q: float  # Linear polarization (horizontal-vertical)
    u: float  # Linear polarization (45-degree)
    v: float  # Circular polarization
    source_id: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncStatus(BaseModel):
    """Status of data lake synchronization."""
    last_sync_time: Optional[float] = None
    records_synced: int = 0
    status: str = "idle"
    error: Optional[str] = None


class CacheEntry(BaseModel):
    """Cache entry for processed Stokes data."""
    key: str
    data: StokesData
    cached_at: float
    ttl_seconds: float = 300.0


class StokesPolarimetryService:
    """
    Service for managing Stokes polarimetry data streams,
    data lake synchronization, and caching.
    """

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._sync_status = SyncStatus()
        self._sync_lock = threading.Lock()
        self._stream_buffer: deque = deque(maxlen=1000)
        self._stream_lock = threading.Lock()
        self._running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._process_callback: Optional[Callable[[StokesData], Any]] = None

    def set_process_callback(self, callback: Callable[[StokesData], Any]) -> None:
        """Set the callback function for processing data streams."""
        self._process_callback = callback

    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is still valid."""
        return (time.time() - entry.cached_at) < entry.ttl_seconds

    def get_from_cache(self, key: str) -> Optional[StokesData]:
        """Retrieve data from cache if valid."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and self._is_cache_valid(entry):
                return entry.data
            elif entry:
                del self._cache[key]
        return None

    def put_to_cache(self, key: str, data: StokesData, ttl_seconds: float = 300.0) -> None:
        """Store data in cache."""
        with self._cache_lock:
            self._cache[key] = CacheEntry(
                key=key,
                data=data,
                cached_at=time.time(),
                ttl_seconds=ttl_seconds
            )

    def clear_cache(self) -> int:
        """Clear all cache entries. Returns number of entries removed."""
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            valid = sum(1 for e in self._cache.values() if self._is_cache_valid(e))
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid,
                "expired_entries": len(self._cache) - valid
            }

    def _simulate_kafka_produce(self, data: StokesData) -> bool:
        """
        Simulate producing a message to a Kafka stream.
        In production, this would use a Kafka client.
        """
        try:
            # Simulate network latency
            time.sleep(0.001)
            # In a real implementation, we'd serialize and send to Kafka
            logger.debug(f"Produced to Kafka: {data.source_id} at {data.timestamp}")
            return True
        except Exception as e:
            logger.error(f"Kafka produce failed: {e}")
            return False

    def _simulate_kafka_consume(self) -> Optional[StokesData]:
        """
        Simulate consuming a message from a Kafka stream.
        In production, this would poll from a Kafka consumer.
        """
        # For simulation, return None to indicate no new messages
        # In a real implementation, this would block or poll
        return None

    def _sync_to_data_lake(self, data: StokesData) -> bool:
        """
        Synchronize data to the data lake.
        In production, this would write to S3, GCS, or similar.
        """
        try:
            # Simulate data lake write
            time.sleep(0.002)
            with self._sync_lock:
                self._sync_status.records_synced += 1
                self._sync_status.last_sync_time = time.time()
                self._sync_status.status = "synced"
            logger.debug(f"Synced to data lake: {data.source_id}")
            return True
        except Exception as e:
            with self._sync_lock:
                self._sync_status.status = "error"
                self._sync_status.error = str(e)
            logger.error(f"Data lake sync failed: {e}")
            return False

    def process_data_stream(self, data: StokesData) -> Dict[str, Any]:
        """
        Process a single Stokes data point through the pipeline:
        1. Cache lookup
        2. Engine processing (if callback set)
        3. Kafka produce
        4. Data lake sync
        5. Cache store
        """
        key = f"{data.source_id}_{data.timestamp}_{data.frequency_hz}"

        # Check cache first
        cached = self.get_from_cache(key)
        if cached:
            return {"status": "cached", "data": cached, "key": key}

        # Process through engine if callback is set
        processed_result = None
        if self._process_callback:
            try:
                processed_result = self._process_callback(data)
            except Exception as e:
                logger.error(f"Engine processing failed: {e}")
                return {"status": "error", "error": str(e)}

        # Produce to Kafka stream
        kafka_ok = self._simulate_kafka_produce(data)

        # Sync to data lake
        sync_ok = self._sync_to_data_lake(data)

        # Store in cache
        self.put_to_cache(key, data)

        # Add to stream buffer
        with self._stream_lock:
            self._stream_buffer.append(data)

        return {
            "status": "processed",
            "data": data,
            "key": key,
            "kafka_produced": kafka_ok,
            "data_lake_synced": sync_ok,
            "engine_result": processed_result
        }

    def get_stream_buffer(self, limit: int = 100) -> List[StokesData]:
        """Get recent data from the stream buffer."""
        with self._stream_lock:
            return list(self._stream_buffer)[-limit:]

    def get_sync_status(self) -> SyncStatus:
        """Get current synchronization status."""
        with self._sync_lock:
            return self._sync_status.copy()

    def start_stream_consumer(self) -> None:
        """Start the background stream consumer thread."""
        if self._running:
            return

        self._running = True

        def _consumer_loop():
            while self._running:
                try:
                    data = self._simulate_kafka_consume()
                    if data:
                        self.process_data_stream(data)
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Stream consumer error: {e}")
                    time.sleep(1.0)

        self._stream_thread = threading.Thread(target=_consumer_loop, daemon=True)
        self._stream_thread.start()
        logger.info("Stream consumer started")

    def stop_stream_consumer(self) -> None:
        """Stop the background stream consumer thread."""
        self._running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None
        logger.info("Stream consumer stopped")

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the service."""
        return {
            "status": "healthy",
            "stream_consumer_running": self._running,
            "cache_stats