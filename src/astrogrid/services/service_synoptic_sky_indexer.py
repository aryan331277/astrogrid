"""Service for synoptic sky indexer: data lake sync, Kafka stream, and caching."""

import time
import threading
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import numpy as np

# Simulated Kafka producer/consumer using standard library
class _KafkaProducer:
    """Simulated Kafka producer using in-memory queue."""
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
    
    def send(self, topic: str, value: Any):
        with self._lock:
            self._queue.append((topic, value, time.time()))
    
    def flush(self):
        pass

class _KafkaConsumer:
    """Simulated Kafka consumer."""
    def __init__(self, producer: _KafkaProducer):
        self._producer = producer
        self._offset = 0
    
    def poll(self, timeout: float = 1.0) -> List[tuple]:
        with self._producer._lock:
            available = self._producer._queue[self._offset:]
            if available:
                self._offset = len(self._producer._queue)
                return available
        return []

class SynopticSkyIndexerService:
    """Service managing data lake synchronization, Kafka streaming, and caching for synoptic sky indexer."""
    
    def __init__(self, cache_size: int = 10000, kafka_topic: str = "synoptic_sky_indexer"):
        self._cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size
        self._lock = threading.RLock()
        self._kafka_producer = _KafkaProducer()
        self._kafka_consumer = _KafkaConsumer(self._kafka_producer)
        self._kafka_topic = kafka_topic
        self._sync_state: Dict[str, float] = {}
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
    
    def _cache_get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                # Move to end for LRU
                if key in self._cache_order:
                    self._cache_order.remove(key)
                    self._cache_order.append(key)
                return self._cache[key]
        return None
    
    def _cache_set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                if key in self._cache_order:
                    self._cache_order.remove(key)
            self._cache[key] = value
            self._cache_order.append(key)
            # Evict oldest if over size
            while len(self._cache_order) > self._cache_size:
                oldest_key = self._cache_order.pop(0)
                self._cache.pop(oldest_key, None)
    
    def _cache_invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
            if key in self._cache_order:
                self._cache_order.remove(key)
    
    def publish_event(self, event: Dict[str, Any]) -> None:
        """Publish an observation event to the Kafka stream."""
        self._kafka_producer.send(self._kafka_topic, event)
        self._kafka_producer.flush()
    
    def consume_events(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Consume events from the Kafka stream."""
        messages = self._kafka_consumer.poll(timeout=timeout)
        return [msg[1] for msg in messages if msg[0] == self._kafka_topic]
    
    def sync_data_lake(self, cell_id: str, data: Dict[str, Any]) -> bool:
        """Synchronize data to the data lake for a given cell."""
        with self._lock:
            self._sync_state[cell_id] = time.time()
        # Simulate data lake write
        self._cache_set(f"lake:{cell_id}", data)
        return True
    
    def get_sync_status(self, cell_id: str) -> Optional[float]:
        """Get the last sync timestamp for a cell."""
        with self._lock:
            return self._sync_state.get(cell_id)
    
    def get_cached_cell(self, cell_id: str) -> Optional[Dict[str, Any]]:
        """Get a cached cell from the data lake."""
        return self._cache_get(f"lake:{cell_id}")
    
    def start_sync_loop(self, interval: float = 5.0) -> None:
        """Start the background synchronization loop."""
        if self._running:
            return
        self._running = True
        
        def _sync_loop():
            while self._running:
                events = self.consume_events(timeout=interval)
                for event in events:
                    cell_id = event.get("cell_id", "unknown")
                    self.sync_data_lake(cell_id, event)
                time.sleep(0.1)
        
        self._sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        self._sync_thread.start()
    
    def stop_sync_loop(self) -> None:
        """Stop the background synchronization loop."""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)
            self._sync_thread = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "cache_capacity": self._cache_size,
                "synced_cells": len(self._sync_state),
                "kafka_queue_size": len(self._kafka_producer._queue),
                "running": self._running,
            }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()
    
    def __del__(self):
        self.stop_sync_loop()