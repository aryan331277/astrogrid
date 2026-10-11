"""Service for stellar spectral classification with data lake sync, Kafka stream, and caching."""
import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpectralClassificationRequest(BaseModel):
    """Request model for spectral classification."""
    spectrum: List[float] = Field(..., description="Flux values of the spectrum")
    wavelengths: List[float] = Field(..., description="Wavelength values in Angstroms")
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio")


class SpectralClassificationResponse(BaseModel):
    """Response model for spectral classification."""
    spectral_type: str
    confidence: float
    temperature_k: float
    log_g: float
    metallicity: float
    processing_time_ms: float


class LRUCache:
    """Simple LRU cache implementation."""
    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()


class KafkaStreamSimulator:
    """Simulates Kafka stream for publishing classification results."""
    def __init__(self, topic: str = "stellar-spectral-classifications"):
        self.topic = topic
        self.messages: List[Dict[str, Any]] = []
        self._offset = 0

    def publish(self, payload: Dict[str, Any]) -> int:
        self._offset += 1
        msg = {
            "topic": self.topic,
            "offset": self._offset,
            "timestamp": time.time(),
            "payload": payload
        }
        self.messages.append(msg)
        logger.debug(f"Published to Kafka topic '{self.topic}' at offset {self._offset}")
        return self._offset

    def get_messages(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self._offset = 0


class DataLakeSyncService:
    """Simulates data lake synchronization for classification results."""
    def __init__(self, bucket: str = "astrogrid-spectral-data"):
        self.bucket = bucket
        self.stored_objects: Dict[str, Dict[str, Any]] = {}

    def sync(self, key: str, data: Dict[str, Any]) -> bool:
        self.stored_objects[key] = {
            "bucket": self.bucket,
            "key": key,
            "data": data,
            "synced_at": time.time()
        }
        logger.debug(f"Synced object to data lake: {self.bucket}/{key}")
        return True

    def get_object(self, key: str) -> Optional[Dict[str, Any]]:
        return self.stored_objects.get(key)

    def list_objects(self, prefix: str = "") -> List[str]:
        return [k for k in self.stored_objects if k.startswith(prefix)]


class StellarSpectralClassifierService:
    """
    Service for stellar spectral classification with data lake sync,
    Kafka stream publishing, and LRU caching.
    """

    def __init__(self, cache_size: int = 1024, kafka_topic: str = "stellar-spectral-classifications",
                 data_lake_bucket: str = "astrogrid-spectral-data"):
        self.cache = LRUCache(max_size=cache_size)
        self.kafka_stream = KafkaStreamSimulator(topic=kafka_topic)
        self.data_lake = DataLakeSyncService(bucket=data_lake_bucket)
        self._classifier = None

    def _get_classifier(self):
        """Lazy-initialize the spectral classifier."""
        if self._classifier is None:
            try:
                from astrogrid.engines.stellar_spectral_classifier import StellarSpectralClassifier
                self._classifier = StellarSpectralClassifier()
            except ImportError:
                logger.warning("StellarSpectralClassifier engine not available, using fallback")
                self._classifier = _FallbackClassifier()
        return self._classifier

    def _compute_cache_key(self, request: SpectralClassificationRequest) -> str:
        """Compute a deterministic cache key from the request."""
        spectrum_bytes = np.array(request.spectrum, dtype=np.float64).tobytes()
        wavelength_bytes = np.array(request.wavelengths, dtype=np.float64).tobytes()
        combined = spectrum_bytes + wavelength_bytes
        if request.snr is not None:
            combined += str(request.snr).encode()
        return hashlib.sha256(combined).hexdigest()

    def classify(self, request: SpectralClassificationRequest) -> SpectralClassificationResponse:
        """
        Classify a stellar spectrum with caching, Kafka publishing, and data lake sync.
        """
        cache_key = self._compute_cache_key(request)

        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for key {cache_key[:16]}...")
            return SpectralClassificationResponse(**cached)

        # Perform classification
        start_time = time.time()
        classifier = self._get_classifier()

        spectrum = np.array(request.spectrum, dtype=np.float64)
        wavelengths = np.array(request.wavelengths, dtype=np.float64)

        result = classifier.classify(spectrum, wavelengths, snr=request.snr)

        processing_time_ms = (time.time() - start_time) * 1000.0

        response_data = {
            "spectral_type": result.spectral_type,
            "confidence": result.confidence,
            "temperature_k": result.temperature_k,
            "log_g": result.log_g,
            "metallicity": result.metallicity,
            "processing_time_ms": processing_time_ms
        }

        # Store in cache
        self.cache.put(cache_key, response_data)

        # Publish to Kafka stream
        kafka_payload = {
            "cache_key": cache_key,
            "result": response_data,
            "timestamp": time.time()
        }
        self.kafka_stream.publish(kafka_payload)

        # Sync to data lake
        self.data_lake.sync(cache_key, kafka_payload)

        return SpectralClassificationResponse(**response_data)

    def get_cached_result(self, cache_key: str) -> Optional[SpectralClassificationResponse]:
        """Retrieve a cached classification result by key."""
        cached = self.cache.get(cache_key)
        if cached is not None:
            return SpectralClassificationResponse(**cached)
        return None

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self.cache.clear()
        logger.info("Spectral classification cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "cache_size": len(self.cache.cache),
            "cache_max_size": self.cache.max_size,
            "kafka_messages_published": len(self.kafka_stream.messages),
            "data_lake_objects": len(self.data_lake.stored_objects)
        }


class _FallbackClassifier:
    """Fallback classifier when the engine is not available."""

    def classify(self, spectrum: np.ndarray, wavelengths: np.ndarray,
                 snr: Optional[float] = None) -> Any:
        """Simple fallback classification based on spectral slope."""
        if len(spectrum) < 2:
            return _FallbackResult("O", 0.1, 30000.0, 4.0, 0.0)

        # Simple heuristic: use spectral slope as proxy
        slope = np.polyfit(wavelengths, spectrum, 1)[0]
        if slope > 0.01:
            return _FallbackResult("M", 0.5, 3500.0, 4.5, 0.0)
        elif slope > 0.001:
            return _FallbackResult("G", 0.6, 5800.0, 4.4, 0.0)
        elif slope > -0.001:
            return _FallbackResult("F", 0.6, 7000.0, 4.0, 0.0)
        elif slope > -0.01:
            return _FallbackResult("A", 0.7, 10000