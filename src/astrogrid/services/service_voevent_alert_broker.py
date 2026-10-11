"""VOEvent Alert Broker service for data lake synchronization, Kafka stream, and caching."""

import hashlib
import json
import time
import threading
from typing import Dict, List, Optional, Any
from collections import deque

from pydantic import BaseModel, Field


class AlertType:
    """Enumeration of alert types."""
    TRANSIENT = "transient"
    VARIABLE = "variable"
    PERIODIC = "periodic"
    UNKNOWN = "unknown"


class AlertSeverity:
    """Enumeration of alert severities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VOEventAlert(BaseModel):
    """Pydantic model for a VOEvent alert."""
    alert_id: str
    alert_type: str = AlertType.UNKNOWN
    severity: str = AlertSeverity.LOW
    timestamp: float = Field(default_factory=time.time)
    ra: Optional[float] = None
    dec: Optional[float] = None
    magnitude: Optional[float] = None
    source: str = "unknown"
    payload: Dict[str, Any] = Field(default_factory=dict)

    def parse_timestamp(self, ts: str) -> float:
        """Parse ISO timestamp string to epoch float."""
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return time.time()

    def parse_alert_type(self, type_str: str) -> str:
        """Parse alert type string to canonical form."""
        t = type_str.lower().strip()
        if t in (AlertType.TRANSIENT, AlertType.VARIABLE, AlertType.PERIODIC):
            return t
        return AlertType.UNKNOWN


class AlertBroker:
    """Core alert broker with deduplication and hashing."""

    def __init__(self, max_cache_size: int = 10000):
        self._cache: Dict[str, VOEventAlert] = {}
        self._seen_hashes: set = set()
        self._max_cache_size = max_cache_size
        self._lock = threading.Lock()

    def _compute_hash(self, alert: VOEventAlert) -> str:
        """Compute a deterministic hash for deduplication."""
        key_data = f"{alert.alert_id}:{alert.source}:{alert.timestamp:.6f}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _is_duplicate(self, alert: VOEventAlert) -> bool:
        """Check if alert is a duplicate based on hash."""
        h = self._compute_hash(alert)
        return h in self._seen_hashes

    def ingest(self, alert: VOEventAlert) -> bool:
        """Ingest an alert, returning True if new, False if duplicate."""
        with self._lock:
            if self._is_duplicate(alert):
                return False
            h = self._compute_hash(alert)
            self._seen_hashes.add(h)
            self._cache[alert.alert_id] = alert
            if len(self._cache) > self._max_cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            return True

    def get(self, alert_id: str) -> Optional[VOEventAlert]:
        """Retrieve alert by ID."""
        with self._lock:
            return self._cache.get(alert_id)

    def list_alerts(self, limit: int = 100) -> List[VOEventAlert]:
        """List recent alerts."""
        with self._lock:
            return list(self._cache.values())[-limit:]


class KafkaStreamSimulator:
    """Simulated Kafka stream for alert distribution."""

    def __init__(self, topic: str = "voevent_alerts"):
        self.topic = topic
        self._messages: deque = deque(maxlen=10000)
        self._subscribers: List = []
        self._lock = threading.Lock()

    def publish(self, alert: VOEventAlert) -> None:
        """Publish alert to stream."""
        msg = {
            "topic": self.topic,
            "key": alert.alert_id,
            "value": alert.model_dump(),
            "timestamp": time.time(),
        }
        with self._lock:
            self._messages.append(msg)
            for cb in self._subscribers:
                try:
                    cb(msg)
                except Exception:
                    pass

    def subscribe(self, callback) -> None:
        """Subscribe a callback to the stream."""
        with self._lock:
            self._subscribers.append(callback)

    def poll(self, max_msgs: int = 100) -> List[Dict]:
        """Poll messages from the stream."""
        with self._lock:
            msgs = []
            while self._messages and len(msgs) < max_msgs:
                msgs.append(self._messages.popleft())
            return msgs


class DataLakeSync:
    """Simulated data lake synchronization."""

    def __init__(self, batch_size: int = 100):
        self._buffer: List[Dict] = []
        self._batch_size = batch_size
        self._synced_count = 0
        self._lock = threading.Lock()

    def buffer_alert(self, alert: VOEventAlert) -> None:
        """Buffer an alert for sync."""
        with self._lock:
            self._buffer.append(alert.model_dump())
            if len(self._buffer) >= self._batch_size:
                self._flush()

    def _flush(self) -> None:
        """Flush buffered alerts to data lake."""
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        self._synced_count += len(batch)

    def force_flush(self) -> int:
        """Force flush and return number of records synced."""
        with self._lock:
            self._flush()
            return self._synced_count

    @property
    def synced_count(self) -> int:
        return self._synced_count


class VOEventAlertBrokerService:
    """Main service combining broker, Kafka stream, and data lake sync."""

    def __init__(self, max_cache_size: int = 10000, batch_size: int = 100):
        self.broker = AlertBroker(max_cache_size=max_cache_size)
        self.kafka = KafkaStreamSimulator(topic="voevent_alerts")
        self.data_lake = DataLakeSync(batch_size=batch_size)

    def process_alert(self, alert: VOEventAlert) -> bool:
        """Process an alert through the full pipeline."""
        is_new = self.broker.ingest(alert)
        if is_new:
            self.kafka.publish(alert)
            self.data_lake.buffer_alert(alert)
        return is_new

    def get_alert(self, alert_id: str) -> Optional[VOEventAlert]:
        """Get alert by ID."""
        return self.broker.get(alert_id)

    def list_alerts(self, limit: int = 100) -> List[VOEventAlert]:
        """List recent alerts."""
        return self.broker.list_alerts(limit=limit)

    def flush_data_lake(self) -> int:
        """Force flush data lake buffer."""
        return self.data_lake.force_flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "cached_alerts": len(self.broker._cache),
            "synced_records": self.data_lake.synced_count,
            "kafka_buffered": len(self.kafka._messages),
        }