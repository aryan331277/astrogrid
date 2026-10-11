"""
VOEvent Alert Broker Engine for AstroGrid.

Handles ingestion, parsing, validation, and routing of VOEvent 2.0 alerts
from multi-messenger observatories.
"""

import json
import re
import time
import hashlib
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import numpy as np

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Enumerated types of astrophysical alerts."""
    GAMMA_RAY = "gamma_ray"
    NEUTRINO = "neutrino"
    GRAVITATIONAL_WAVE = "gravitational_wave"
    TRANSIENT_OPTICAL = "transient_optical"
    FAST_RADIO_BURST = "fast_radio_burst"
    UNKNOWN = "unknown"


class AlertSeverity(int, Enum):
    """Severity levels for alert prioritization."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class VOEventAlert(BaseModel):
    """Parsed and validated VOEvent alert representation."""
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertType = Field(AlertType.UNKNOWN, description="Type of astrophysical alert")
    timestamp: datetime = Field(..., description="Observation timestamp (UTC)")
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")
    error_radius_arcsec: float = Field(0.0, ge=0.0, description="Positional uncertainty in arcseconds")
    significance: float = Field(0.0, description="Detection significance in sigma")
    severity: AlertSeverity = Field(AlertSeverity.LOW, description="Alert severity level")
    source_name: Optional[str] = Field(None, description="Source catalog name if known")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    raw_xml_hash: Optional[str] = Field(None, description="SHA-256 hash of raw XML payload")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @field_validator("alert_type", mode="before")
    @classmethod
    def parse_alert_type(cls, v):
        if isinstance(v, str):
            v_lower = v.lower().replace(" ", "_").replace("-", "_")
            for at in AlertType:
                if at.value == v_lower or at.value in v_lower:
                    return at
            return AlertType.UNKNOWN
        return v


class AlertBroker:
    """
    Broker for ingesting, validating, and routing VOEvent alerts.
    
    Maintains an in-memory alert queue with deduplication and
    severity-based prioritization.
    """

    def __init__(self, max_queue_size: int = 10000, dedup_window_sec: float = 300.0):
        self._queue: List[VOEventAlert] = []
        self._seen_hashes: Dict[str, float] = {}
        self._max_queue_size = max_queue_size
        self._dedup_window_sec = dedup_window_sec
        self._lock = None  # Placeholder for thread safety if needed

    def _compute_hash(self, alert: VOEventAlert) -> str:
        """Compute a deduplication hash from alert identity fields."""
        key = f"{alert.alert_id}|{alert.ra:.6f}|{alert.dec:.6f}|{alert.timestamp.isoformat()}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _is_duplicate(self, alert: VOEventAlert) -> bool:
        """Check if alert is a duplicate within the dedup window."""
        h = self._compute_hash(alert)
        now = time.time()
        if h in self._seen_hashes:
            if now - self._seen_hashes[h] < self._dedup_window_sec:
                return True
        self._seen_hashes[h] = now
        # Prune old entries
        if len(self._seen_hashes) > 50000:
            cutoff = now - self._dedup_window_sec
            self._seen_hashes = {k: v for k, v in self._seen_hashes.items() if v > cutoff}
        return False

    def ingest(self, raw_payload: Union[str, Dict[str, Any]]) -> Optional[VOEventAlert]:
        """
        Ingest a raw VOEvent payload (XML string or pre-parsed dict).
        Returns the validated alert or None if duplicate/invalid.
        """
        try:
            if isinstance(raw_payload, str):
                alert = self._parse_xml_payload(raw_payload)
            elif isinstance(raw_payload, dict):
                alert = VOEventAlert(**raw_payload)
            else:
                logger.warning("Unsupported payload type: %s", type(raw_payload))
                return None
        except Exception as e:
            logger.error("Failed to parse alert payload: %s", str(e))
            return None

        if self._is_duplicate(alert):
            logger.debug("Duplicate alert skipped: %s", alert.alert_id)
            return None

        self._enqueue(alert)
        logger.info("Ingested alert %s (type=%s, severity=%s)",
                     alert.alert_id, alert.alert_type.value, alert.severity.name)
        return alert

    def _parse_xml_payload(self, xml_str: str) -> VOEventAlert:
        """
        Minimal XML parsing for VOEvent 2.0 structure.
        Extracts key fields without external XML libraries.
        """
        raw_hash = hashlib.sha256(xml_str.encode()).hexdigest()
        data: Dict[str, Any] = {"raw_xml_hash": raw_hash}

        # Extract alert ID
        m = re.search(r'<ID[^>]*>([^<]+)</ID>', xml_str)
        if m:
            data["alert_id"] = m.group(1).strip()
        else:
            data["alert_id"] = f"unknown_{raw_hash[:12]}"

        # Extract timestamp
        m = re.search(r'<Time[^>]*>([^<]+)</Time>', xml_str)
        if m:
            data["timestamp"] = m.group(1).strip()

        # Extract RA/Dec from Position or SkyPosition
        m_ra = re.search(r'<RA[^>]*>([^<]+)</RA>', xml_str)
        m_dec = re.search(r'<Dec[^>]*>([^<]+)</Dec>', xml_str)
        if m_ra:
            data["ra"] = float(m_ra.group(1).strip())
        if m_dec:
            data["dec"] = float(m_dec.group(1).strip())

        # Extract error radius
        m_err = re.search(r'<ErrorRadius[^>]*>([^<]+)</ErrorRadius>', xml_str)
        if m_err:
            data["error_radius_arcsec"] = float(m_err.group(1).strip())

        # Extract significance
        m_sig = re.search(r'<Significance[^>]*>([^<]+)</Significance>', xml_str)
        if m_sig:
            data["significance"] = float(m_sig.group(1).strip())

        # Infer alert type from content
        data["alert_type"] = self._infer_alert_type(xml_str)

        # Severity from significance
        sig = data.get("significance", 0.0)
        if sig >= 5.0:
            data["severity"] = AlertSeverity.CRITICAL
        elif sig >= 4.0:
            data["severity"] = AlertSeverity.HIGH
        elif sig >= 3.0:
            data["severity"] = AlertSeverity.MEDIUM
        else:
            data["severity"] = AlertSeverity.LOW

        return VOEventAlert(**data)

    def _infer_alert_type(self, xml_str: str) -> AlertType:
        """Infer alert type from VOEvent XML content keywords."""
        lower = xml_str.lower()
        if "neutrino" in lower or "icecube" in lower:
            return AlertType.NEUTRINO
        if "gravitational" in lower or "ligo" in lower or "virgo" in lower:
            return AlertType.GRAVITATIONAL_WAVE
        if "gamma" in lower or "fermi" in lower or "integral" in lower:
            return AlertType.GAMMA_RAY
        if "fast radio" in lower or "frb" in lower:
            return AlertType.FAST_RADIO_BURST
        if "transient" in lower or "supernova" in lower or "afterglow" in lower:
            return AlertType.TRANSIENT_OPTICAL
        return AlertType.UNKNOWN

    def _enqueue(self, alert: VOEventAlert) -> None:
        """Insert alert into queue maintaining severity order."""
        self._queue.append(alert)
        if len(self._queue) > self._max_queue_size:
            self._queue = self._queue[-self._max_queue_size:]

    def get_alerts(self, min_severity: AlertSeverity = AlertSeverity.LOW,
                   alert_type: Optional[AlertType] = None,
                   limit: int = 100) -> List[VOEventAlert]:
        """Retrieve alerts filtered by severity and type."""
        filtered = [
            a for a in self._queue
            if a.severity >= min_severity and (alert_type is None or a.alert_type == alert_type)
        ]
        # Sort by severity