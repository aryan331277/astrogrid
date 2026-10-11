"""Synoptic sky indexer for unified multi-messenger observational astrophysics."""

import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class MessagerType(str):
    PHOTON = "photon"
    NEUTRINO = "neutrino"
    GRAVITATIONAL_WAVE = "gravitational_wave"


class ObservationEvent(BaseModel):
    """A single observational event to be indexed."""
    event_id: str
    messenger: MessagerType
    ra_deg: float = Field(ge=0.0, lt=360.0)
    dec_deg: float = Field(ge=-90.0, le=90.0)
    timestamp: datetime
    energy_gev: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class SkyIndexCell(BaseModel):
    """A cell in the spatiotemporal sky index."""
    cell_id: str
    ra_center_deg: float
    dec_center_deg: float
    t_start: datetime
    t_end: datetime
    events: List[ObservationEvent] = Field(default_factory=list)
    count: int = 0


class SynopticSkyIndexer:
    """
    Maps discrete observational events to a unified spatiotemporal grid.
    
    Uses a HEALPix-like equal-area pixelization on the sphere combined
    with temporal binning to create a 3D index (RA, Dec, Time).
    """

    def __init__(
        self,
        n_ra_bins: int = 360,
        n_dec_bins: int = 180,
        time_bin_seconds: float = 60.0,
    ):
        self.n_ra_bins = n_ra_bins
        self.n_dec_bins = n_dec_bins
        self.time_bin_seconds = time_bin_seconds
        self._index: Dict[str, SkyIndexCell] = {}
        self._events: List[ObservationEvent] = []

    def _compute_cell_id(self, ra_deg: float, dec_deg: float, timestamp: datetime) -> str:
        """Compute a unique cell identifier from spatial and temporal coordinates."""
        ra_bin = int(ra_deg / (360.0 / self.n_ra_bins)) % self.n_ra_bins
        dec_bin = int((dec_deg + 90.0) / (180.0 / self.n_dec_bins)) % self.n_dec_bins
        t_epoch = timestamp.timestamp()
        t_bin = int(t_epoch // self.time_bin_seconds)
        return f"{ra_bin}_{dec_bin}_{t_bin}"

    def _compute_cell_centers(self, ra_bin: int, dec_bin: int) -> Tuple[float, float]:
        """Compute the center RA and Dec for a given bin."""
        ra_center = (ra_bin + 0.5) * (360.0 / self.n_ra_bins)
        dec_center = (dec_bin + 0.5) * (180.0 / self.n_dec_bins) - 90.0
        return ra_center, dec_center

    def index_event(self, event: ObservationEvent) -> SkyIndexCell:
        """Index a single observation event into the spatiotemporal grid."""
        cell_id = self._compute_cell_id(event.ra_deg, event.dec_deg, event.timestamp)
        
        if cell_id not in self._index:
            ra_bin = int(event.ra_deg / (360.0 / self.n_ra_bins)) % self.n_ra_bins
            dec_bin = int((event.dec_deg + 90.0) / (180.0 / self.n_dec_bins)) % self.n_dec_bins
            ra_center, dec_center = self._compute_cell_centers(ra_bin, dec_bin)
            
            t_epoch = event.timestamp.timestamp()
            t_start_epoch = int(t_epoch // self.time_bin_seconds) * self.time_bin_seconds
            t_end_epoch = t_start_epoch + self.time_bin_seconds
            
            cell = SkyIndexCell(
                cell_id=cell_id,
                ra_center_deg=ra_center,
                dec_center_deg=dec_center,
                t_start=datetime.fromtimestamp(t_start_epoch, tz=timezone.utc),
                t_end=datetime.fromtimestamp(t_end_epoch, tz=timezone.utc),
            )
            self._index[cell_id] = cell
        
        self._index[cell_id].events.append(event)
        self._index[cell_id].count += 1
        self._events.append(event)
        return self._index[cell_id]

    def index_events(self, events: List[ObservationEvent]) -> Dict[str, SkyIndexCell]:
        """Index multiple observation events."""
        for event in events:
            self.index_event(event)
        return self._index

    def query_spatial(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        t_start: Optional[datetime] = None,
        t_end: Optional[datetime] = None,
    ) -> List[ObservationEvent]:
        """
        Query events within a circular region on the sky.
        
        Uses the spherical law of cosines for angular distance.
        """
        results: List[ObservationEvent] = []
        ra_rad = math.radians(ra_deg)
        dec_rad = math.radians(dec_deg)
        radius_rad = math.radians(radius_deg)
        
        for event in self._events:
            if t_start and event.timestamp < t_start:
                continue
            if t_end and event.timestamp > t_end:
                continue
            
            event_ra_rad = math.radians(event.ra_deg)
            event_dec_rad = math.radians(event.dec_deg)
            
            # Spherical law of cosines
            cos_dist = (
                math.sin(dec_rad) * math.sin(event_dec_rad)
                + math.cos(dec_rad) * math.cos(event_dec_rad)
                * math.cos(ra_rad - event_ra_rad)
            )
            cos_dist = max(-1.0, min(1.0, cos_dist))
            angular_dist = math.acos(cos_dist)
            
            if angular_dist <= radius_rad:
                results.append(event)
        
        return results

    def query_temporal(
        self,
        t_start: datetime,
        t_end: datetime,
        messenger: Optional[MessagerType] = None,
    ) -> List[ObservationEvent]:
        """Query events within a time window, optionally filtered by messenger type."""
        results: List[ObservationEvent] = []
        for event in self._events:
            if event.timestamp < t_start or event.timestamp > t_end:
                continue
            if messenger and event.messenger != messenger:
                continue
            results.append(event)
        return results

    def get_cell(self, cell_id: str) -> Optional[SkyIndexCell]:
        """Retrieve a specific cell by its ID."""
        return self._index.get(cell_id)

    def get_all_cells(self) -> List[SkyIndexCell]:
        """Return all indexed cells."""
        return list(self._index.values())

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the index."""
        total_events = len(self._events)
        total_cells = len(self._index)
        messenger_counts: Dict[str, int] = {}
        for event in self._events:
            key = event.messenger
            messenger_counts[key] = messenger_counts.get(key, 0) + 1
        
        return {
            "total_events": total_events,
            "total_cells": total_cells,
            "messenger_counts": messenger_counts,
            "n_ra_bins": self.n_ra_bins,
            "n_dec_bins": self.n_dec_bins,
            "time_bin_seconds": self.time_bin_seconds,
        }

    def clear(self) -> None:
        """Clear all indexed data."""
        self._index.clear()
        self._events.clear()