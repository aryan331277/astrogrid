"""Zero-mock unit tests for synoptic_sky_indexer module."""

import math
import sys
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.synoptic_sky_indexer import (
    MessagerType,
    ObservationEvent,
    SkyIndexCell,
    SynopticSkyIndexer,
)


class TestMessagerType:
    """Tests for MessagerType enum."""

    def test_photon_value(self):
        assert MessagerType.PHOTON == "photon"

    def test_neutrino_value(self):
        assert MessagerType.NEUTRINO == "neutrino"

    def test_gravitational_wave_value(self):
        assert MessagerType.GRAVITATIONAL_WAVE == "gravitational_wave"

    def test_all_values_distinct(self):
        values = [MessagerType.PHOTON, MessagerType.NEUTRINO, MessagerType.GRAVITATIONAL_WAVE]
        assert len(set(values)) == 3


class TestObservationEvent:
    """Tests for ObservationEvent model."""

    def test_basic_creation(self):
        event = ObservationEvent(
            event_id="evt_001",
            messenger_type=MessagerType.PHOTON,
            ra_deg=45.0,
            dec_deg=30.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert event.event_id == "evt_001"
        assert event.messenger_type == MessagerType.PHOTON
        assert event.ra_deg == 45.0
        assert event.dec_deg == 30.0

    def test_ensure_utc_naive_datetime(self):
        """Test that ensure_utc converts naive datetime to UTC."""
        event = ObservationEvent(
            event_id="evt_002",
            messenger_type=MessagerType.NEUTRINO,
            ra_deg=0.0,
            dec_deg=0.0,
            timestamp=datetime(2024, 6, 15, 8, 30, 0),
        )
        event.ensure_utc()
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.utcoffset() == datetime.min.utcoffset()  # UTC offset is 0

    def test_ensure_utc_already_utc(self):
        """Test that ensure_utc preserves already-UTC datetime."""
        original = datetime(2024, 3, 10, 14, 22, 5, tzinfo=timezone.utc)
        event = ObservationEvent(
            event_id="evt_003",
            messenger_type=MessagerType.GRAVITATIONAL_WAVE,
            ra_deg=180.0,
            dec_deg=-45.0,
            timestamp=original,
        )
        event.ensure_utc()
        assert event.timestamp == original

    def test_ensure_utc_with_offset(self):
        """Test that ensure_utc converts offset timezone to UTC."""
        from datetime import timedelta
        tz_plus2 = timezone(timedelta(hours=2))
        original = datetime(2024, 7, 4, 10, 0, 0, tzinfo=tz_plus2)
        event = ObservationEvent(
            event_id="evt_004",
            messenger_type=MessagerType.PHOTON,
            ra_deg=90.0,
            dec_deg=10.0,
            timestamp=original,
        )
        event.ensure_utc()
        # Should be 08:00 UTC
        assert event.timestamp.hour == 8
        assert event.timestamp.minute == 0
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.utcoffset() == timedelta(0)


class TestSkyIndexCell:
    """Tests for SkyIndexCell model."""

    def test_basic_creation(self):
        cell = SkyIndexCell(
            cell_id=1,
            center_ra_deg=45.0,
            center_dec_deg=30.0,
            events=[],
        )
        assert cell.cell_id == 1
        assert cell.center_ra_deg == 45.0
        assert cell.center_dec_deg == 30.0
        assert cell.events == []


class TestSynopticSkyIndexer:
    """Tests for SynopticSkyIndexer class."""

    def test_init_default_params(self):
        indexer = SynopticSkyIndexer()
        assert indexer is not None

    def test_init_custom_params(self):
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        assert indexer is not None

    def test_compute_cell_id_basic(self):
        """Test cell ID computation for a known position."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        # RA=45.0, Dec=30.0 with cell_size=10
        # RA cell index: floor(45.0 / 10.0) = 4
        # Dec cell index: floor(30.0 / 10.0) = 3
        cell_id = indexer._compute_cell_id(45.0, 30.0)
        assert isinstance(cell_id, int)
        assert cell_id >= 0

    def test_compute_cell_id_same_cell(self):
        """Test that positions in the same cell get the same ID."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        # Both in RA [40,50), Dec [30,40)
        id1 = indexer._compute_cell_id(42.0, 35.0)
        id2 = indexer._compute_cell_id(48.0, 39.0)
        assert id1 == id2

    def test_compute_cell_id_different_cells(self):
        """Test that positions in different cells get different IDs."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        # RA [40,50), Dec [30,40)
        id1 = indexer._compute_cell_id(42.0, 35.0)
        # RA [50,60), Dec [30,40)
        id2 = indexer._compute_cell_id(52.0, 35.0)
        assert id1 != id2

    def test_compute_cell_id_negative_dec(self):
        """Test cell ID computation with negative declination."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        cell_id = indexer._compute_cell_id(45.0, -30.0)
        assert isinstance(cell_id, int)
        assert cell_id >= 0

    def test_compute_cell_id_ra_wrapping(self):
        """Test that RA values wrap around 360 degrees."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        # RA=355 and RA=5 should be in adjacent cells (wrapping)
        id_355 = indexer._compute_cell_id(355.0, 0.0)
        id_5 = indexer._compute_cell_id(5.0, 0.0)
        # They should be different cells
        assert id_355 != id_5

    def test_compute_cell_id_boundary(self):
        """Test cell ID at exact boundary."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        # RA=40.0 should be in cell starting at 40
        id_at_40 = indexer._compute_cell_id(40.0, 0.0)
        id_at_39_99 = indexer._compute_cell_id(39.99, 0.0)
        assert id_at_40 != id_at_39_99

    def test_compute_cell_centers(self):
        """Test computation of cell centers."""
        indexer = SynopticSkyIndexer(cell_size_deg=10.0)
        centers = indexer._compute_cell_centers(45.0, 30.0)
        assert isinstance(centers, tuple)
        assert len(centers) == 2
        center_ra, center_dec = centers
        # Center should be within the cell
        assert 40.0 <= center_ra < 50.0
        assert 30.0 <= center_dec < 40.0

    def test_compute_cell_centers_negative_dec(self):
        """Test cell centers with negative declination."""
        indexer = SynopticSkyIndexer(cell_size