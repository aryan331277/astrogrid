THIS WAS BUILT BY 339 AI AGENTS, THERE ARE SOME ERRORS IN THIS. INITITALLY THE PLAN WAS 500 AGENTS BUT DUE TO SOME ISSUES 500 IS TAKING LONGER THAN EXPECTED, PROGRESS TILL 339 IS HERE AND THE QA DESK AND THE EDITORIAL ARE LEFT TO RUN.

# AstroGrid: Unified Multi-Messenger Observational Astrophysics Platform

**Mission**: To engineer a scalable, real-time pipeline for processing, calibrating, and analyzing multi-wavelength astronomical data, enabling automated discovery of transient events, precise astrometry, and deep-field characterization through a modular, high-performance computing architecture.

---

## Architecture Overview

AstroGrid is structured into modular layers adhering to strict architectural boundaries:

1. **Computational Physics Engines (`src/astrogrid/engines/`)**:
   - `celestial_coordinate_transform`: High-precision astrometric frame conversions (ICRF, GCRS, ITRF).
   - `healpix_spherical_pixelator`: HEALPix equal-area spherical tessellation (RING and NESTED schemes).
   - `wcs_tangent_solver`: Gnomonic tangent-plane World Coordinate System projections.
   - `atmospheric_extinction_model`: Rayleigh scattering, ozone absorption, and Kasten-Young airmass modeling.
   - `cosmic_ray_lacosmic`: Laplacian edge filtering for high-energy cosmic-ray strike removal.
   - `transit_bls_detector`: Box-Least-Squares exoplanetary transit periodogram searches.
   - `radio_interferometry_synthesizer`: Visibility grid inversion and Fourier sky synthesis.
   - `gravitational_lensing_engine`: Deflection angles, magnification maps, and Einstein radii.
   - `keplerian_orbit_engine`: Numerical and analytical two-body orbital state vectors.
   - `multi_messenger_correlator`: Multi-channel coincidence correlation (GW, Neutrino, EM).

2. **Pydantic v2 Data Contracts (`src/astrogrid/models/`)**:
   - Strict input validation schemas, telemetry records, and domain invariants with `@field_validator`.

3. **FastAPI API Routers (`src/astrogrid/routers/`)**:
   - 15 dedicated routers mounted on the central application gateway (`src/astrogrid/main.py`).

4. **Streaming & Lake Services (`src/astrogrid/services/`)**:
   - Event-driven Kafka ingestion simulations, in-memory caching, and data lake synchronization.

5. **Frontend WebGL/Canvas Visualizers (`frontend/`)**:
   - React 18 TypeScript interactive visualizer dashboards with HTML5 Canvas and SVG plots.

6. **Principle 1 Zero-Mock Test Suite (`tests/unit/`)**:
   - 100% executable unit tests asserting physical conservation laws, numerical stability, and domain bounds without mocks.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/aryan331277/astrogrid.git
cd astrogrid

# Install dependencies
pip install -e .

# Run the test suite
pytest tests/unit -v

# Launch the FastAPI Gateway
uvicorn src.astrogrid.main:app --reload --port 8000
```
