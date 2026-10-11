"""
AstroGrid: Unified Multi-Messenger Observational Astrophysics Platform
Main FastAPI application entry point.
"""

from fastapi import FastAPI

# Import all router modules
from astrogrid.routers import (
    router_fits_header_wcs_parser,
    router_celestial_coordinate_transform,
    router_healpix_spherical_pixelator,
    router_wcs_tangent_solver,
    router_atmospheric_extinction_model,
    router_cosmic_ray_lacosmic,
    router_psf_fitting_engine,
    router_aperture_photometry,
    router_light_curve_detrender,
    router_transit_bls_detector,
    router_radial_velocity_solver,
    router_doppler_redshift_estimator,
    router_stellar_spectral_classifier,
    router_stellar_isochrone_interpolator,
    router_supernova_arnett_model,
)

# Create the FastAPI application
app = FastAPI(
    title="AstroGrid: Unified Multi-Messenger Observational Astrophysics Platform",
    version="1.0.0",
    description="A unified platform for multi-messenger observational astrophysics, "
                "providing modular services for FITS parsing, coordinate transformations, "
                "photometry, spectroscopy, and more.",
)

# Mount all 15 modular routers
app.include_router(router_fits_header_wcs_parser.router)
app.include_router(router_celestial_coordinate_transform.router)
app.include_router(router_healpix_spherical_pixelator.router)
app.include_router(router_wcs_tangent_solver.router)
app.include_router(router_atmospheric_extinction_model.router)
app.include_router(router_cosmic_ray_lacosmic.router)
app.include_router(router_psf_fitting_engine.router)
app.include_router(router_aperture_photometry.router)
app.include_router(router_light_curve_detrender.router)
app.include_router(router_transit_bls_detector.router)
app.include_router(router_radial_velocity_solver.router)
app.include_router(router_doppler_redshift_estimator.router)
app.include_router(router_stellar_spectral_classifier.router)
app.include_router(router_stellar_isochrone_interpolator.router)
app.include_router(router_supernova_arnett_model.router)


@app.get("/")
async def root():
    """Root endpoint returning service information."""
    return {
        "service": "AstroGrid: Unified Multi-Messenger Observational Astrophysics Platform",
        "version": "1.0.0",
        "status": "online",
        "routers": [
            "fits_header_wcs_parser",
            "celestial_coordinate_transform",
            "healpix_spherical_pixelator",
            "wcs_tangent_solver",
            "atmospheric_extinction_model",
            "cosmic_ray_lacosmic",
            "psf_fitting_engine",
            "aperture_photometry",
            "light_curve_detrender",
            "transit_bls_detector",
            "radial_velocity_solver",
            "doppler_redshift_estimator",
            "stellar_spectral_classifier",
            "stellar_isochrone_interpolator",
            "supernova_arnett_model",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}