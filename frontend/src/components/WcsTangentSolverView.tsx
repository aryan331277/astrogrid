import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

// ============================================================================
// Types
// ============================================================================

interface Vec2 {
  x: number;
  y: number;
}

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

interface WcsParams {
  crval1: number;
  crval2: number;
  crpix1: number;
  crpix2: number;
  cdelt1: number;
  cdelt2: number;
  cde1_1: number;
  cde1_2: number;
  cde2_1: number;
  cde2_2: number;
  equinox: number;
}

interface TangentPoint {
  x: number;
  y: number;
  ra: number;
  dec: number;
}

interface GridLine {
  points: Vec2[];
  isRA: boolean;
  label: string;
}

interface SolverState {
  wcs: WcsParams;
  tangent: TangentPoint;
  gridLines: GridLine[];
  pixelGrid: Vec2[][];
  celestialGrid: Vec2[][];
  zoom: number;
  pan: Vec2;
  showPixelGrid: boolean;
  showCelestialGrid: boolean;
  showTangent: boolean;
  showLabels: boolean;
  selectedPoint: Vec2 | null;
  hoverPoint: Vec2 | null;
  isDragging: boolean;
  dragStart: Vec2 | null;
  panStart: Vec2 | null;
}

interface TooltipData {
  x: number;
  y: number;
  ra: number;
  dec: number;
  px: number;
  py: number;
}

// ============================================================================
// Constants
// ============================================================================

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

const COLORS = {
  background: '#0a0e1a',
  gridPixel: 'rgba(100, 140, 200, 0.15)',
  gridPixelMajor: 'rgba(100, 140, 200, 0.35)',
  gridCelestial: 'rgba(80, 200, 160, 0.25)',
  gridCelestialMajor: 'rgba(80, 200, 160, 0.5)',
  tangent: '#ff6b6b',
  tangentGlow: 'rgba(255, 107, 107, 0.3)',
  point: '#ffd93d',
  pointGlow: 'rgba(255, 217, 61, 0.4)',
  label: 'rgba(200, 220, 255, 0.8)',
  labelMajor: 'rgba(200, 220, 255, 1.0)',
  axis: 'rgba(150, 180, 220, 0.6)',
  tooltipBg: 'rgba(15, 20, 35, 0.95)',
  tooltipBorder: 'rgba(100, 140, 200, 0.5)',
  tooltipText: '#e0e8ff',
  panelBg: 'rgba(12, 16, 28, 0.95)',
  panelBorder: 'rgba(60, 80, 120, 0.4)',
  panelText: '#c0d0f0',
  panelAccent: '#4a9eff',
  buttonBg: 'rgba(40, 60, 100, 0.6)',
  buttonActive: 'rgba(74, 158, 255, 0.3)',
  buttonBorder: 'rgba(100, 140, 200, 0.4)',
};

const DEFAULT_WCS: WcsParams = {
  crval1: 180.0,
  crval2: -45.0,
  crpix1: 512.0,
  crpix2: 512.0,
  cdelt1: -0.001,
  cdelt2: 0.001,
  cde1_1: 1.0,
  cde1_2: 0.0,
  cde2_1: 0.0,
  cde2_2: 1.0,
  equinox: 2000.0,
};

const IMAGE_WIDTH = 1024;
const IMAGE_HEIGHT = 1024;

// ============================================================================
// WCS Math Functions
// ============================================================================

/**
 * Convert pixel coordinates to celestial coordinates (RA, Dec) using tangent-plane projection.
 */
function pixelToCelestial(px: number, py: number, wcs: WcsParams): { ra: number; dec: number } {
  const dx = px - wcs.crpix1;
  const dy = py - wcs.crpix2;

  // Apply CD matrix
  const x = wcs.cde1_1 * dx + wcs.cde1_2 * dy;
  const y = wcs.cde2_1 * dx + wcs.cde2_2 * dy;

  // Apply pixel scale
  const theta1 = x * wcs.cdelt1 * DEG_TO_RAD;
  const theta2 = y * wcs.cdelt2 * DEG_TO_RAD;

  // Tangent plane projection (gnomonic)
  const crval1Rad = wcs.crval1 * DEG_TO_RAD;
  const crval2Rad = wcs.crval2 * DEG_TO_RAD;

  const cosDec = Math.cos(crval2Rad);
  const sinDec = Math.sin(crval2Rad);

  const xPrime = theta1 * cosDec;
  const yPrime = theta2;

  const rho2 = xPrime * xPrime + yPrime * yPrime;
  const rho = Math.sqrt(rho2);

  if (rho === 0) {
    return { ra: wcs.crval1, dec: wcs.crval2 };
  }

  const chi = 2 * Math.atan2(rho, 1 + Math.sqrt(1 - rho2));
  const sinChi = Math.sin(chi);
  const cosChi = Math.cos(chi);

  const dec = Math.asin(
    sinDec * cosChi + cosDec * (xPrime * Math.sin(crval1Rad) - yPrime * Math.cos(crval1Rad)) * sinChi / rho
  );

  const ra = crval1Rad + Math.atan2(
    xPrime * Math.cos(crval1Rad) + yPrime * Math.sin(crval1Rad),
    rho * cosDec * cosChi - (xPrime * Math.sin(crval1Rad) - yPrime * Math.cos(crval1Rad)) * sinDec * sinChi
  );

  return { ra: ra * RAD_TO_DEG, dec: dec * RAD_TO_DEG };
}

/**
 * Convert celestial coordinates (RA, Dec) to pixel coordinates using tangent-plane projection.
 */
function celestialToPixel(ra: number, dec: number, wcs: WcsParams): { px: number; py: number } {
  const raRad = ra * DEG_TO_RAD;
  const decRad = dec * DEG_TO_RAD;
  const crval1Rad = wcs.crval1 * DEG_TO_RAD;
  const crval2Rad = wcs.crval2 * DEG_TO_RAD;

  const sinDec = Math.sin(decRad);
  const cosDec = Math.cos(decRad);
  const sinCrval2 = Math.sin(crval2Rad);
  const cosCrval2 = Math.cos(crval2Rad);

  const cosDelta = sinDec * sinCrval2 + cosDec * cosCrval2 * Math.cos(raRad - crval1Rad);

  if (cosDelta <= 0) {
    return { px: NaN, py: NaN };
  }

  const sinDelta = Math.sqrt(1 - cosDelta * cosDelta);
  const tanDelta = sinDelta / cosDelta;

  const xPrime = cosDec * Math.sin(raRad - crval1Rad) * tanDelta;
  const yPrime = (sinDec * cosCrval2 - cosDec * sinCrval2 * Math.cos(raRad - crval1Rad)) * tanDelta;

  const theta1 = xPrime / cosCrval2