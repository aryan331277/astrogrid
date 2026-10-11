import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

// ============================================================================
// Types
// ============================================================================

type CoordinateSystem = 'equatorial' | 'galactic' | 'ecliptic' | 'horizontal';

interface CelestialCoordinate {
  /** Right Ascension / Longitude / Azimuth in degrees */
  primary: number;
  /** Declination / Latitude / Altitude in degrees */
  secondary: number;
  /** Distance in parsecs (optional, for 3D context) */
  distance?: number;
}

interface TransformResult {
  system: CoordinateSystem;
  coordinate: CelestialCoordinate;
  label: string;
}

interface TransformMatrix {
  name: string;
  matrix: number[][];
}

interface CelestialCoordinateTransformViewProps {
  /** Initial coordinate to display */
  initialCoordinate?: CelestialCoordinate;
  /** Initial coordinate system */
  initialSystem?: CoordinateSystem;
  /** Observer latitude in degrees (for horizontal system) */
  observerLatitude?: number;
  /** Observer longitude in degrees (for horizontal system) */
  observerLongitude?: number;
  /** Local sidereal time in hours (for horizontal system) */
  localSiderealTime?: number;
  /** Whether to show the 3D sphere */
  showSphere?: boolean;
  /** Whether to show grid lines */
  showGrid?: boolean;
  /** Whether to show coordinate labels */
  showLabels?: boolean;
  /** Animation speed multiplier */
  animationSpeed?: number;
  /** Callback when coordinate changes */
  onCoordinateChange?: (result: TransformResult) => void;
  /** Callback when transform matrix is computed */
  onMatrixChange?: (matrix: TransformMatrix) => void;
  /** Custom class name */
  className?: string;
  /** Width of the component */
  width?: number;
  /** Height of the component */
  height?: number;
}

// ============================================================================
// Constants
// ============================================================================

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

const SYSTEM_LABELS: Record<CoordinateSystem, string> = {
  equatorial: 'Equatorial (RA/Dec)',
  galactic: 'Galactic (l/b)',
  ecliptic: 'Ecliptic (λ/β)',
  horizontal: 'Horizontal (Az/Alt)',
};

const SYSTEM_COLORS: Record<CoordinateSystem, string> = {
  equatorial: '#4fc3f7',
  galactic: '#81c784',
  ecliptic: '#ffb74d',
  horizontal: '#e57373',
};

const GRID_COLOR = 'rgba(255, 255, 255, 0.15)';
const AXIS_COLOR = 'rgba(255, 255, 255, 0.4)';
const SPHERE_COLOR = 'rgba(30, 60, 120, 0.3)';
const POINT_COLOR = '#ffffff';
const LABEL_COLOR = 'rgba(255, 255, 255, 0.8)';

// ============================================================================
// Coordinate Transformation Math
// ============================================================================

/**
 * Convert equatorial (RA in hours, Dec in degrees) to galactic (l, b in degrees)
 */
function equatorialToGalactic(raHours: number, decDeg: number): { l: number; b: number } {
  const ra = raHours * 15 * DEG_TO_RAD;
  const dec = decDeg * DEG_TO_RAD;

  // Galactic pole in equatorial: RA = 192.85948°, Dec = 27.12825°
  const galPoleRA = 192.85948 * DEG_TO_RAD;
  const galPoleDec = 27.12825 * DEG_TO_RAD;

  // Galactic center in equatorial: RA = 266.4051°, Dec = -28.936175°
  const galCenterRA = 266.4051 * DEG_TO_RAD;
  const galCenterDec = -28.936175 * DEG_TO_RAD;

  const sinB =
    Math.sin(galPoleDec) * Math.sin(dec) +
    Math.cos(galPoleDec) * Math.cos(dec) * Math.cos(ra - galPoleRA);

  const b = Math.asin(Math.max(-1, Math.min(1, sinB)));

  const cosL =
    (Math.sin(dec) - Math.sin(galPoleDec) * sinB) /
    (Math.cos(galPoleDec) * Math.cos(b));

  const sinL =
    (Math.cos(dec) * Math.sin(ra - galPoleRA)) / Math.cos(b);

  let l = Math.atan2(sinL, cosL);
  if (l < 0) l += 2 * Math.PI;

  return { l: l * RAD_TO_DEG, b: b * RAD_TO_DEG };
}

/**
 * Convert galactic (l, b in degrees) to equatorial (RA in hours, Dec in degrees)
 */
function galacticToEquatorial(lDeg: number, bDeg: number): { raHours: number; decDeg: number } {
  const l = lDeg * DEG_TO_RAD;
  const b = bDeg * DEG_TO_RAD;

  const galPoleRA = 192.85948 * DEG_TO_RAD;
  const galPoleDec = 27.12825 * DEG_TO_RAD;
  const galCenterRA = 266.4051 * DEG_TO_RAD;
  const galCenterDec = -28.936175 * DEG_TO_RAD;

  const sinDec =
    Math.sin(galPoleDec) * Math.sin(b) +
    Math.cos(galPoleDec) * Math.cos(b) * Math.cos(l);

  const dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));

  const cosRA =
    (Math.sin(l) * Math.cos(b) * Math.cos(galPoleDec) -
      Math.sin(b) * Math.sin(galPoleDec)) /
    Math.cos(dec);

  const sinRA =
    (Math.cos(b) * Math.sin(l)) / Math.cos(dec);

  let ra = Math.atan2(sinRA, cosRA);
  if (ra < 0) ra += 2 * Math.PI;

  return { raHours: ra * RAD_TO_DEG / 15, decDeg: dec * RAD_TO_DEG };
}

/**
 * Convert equatorial (RA in hours, Dec in degrees) to ecliptic (λ, β in degrees)
 */
function equatorialToEcliptic(raHours: number, decDeg: number): { lambda: number; beta: number } {
  const ra = raHours * 15 * DEG_TO_RAD;
  const dec = decDeg * DEG_TO_RAD;
  const obliquity = 23.4392911 * DEG_TO_RAD;

  const sinBeta =
    Math.sin(dec) * Math.cos(obliquity) -
    Math.cos(dec) * Math.sin(obliquity) * Math.sin(ra);

  const beta = Math.asin(Math.max(-1, Math.min(1, sinBeta)));

  const cosLambda =
    (Math.cos(dec) * Math.cos(ra)) / Math.cos(beta);
  const sinLambda =
    (Math.cos(dec) * Math.sin(ra) * Math.cos(obliquity) +
      Math.sin(dec) * Math.sin(obliquity)) /
    Math.cos(beta);

  let lambda = Math.atan2(sinLambda, cosLambda);
  if (lambda < 0) lambda += 2 * Math.PI;

  return { lambda: lambda * RAD_TO_DEG, beta: beta * RAD_TO_DEG };
}

/**
 * Convert ecliptic (λ, β in degrees) to equatorial (RA in hours, Dec in degrees)
 */
function eclipticToEquatorial(lambdaDeg: number, betaDeg: number): { raHours: number; decDeg: number } {
  const lambda = lambdaDeg * DEG_TO_RAD;
  const beta = betaDeg * DEG_TO_RAD;
  const obliquity = 23.4392911 * DEG_TO_RAD;

  const sinDec =
    Math.sin(beta) * Math.cos(obliquity) +
    Math.cos(beta) * Math.sin(obliquity) * Math.sin(lambda);

  const dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));