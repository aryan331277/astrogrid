import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

// ============================================================================
// Types
// ============================================================================

interface HealpixSphericalPixelatorViewProps {
  /** Number of pixels in the HEALPix grid (must be 12 * N^2) */
  nside?: number;
  /** Ring or RING ordering */
  ordering?: 'RING' | 'NESTED';
  /** Pixel values to color (length must equal 12 * nside^2). If omitted, a demo pattern is generated. */
  pixelValues?: number[];
  /** Color scale min */
  colorMin?: number;
  /** Color scale max */
  colorMax?: number;
  /** Whether to show pixel borders */
  showBorders?: boolean;
  /** Border color */
  borderColor?: string;
  /** Border width in pixels */
  borderWidth?: number;
  /** Background color */
  backgroundColor?: string;
  /** Whether to enable rotation interaction */
  interactive?: boolean;
  /** Initial rotation in radians [x, y, z] */
  initialRotation?: [number, number, number];
  /** Canvas width in CSS pixels */
  width?: number;
  /** Canvas height in CSS pixels */
  height?: number;
  /** Callback when a pixel is clicked */
  onPixelClick?: (pixelIndex: number, theta: number, phi: number) => void;
  /** Callback when rotation changes */
  onRotationChange?: (rotation: [number, number, number]) => void;
  /** Class name for the wrapper */
  className?: string;
  /** Style for the wrapper */
  style?: React.CSSProperties;
  /** Show info overlay */
  showInfo?: boolean;
  /** Color map name */
  colorMap?: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm' | 'jet';
}

// ============================================================================
// Color Maps
// ============================================================================

type ColorMapName = 'viridis' | 'plasma' | 'inferno' | 'magma' | 'coolwarm' | 'jet';

interface ColorStop {
  t: number;
  r: number;
  g: number;
  b: number;
}

const COLOR_MAPS: Record<ColorMapName, ColorStop[]> = {
  viridis: [
    { t: 0.0, r: 68, g: 1, b: 84 },
    { t: 0.25, r: 59, g: 82, b: 139 },
    { t: 0.5, r: 33, g: 145, b: 140 },
    { t: 0.75, r: 94, g: 201, b: 98 },
    { t: 1.0, r: 253, g: 231, b: 37 },
  ],
  plasma: [
    { t: 0.0, r: 13, g: 8, b: 135 },
    { t: 0.25, r: 126, g: 3, b: 168 },
    { t: 0.5, r: 204, g: 71, b: 120 },
    { t: 0.75, r: 248, g: 149, b: 64 },
    { t: 1.0, r: 240, g: 249, b: 33 },
  ],
  inferno: [
    { t: 0.0, r: 0, g: 0, b: 4 },
    { t: 0.25, r: 87, g: 16, b: 110 },
    { t: 0.5, r: 188, g: 55, b: 84 },
    { t: 0.75, r: 249, g: 142, b: 7 },
    { t: 1.0, r: 252, g: 255, b: 164 },
  ],
  magma: [
    { t: 0.0, r: 0, g: 0, b: 4 },
    { t: 0.25, r: 81, g: 18, b: 124 },
    { t: 0.5, r: 183, g: 55, b: 121 },
    { t: 0.75, r: 252, g: 137, b: 97 },
    { t: 1.0, r: 252, g: 253, b: 191 },
  ],
  coolwarm: [
    { t: 0.0, r: 59, g: 76, b: 192 },
    { t: 0.25, r: 142, g: 155, b: 214 },
    { t: 0.5, r: 221, g: 221, b: 221 },
    { t: 0.75, r: 245, g: 156, b: 115 },
    { t: 1.0, r: 180, g: 4, b: 38 },
  ],
  jet: [
    { t: 0.0, r: 0, g: 0, b: 131 },
    { t: 0.25, r: 0, g: 255, b: 255 },
    { t: 0.5, r: 255, g: 255, b: 0 },
    { t: 0.75, r: 255, g: 0, b: 0 },
    { t: 1.0, r: 128, g: 0, b: 0 },
  ],
};

function sampleColorMap(mapName: ColorMapName, t: number): [number, number, number] {
  const stops = COLOR_MAPS[mapName] || COLOR_MAPS.viridis;
  const clampedT = Math.max(0, Math.min(1, t));

  let i = 0;
  while (i < stops.length - 1 && stops[i + 1].t < clampedT) {
    i++;
  }

  const s0 = stops[i];
  const s1 = stops[Math.min(i + 1, stops.length - 1)];
  const range = s1.t - s0.t;
  const frac = range > 0 ? (clampedT - s0.t) / range : 0;

  return [
    Math.round(s0.r + (s1.r - s0.r) * frac),
    Math.round(s0.g + (s1.g - s0.g) * frac),
    Math.round(s0.b + (s1.b - s0.b) * frac),
  ];
}

// ============================================================================
// HEALPix Geometry
// ============================================================================

interface HealpixPixel {
  index: number;
  /** Center theta (colatitude) in radians */
  theta: number;
  /** Center phi (longitude) in radians */
  phi: number;
  /** 4 corner vertices in 3D (x, y, z) on unit sphere */
  corners: [number, number, number][];
  /** Barycenter in 3D */
  barycenter: [number, number, number];
}

/**
 * Compute HEALPix pixel geometry for a given nside.
 * Uses the RING scheme for simplicity.
 */
function computeHealpixPixels(nside: number): HealpixPixel[] {
  const npix = 12 * nside * nside;
  const pixels: HealpixPixel[] = [];

  // For each pixel, compute its center and approximate corners
  // Using a simplified approach: for each pixel, we compute the center
  // and then approximate the 4 corners by offsetting in theta and phi

  const dTheta = Math.PI / (2 * nside); // approximate angular size in theta
  const dPhi = Math.PI / (2 * nside); // approximate angular size in phi

  for (let ipix = 0; ipix < npix; ip