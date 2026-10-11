import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ExtinctionPoint {
  wavelength: number; // micrometers
  extinction: number; // magnitudes
}

interface ExtinctionModel {
  id: string;
  name: string;
  description: string;
  color: string;
  points: ExtinctionPoint[];
}

interface Bandpass {
  id: string;
  name: string;
  center: number; // micrometers
  width: number; // micrometers
  color: string;
}

interface Viewport {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface HoverState {
  x: number;
  y: number;
  wavelength: number;
  values: { model: ExtinctionModel; extinction: number }[];
}

interface ExtinctionModelViewProps {
  models?: ExtinctionModel[];
  bandpasses?: Bandpass[];
  airmass?: number;
  altitude?: number; // meters
  showBandpasses?: boolean;
  showGrid?: boolean;
  showLegend?: boolean;
  showAirmassScaling?: boolean;
  height?: number;
  className?: string;
  onHover?: (state: HoverState | null) => void;
}

// ---------------------------------------------------------------------------
// Default data
// ---------------------------------------------------------------------------

const DEFAULT_MODELS: ExtinctionModel[] = [
  {
    id: 'cardelli',
    name: 'Cardelli (1989)',
    description: 'Milky Way R_V=3.1 extinction law',
    color: '#4f8ef7',
    points: [
      { wavelength: 0.3, extinction: 1.086 },
      { wavelength: 0.4, extinction: 0.886 },
      { wavelength: 0.5, extinction: 0.786 },
      { wavelength: 0.6, extinction: 0.786 },
      { wavelength: 0.7, extinction: 0.786 },
      { wavelength: 0.8, extinction: 0.786 },
      { wavelength: 0.9, extinction: 0.786 },
      { wavelength: 1.0, extinction: 0.786 },
      { wavelength: 1.2, extinction: 0.786 },
      { wavelength: 1.5, extinction: 0.786 },
      { wavelength: 2.0, extinction: 0.786 },
      { wavelength: 2.5, extinction: 0.786 },
      { wavelength: 3.0, extinction: 0.786 },
      { wavelength: 4.0, extinction: 0.786 },
      { wavelength: 5.0, extinction: 0.786 },
    ],
  },
  {
    id: 'calzetti',
    name: 'Calzetti (2000)',
    description: 'Starburst R_V=3.0 extinction law',
    color: '#f76e4f',
    points: [
      { wavelength: 0.3, extinction: 1.15 },
      { wavelength: 0.4, extinction: 0.95 },
      { wavelength: 0.5, extinction: 0.85 },
      { wavelength: 0.6, extinction: 0.85 },
      { wavelength: 0.7, extinction: 0.85 },
      { wavelength: 0.8, extinction: 0.85 },
      { wavelength: 0.9, extinction: 0.85 },
      { wavelength: 1.0, extinction: 0.85 },
      { wavelength: 1.2, extinction: 0.85 },
      { wavelength: 1.5, extinction: 0.85 },
      { wavelength: 2.0, extinction: 0.85 },
      { wavelength: 2.5, extinction: 0.85 },
      { wavelength: 3.0, extinction: 0.85 },
      { wavelength: 4.0, extinction: 0.85 },
      { wavelength: 5.0, extinction: 0.85 },
    ],
  },
  {
    id: 'odonoghue',
    name: 'Odonoghue (1994)',
    description: 'Galactic R_V=3.1 extinction law',
    color: '#4ff7a8',
    points: [
      { wavelength: 0.3, extinction: 1.05 },
      { wavelength: 0.4, extinction: 0.85 },
      { wavelength: 0.5, extinction: 0.75 },
      { wavelength: 0.6, extinction: 0.75 },
      { wavelength: 0.7, extinction: 0.75 },
      { wavelength: 0.8, extinction: 0.75 },
      { wavelength: 0.9, extinction: 0.75 },
      { wavelength: 1.0, extinction: 0.75 },
      { wavelength: 1.2, extinction: 0.75 },
      { wavelength: 1.5, extinction: 0.75 },
      { wavelength: 2.0, extinction: 0.75 },
      { wavelength: 2.5, extinction: 0.75 },
      { wavelength: 3.0, extinction: 0.75 },
      { wavelength: 4.0, extinction: 0.75 },
      { wavelength: 5.0, extinction: 0.75 },
    ],
  },
];

const DEFAULT_BANDPASSES: Bandpass[] = [
  { id: 'u', name: 'U', center: 0.365, width: 0.06, color: '#8b5cf6' },
  { id: 'g', name: 'G', center: 0.475, width: 0.06, color: '#22c55e' },
  { id: 'r', name: 'R', center: 0.616, width: 0.06, color: '#ef4444' },
  { id: 'i', name: 'I', center: 0.748, width: 0.06, color: '#f97316' },
  { id: 'z', name: 'Z', center: 0.893, width: 0.06, color: '#eab308' },
  { id: 'j', name: 'J', center: 1.25, width: 0.15, color: '#ec4899' },
  { id: 'h', name: 'H', center: 1.65, width: 0.15, color: '#06b6d4' },
  { id: 'k', name: 'K', center: 2.2, width: 0.2, color: '#a855f7' },
];

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WAVELENGTH_MIN = 0.25;
const WAVELENGTH_MAX = 5.5;
const EXTINCTION_MIN = 0;
const EXTINCTION_MAX = 1.5;
const PADDING = { top: 40, right: 30, bottom: 50, left: 60 };
const GRID_LINES = 10;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function interpolateExtinction(
  model: ExtinctionModel,
  wavelength: number
): number {
  const points = model.points;
  if (wavelength <= points[0].wavelength) return points[0].extinction;
  if (wavelength >= points[points.length - 1].wavelength)
    return points[points.length - 1].extinction;

  for (let i = 0; i < points.length - 1; i++) {
    if (
      wavelength >= points[i].wavelength &&
      wavelength <= points[i + 1].wavelength
    ) {
      const t =
        (wavelength - points[i].wavelength) /
        (points[i + 1].wavelength - points[i].wavelength);
      return points[i].ext