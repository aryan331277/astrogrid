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

export interface TransitEvent {
  id: string;
  t0: number;
  duration: number;
  depth: number;
  period: number;
  ingress: number;
  egress: number;
  u1?: number;
  u2?: number;
}

export interface TransitBlsDetectorViewProps {
  fluxData: number[];
  timeData: number[];
  uncertainties?: number[];
  transitEvents?: TransitEvent[];
  period?: number;
  t0?: number;
  title?: string;
  showPhaseFolded?: boolean;
  showBlsSpectrum?: boolean;
  showModel?: boolean;
  theme?: 'dark' | 'light';
  width?: number;
  height?: number;
  onEventClick?: (event: TransitEvent) => void;
  onPointHover?: (index: number | null) => void;
}

interface BlsPeak {
  period: number;
  power: number;
  depth: number;
  duration: number;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  text: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COLORS = {
  dark: {
    bg: '#0d1117',
    grid: '#21262d',
    axis: '#484f58',
    text: '#c9d1d9',
    textSecondary: '#8b949e',
    flux: '#58a6ff',
    fluxFill: 'rgba(88, 166, 255, 0.15)',
    model: '#f0883e',
    modelFill: 'rgba(240, 136, 62, 0.2)',
    event: '#3fb950',
    eventFill: 'rgba(63, 185, 80, 0.15)',
    bls: '#bc8cff',
    blsFill: 'rgba(188, 140, 255, 0.15)',
    blsPeak: '#ff7b72',
    uncertainty: 'rgba(88, 166, 255, 0.3)',
    crosshair: 'rgba(201, 209, 217, 0.4)',
    tooltipBg: 'rgba(22, 27, 34, 0.95)',
    tooltipBorder: '#30363d',
    panelBg: '#161b22',
    panelBorder: '#30363d',
    highlight: '#d29922',
  },
  light: {
    bg: '#ffffff',
    grid: '#e1e4e8',
    axis: '#6a737d',
    text: '#24292e',
    textSecondary: '#586069',
    flux: '#0366d6',
    fluxFill: 'rgba(3, 102, 214, 0.1)',
    model: '#e36209',
    modelFill: 'rgba(227, 98, 9, 0.15)',
    event: '#28a745',
    eventFill: 'rgba(40, 167, 69, 0.1)',
    bls: '#6f42c1',
    blsFill: 'rgba(111, 66, 193, 0.1)',
    blsPeak: '#d73a49',
    uncertainty: 'rgba(3, 102, 214, 0.2)',
    crosshair: 'rgba(36, 41, 46, 0.3)',
    tooltipBg: 'rgba(255, 255, 255, 0.95)',
    tooltipBorder: '#d1d5da',
    panelBg: '#f6f8fa',
    panelBorder: '#d1d5da',
    highlight: '#b08800',
  },
};

const PADDING = { top: 40, right: 20, bottom: 50, left: 60 };
const BLS_HEIGHT_RATIO = 0.35;
const PHASE_HEIGHT_RATIO = 0.35;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

function formatNumber(n: number, decimals = 4): string {
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2);
  return n.toFixed(decimals);
}

function formatTime(t: number): string {
  if (Math.abs(t) > 1000) return t.toFixed(2);
  return t.toFixed(4);
}

function computeBlsSpectrum(
  time: number[],
  flux: number[],
  periods: number[],
): BlsPeak[] {
  const peaks: BlsPeak[] = [];
  const n = time.length;
  if (n < 10 || periods.length === 0) return peaks;

  for (const p of periods) {
    if (p <= 0 || !isFinite(p)) continue;

    const phases = time.map((t) => ((t % p) + p) % p);

    let bestPower = -Infinity;
    let bestDepth = 0;
    let bestDuration = 0;

    const durationFractions = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3];
    for (const df of durationFractions) {
      const dur = df * p;
      let inTransit = 0;
      let outTransit = 0;
      let sumIn = 0;
      let sumOut = 0;

      for (let i = 0; i < n; i++) {
        const ph = phases[i];
        const dist = Math.min(ph, p - ph);
        if (dist < dur / 2) {
          inTransit++;
          sumIn += flux[i];
        } else {
          outTransit++;
          sumOut += flux[i];
        }
      }

      if (inTransit < 3 || outTransit < 3) continue;

      const meanIn = sumIn / inTransit;
      const meanOut = sumOut / outTransit;
      const depth = meanOut - meanIn;

      let varIn = 0;
      let varOut = 0;
      for (let i = 0; i < n; i++) {
        const ph = phases[i];
        const dist = Math.min(ph, p - ph);
        if (dist < dur / 2) {
          varIn += (flux[i] - meanIn) * (flux[i] - meanIn);
        } else {
          varOut += (flux[i] - meanOut) * (flux[i] - meanOut);
        }
      }
      varIn /= inTransit;
      varOut /= outTransit;

      const totalVar = varIn + varOut;
      if (totalVar < 1e-12) continue;

      const power = (depth * depth * inTransit) / totalVar;
      if (power > bestPower) {
        bestPower = power;
        bestDepth = depth;
        bestDuration = dur;
      }
    }

    if (bestPower > 0) {
      peaks.push({
        period: p,
        power: bestPower,
        depth: bestDepth,
        duration: bestDuration,
      });
    }
  }

  return peaks;
}

function computeTransitModel(
  time: number[],
  period: number,
  t0: number,
  duration: number,
  depth: number,
  ingress: number,
  egress: