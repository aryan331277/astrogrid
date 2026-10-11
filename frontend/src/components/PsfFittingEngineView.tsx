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

interface PsfSource {
  id: string;
  x: number; // pixel coordinate
  y: number;
  flux: number; // relative flux (0..1)
  fwhm: number; // full width at half max (pixels)
  ellipticity: number; // 0..1
  positionAngle: number; // radians
  isTransient: boolean;
}

interface PsfModelParams {
  kernel: 'gaussian' | 'moffat' | 'lorentzian';
  fwhm: number;
  ellipticity: number;
  positionAngle: number;
  truncation: number; // truncation radius in FWHM
  background: number;
}

interface FitResult {
  chi2: number;
  dof: number;
  reducedChi2: number;
  residualRms: number;
  iterations: number;
  converged: boolean;
  residuals: Float32Array;
}

interface EngineState {
  isRunning: boolean;
  progress: number; // 0..1
  iteration: number;
  maxIterations: number;
  log: string[];
}

interface PsfFittingEngineViewProps {
  sources?: PsfSource[];
  onFitComplete?: (result: FitResult) => void;
  onSourceSelect?: (source: PsfSource | null) => void;
  className?: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_SIZE = 512;
const PIXEL_SIZE = 8; // canvas pixels per simulated detector pixel
const GRID_SIZE = CANVAS_SIZE / PIXEL_SIZE; // 64x64 simulated pixels
const MAX_SOURCES = 20;
const DEFAULT_FWHM = 3.0;
const DEFAULT_ELLIPTICITY = 0.1;
const DEFAULT_POSITION_ANGLE = 0.0;
const DEFAULT_TRUNCATION = 4.0;
const DEFAULT_BACKGROUND = 0.02;
const MAX_ITERATIONS = 50;
const CONVERGENCE_THRESHOLD = 1e-6;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function generateGaussianKernel(
  size: number,
  fwhm: number,
  ellipticity: number,
  positionAngle: number,
  truncation: number
): Float32Array {
  const kernel = new Float32Array(size * size);
  const half = size / 2;
  const sigma = fwhm / (2 * Math.sqrt(2 * Math.log(2)));
  const cosPA = Math.cos(positionAngle);
  const sinPA = Math.sin(positionAngle);
  const truncRadius = truncation * fwhm;

  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const dx = i - half;
      const dy = j - half;
      // Rotate coordinates
      const rx = dx * cosPA + dy * sinPA;
      const ry = -dx * sinPA + dy * cosPA;
      // Apply ellipticity
      const ex = rx;
      const ey = ry / (1 - ellipticity);
      const r2 = ex * ex + ey * ey;
      const r = Math.sqrt(r2);
      if (r > truncRadius) {
        kernel[j * size + i] = 0;
      } else {
        kernel[j * size + i] = Math.exp(-r2 / (2 * sigma * sigma));
      }
    }
  }

  // Normalize
  let sum = 0;
  for (let i = 0; i < kernel.length; i++) sum += kernel[i];
  if (sum > 0) {
    for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  }

  return kernel;
}

function generateMoffatKernel(
  size: number,
  fwhm: number,
  ellipticity: number,
  positionAngle: number,
  truncation: number
): Float32Array {
  const kernel = new Float32Array(size * size);
  const half = size / 2;
  const alpha = 2.5; // Moffat parameter
  const beta = fwhm / (2 * Math.sqrt(Math.sqrt(2 * (2 ** (1 / alpha) - 1))));
  const cosPA = Math.cos(positionAngle);
  const sinPA = Math.sin(positionAngle);
  const truncRadius = truncation * fwhm;

  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const dx = i - half;
      const dy = j - half;
      const rx = dx * cosPA + dy * sinPA;
      const ry = -dx * sinPA + dy * cosPA;
      const ex = rx;
      const ey = ry / (1 - ellipticity);
      const r2 = ex * ex + ey * ey;
      const r = Math.sqrt(r2);
      if (r > truncRadius) {
        kernel[j * size + i] = 0;
      } else {
        kernel[j * size + i] = Math.pow(1 + r2 / (beta * beta), -alpha);
      }
    }
  }

  let sum = 0;
  for (let i = 0; i < kernel.length; i++) sum += kernel[i];
  if (sum > 0) {
    for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  }

  return kernel;
}

function generateLorentzianKernel(
  size: number,
  fwhm: number,
  ellipticity: number,
  positionAngle: number,
  truncation: number
): Float32Array {
  const kernel = new Float32Array(size * size);
  const half = size / 2;
  const gamma = fwhm / 2;
  const cosPA = Math.cos(positionAngle);
  const sinPA = Math.sin(positionAngle);
  const truncRadius = truncation * fwhm;

  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const dx = i - half;
      const dy = j - half;
      const rx = dx * cosPA + dy * sinPA;
      const ry = -dx * sinPA + dy * cosPA;
      const ex = rx;
      const ey = ry / (1 - ellipticity);
      const r2 = ex * ex + ey * ey;
      const r = Math.sqrt(r2);
      if (r > truncRadius) {
        kernel[j * size + i] = 0;
      } else {
        kernel[j * size + i] = (gamma * gamma) / (r2 + gamma * gamma);
      }
    }
  }

  let sum = 0;
  for (let i = 0; i < kernel.length; i++) sum += kernel[i];
  if (sum > 0) {
    for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  }

  return kernel;
}

function generateKernel(
  size: number,
  params: PsfModelParams
): Float32Array {
  switch (params.kernel) {
    case 'gaussian':
      return generateGaussianKernel(
        size,
        params.fwhm,
        params.ellipticity,
        params.positionAngle,
        params.truncation
      );
    case 'moffat':
      return generateMoffatKernel(
        size,
        params.fwhm,
        params.ellipticity,
        params.positionAngle,
        params.truncation
      );
    case 'lorentzian':
      return generateLorentzianKernel(
        size,
        params.fwhm,
        params.ellipticity,
        params.positionAngle,
        params.truncation
      );
    default:
      return generateGaussianKernel(
        size,
        params.fwhm,
        params.ellipticity,
        params.positionAngle,
        params.truncation
      );
  }
}

function createDefaultSources(): PsfSource[] {
  const sources: PsfSource[] = [];
  const count = 8