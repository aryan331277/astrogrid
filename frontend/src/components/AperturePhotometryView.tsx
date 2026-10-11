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

interface ApertureSource {
  id: string;
  x: number;
  y: number;
  flux: number;
  fwhm: number;
  ra: number;
  dec: number;
}

interface ApertureRing {
  radius: number;
  label: string;
  color: string;
}

interface AperturePhotometryViewProps {
  sources?: ApertureSource[];
  apertureRings?: ApertureRing[];
  backgroundNoise?: number;
  backgroundLevel?: number;
  imageWidth?: number;
  imageHeight?: number;
  onSourceSelect?: (source: ApertureSource | null) => void;
  onApertureChange?: (radii: number[]) => void;
  className?: string;
}

interface PhotometryResult {
  sourceId: string;
  apertureFlux: number;
  skyFlux: number;
  netFlux: number;
  error: number;
  snr: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_SOURCES: ApertureSource[] = [
  { id: 'src-001', x: 200, y: 180, flux: 12000, fwhm: 8, ra: 12.5, dec: -45.2 },
  { id: 'src-002', x: 450, y: 320, flux: 8500, fwhm: 6, ra: 12.8, dec: -44.9 },
  { id: 'src-003', x: 620, y: 150, flux: 5200, fwhm: 10, ra: 13.1, dec: -45.5 },
  { id: 'src-004', x: 350, y: 480, flux: 3100, fwhm: 5, ra: 12.6, dec: -44.6 },
  { id: 'src-005', x: 780, y: 400, flux: 1800, fwhm: 7, ra: 13.4, dec: -44.8 },
];

const DEFAULT_RINGS: ApertureRing[] = [
  { radius: 10, label: 'Inner', color: '#ff6b6b' },
  { radius: 20, label: 'Outer', color: '#4ecdc4' },
  { radius: 35, label: 'Sky', color: '#a78bfa' },
];

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

function gaussian2D(
  x: number,
  y: number,
  cx: number,
  cy: number,
  sigma: number
): number {
  const dx = x - cx;
  const dy = y - cy;
  const r2 = dx * dx + dy * dy;
  const sigma2 = sigma * sigma;
  return Math.exp(-r2 / (2 * sigma2));
}

function computeApertureFlux(
  source: ApertureSource,
  radius: number,
  imageWidth: number,
  imageHeight: number,
  backgroundNoise: number,
  backgroundLevel: number
): { flux: number; error: number } {
  const sigma = source.fwhm / 2.355;
  let totalFlux = 0;
  let pixelCount = 0;

  const xMin = Math.max(0, Math.floor(source.x - radius));
  const xMax = Math.min(imageWidth - 1, Math.ceil(source.x + radius));
  const yMin = Math.max(0, Math.floor(source.y - radius));
  const yMax = Math.min(imageHeight - 1, Math.ceil(source.y + radius));

  for (let y = yMin; y <= yMax; y++) {
    for (let x = xMin; x <= xMax; x++) {
      const dx = x - source.x;
      const dy = y - source.y;
      if (dx * dx + dy * dy <= radius * radius) {
        const g = gaussian2D(x, y, source.x, source.y, sigma);
        totalFlux += source.flux * g + backgroundLevel + backgroundNoise * 0.1;
        pixelCount++;
      }
    }
  }

  const error = Math.sqrt(totalFlux + backgroundNoise * pixelCount);
  return { flux: totalFlux, error };
}

function computeSkyFlux(
  source: ApertureSource,
  innerRadius: number,
  outerRadius: number,
  imageWidth: number,
  imageHeight: number,
  backgroundNoise: number,
  backgroundLevel: number
): { flux: number; error: number } {
  let totalFlux = 0;
  let pixelCount = 0;

  const xMin = Math.max(0, Math.floor(source.x - outerRadius));
  const xMax = Math.min(imageWidth - 1, Math.ceil(source.x + outerRadius));
  const yMin = Math.max(0, Math.floor(source.y - outerRadius));
  const yMax = Math.min(imageHeight - 1, Math.ceil(source.y + outerRadius));

  for (let y = yMin; y <= yMax; y++) {
    for (let x = xMin; x <= xMax; x++) {
      const dx = x - source.x;
      const dy = y - source.y;
      const r2 = dx * dx + dy * dy;
      if (r2 > innerRadius * innerRadius && r2 <= outerRadius * outerRadius) {
        totalFlux += backgroundLevel + backgroundNoise * 0.1;
        pixelCount++;
      }
    }
  }

  const error = Math.sqrt(backgroundNoise * pixelCount);
  return { flux: totalFlux, error };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface SourceMarkerProps {
  source: ApertureSource;
  selected: boolean;
  onClick: () => void;
}

const SourceMarker: React.FC<SourceMarkerProps> = ({
  source,
  selected,
  onClick,
}) => {
  return (
    <g
      onClick={onClick}
      style={{ cursor: 'pointer' }}
      data-testid={`source-marker-${source.id}`}
    >
      <circle
        cx={source.x}
        cy={source.y}
        r={selected ? 8 : 5}
        fill={selected ? '#ffd700' : 'rgba(255,255,255,0.8)'}
        stroke={selected ? '#fff' : 'rgba(255,255,255,0.5)'}
        strokeWidth={selected ? 2 : 1}
      />
      <text
        x={source.x + 10}
        y={source.y - 8}
        fill="rgba(255,255,255,0.7)"
        fontSize="10"
        fontFamily="monospace"
      >
        {source.id}
      </text>
    </g>
  );
};

interface ApertureRingOverlayProps {
  cx: number;
  cy: number;
  rings: ApertureRing[];
}

const ApertureRingOverlay: React.FC<ApertureRingOverlayProps> = ({
  cx,
  cy,
  rings,
}) => {
  return (
    <g>
      {rings.map((ring) => (
        <circle
          key={ring.label}
          cx={cx}
          cy={cy}
          r={ring.radius}
          fill="none"
          stroke={ring.color}
          strokeWidth={1.5}
          strokeDasharray="4 2"
          opacity={0.8}
        />
      ))}
    </g>
  );
};

interface StatsPanelProps {
  results: PhotometryResult[];
  selectedSource: ApertureSource | null;
}

const StatsPanel: React.FC<StatsPanelProps> = ({
  results,
  selectedSource,
}) => {
  return (
    <div
      style={{
        position: 'absolute',
        top: 10,
        right: 10,