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

export interface LightCurvePoint {
  /** Time in days (MJD or BJD) */
  t: number;
  /** Flux / magnitude value */
  f: number;
  /** Uncertainty on f (1-sigma) */
  err?: number;
  /** Optional quality flag (0 = good, 1 = marginal, 2 = bad) */
  q?: number;
}

export interface LightCurveDetrenderViewProps {
  /** Raw light-curve data */
  data: LightCurvePoint[];
  /** Title displayed in the header */
  title?: string;
  /** X-axis label */
  xLabel?: string;
  /** Y-axis label */
  yLabel?: string;
  /** Whether to show error bars */
  showErrors?: boolean;
  /** Whether to show a linear-fit trend line */
  showTrend?: boolean;
  /** Whether to show a moving-average smoothing overlay */
  showSmoothing?: boolean;
  /** Smoothing window size (in data points) */
  smoothingWindow?: number;
  /** Color of the data points */
  pointColor?: string;
  /** Color of the trend line */
  trendColor?: string;
  /** Color of the smoothing line */
  smoothingColor?: string;
  /** Color of error bars */
  errorColor?: string;
  /** Background color of the plot area */
  backgroundColor?: string;
  /** Grid line color */
  gridColor?: string;
  /** Axis / tick text color */
  textColor?: string;
  /** Minimum width in px (responsive) */
  minWidth?: number;
  /** Height in px */
  height?: number;
  /** Callback when a point is clicked */
  onPointClick?: (point: LightCurvePoint, index: number) => void;
  /** Callback when the user hovers over a point */
  onPointHover?: (point: LightCurvePoint | null, index: number | null) => void;
  /** Callback when the user drags to zoom */
  onZoom?: (xMin: number, xMax: number) => void;
  /** Callback when the user double-clicks to reset zoom */
  onResetZoom?: () => void;
  /** Additional CSS class */
  className?: string;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

interface PlotRegion {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

interface Viewport {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

interface HoverState {
  index: number;
  x: number;
  y: number;
}

/** Compute a simple linear regression (least-squares) over the data. */
function linearFit(data: LightCurvePoint[]): { slope: number; intercept: number } {
  const n = data.length;
  if (n < 2) return { slope: 0, intercept: data.length ? data[0].f : 0 };
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumXX = 0;
  for (const d of data) {
    sumX += d.t;
    sumY += d.f;
    sumXY += d.t * d.f;
    sumXX += d.t * d.t;
  }
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-12) return { slope: 0, intercept: sumY / n };
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

/** Compute a centered moving average. */
function movingAverage(data: LightCurvePoint[], window: number): LightCurvePoint[] {
  const half = Math.floor(window / 2);
  const result: LightCurvePoint[] = [];
  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(data.length - 1, i + half); j++) {
      sum += data[j].f;
      count++;
    }
    result.push({ t: data[i].t, f: sum / count });
  }
  return result;
}

/** Generate "nice" tick values for an axis. */
function niceTicks(min: number, max: number, targetCount: number): number[] {
  if (min === max) return [min];
  const range = max - min;
  const roughStep = range / targetCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const residual = roughStep / magnitude;
  let niceStep: number;
  if (residual <= 1.5) niceStep = 1 * magnitude;
  else if (residual <= 3) niceStep = 2 * magnitude;
  else if (residual <= 7) niceStep = 5 * magnitude;
  else niceStep = 10 * magnitude;

  const start = Math.ceil(min / niceStep) * niceStep;
  const ticks: number[] = [];
  for (let v = start; v <= max + niceStep * 0.001; v += niceStep) {
    ticks.push(Number(v.toFixed(10)));
  }
  return ticks;
}

/** Format a number for axis labels. */
function formatTick(value: number): string {
  if (Math.abs(value) >= 1e6 || (Math.abs(value) < 1e-3 && value !== 0)) {
    return value.toExponential(1);
  }
  if (Number.isInteger(value)) return value.toString();
  return value.toFixed(3).replace(/\.?0+$/, '');
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const LightCurveDetrenderView: React.FC<LightCurveDetrenderViewProps> = ({
  data,
  title = 'Light Curve',
  xLabel = 'Time (days)',
  yLabel = 'Flux',
  showErrors = true,
  showTrend = false,
  showSmoothing = false,
  smoothingWindow = 11,
  pointColor = '#4fc3f7',
  trendColor = '#ff7043',
  smoothingColor = '#ab47bc',
  errorColor = 'rgba(79, 195, 247, 0.45)',
  backgroundColor = '#0d1117',
  gridColor = 'rgba(255, 255, 255, 0.08)',
  textColor = '#c9d1d9',
  minWidth = 600,
  height = 420,
  onPointClick,
  onPointHover,
  onZoom,
  onResetZoom,
  className,
}) => {
  // --- refs ---------------------------------------------------------------
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const [canvasWidth, setCanvasWidth] = useState(minWidth);

  // --- state ---------------------------------------------------------------
  const [hover, setHover] = useState<HoverState | null>(null);
  const [viewport, setViewport] = useState<Viewport | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const dragEndRef = useRef<{ x: number; y: number } | null>(null);

  // --- derived data --------------------------------------------------------
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => a.t - b.t);
  }, [data]);

  const fullExtent = useMemo<Viewport>(() => {
    if (sortedData.length === 0) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
    let xMin = Infinity,
      xMax = -Infinity,
      yMin = Infinity,
      yMax = -Infinity;
    for (const d of sortedData) {
      if (d.t < xMin) xMin = d.t;