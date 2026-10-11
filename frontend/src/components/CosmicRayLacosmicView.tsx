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

type Vec2 = { x: number; y: number };

type Particle = {
  id: number;
  pos: Vec2;
  vel: Vec2;
  energy: number; // GeV
  charge: number;
  mass: number; // proton masses
  age: number;
  maxAge: number;
  trail: Vec2[];
  hue: number;
};

type Event = {
  id: number;
  pos: Vec2;
  time: number;
  magnitude: number;
  kind: 'primary' | 'shower' | 'muon';
};

type SimConfig = {
  particleCount: number;
  speed: number;
  fieldStrength: number;
  turbulence: number;
  showTrails: boolean;
  showField: boolean;
  showEvents: boolean;
  paused: boolean;
};

type Stats = {
  fps: number;
  activeParticles: number;
  eventsThisSecond: number;
  meanEnergy: number;
  maxEnergy: number;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TRAIL_LENGTH = 24;
const EVENT_LIFETIME = 1.4; // seconds
const MAX_EVENTS = 400;
const FIELD_GRID = 24;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const rand = (min: number, max: number): number =>
  min + Math.random() * (max - min);

const clamp = (v: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, v));

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

const energyToHue = (e: number): number => {
  // Map log10(energy) 0..12 to hue 200 (blue) -> 0 (red)
  const le = clamp(Math.log10(Math.max(e, 1)), 0, 12);
  return lerp(200, 0, le / 12);
};

const energyToRadius = (e: number): number => {
  const le = clamp(Math.log10(Math.max(e, 1)), 0, 12);
  return lerp(1.2, 4.5, le / 12);
};

// Simple pseudo-noise for field turbulence
const noise2 = (x: number, y: number, t: number): number => {
  return (
    Math.sin(x * 0.7 + t * 0.6) * 0.5 +
    Math.sin(y * 0.9 - t * 0.4) * 0.3 +
    Math.sin((x + y) * 0.5 + t * 0.8) * 0.2
  );
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const StatChip: React.FC<{ label: string; value: string; accent?: string }> = ({
  label,
  value,
  accent = '#7dd3fc',
}) => (
  <div
    style={{
      display: 'flex',
      flexDirection: 'column',
      padding: '6px 10px',
      borderRadius: 8,
      background: 'rgba(15, 23, 42, 0.75)',
      border: '1px solid rgba(148, 163, 184, 0.2)',
      minWidth: 90,
    }}
  >
    <span
      style={{
        fontSize: 10,
        textTransform: 'uppercase',
        letterSpacing: 1,
        color: '#94a3b8',
      }}
    >
      {label}
    </span>
    <span
      style={{
        fontSize: 16,
        fontWeight: 700,
        color: accent,
        fontVariantNumeric: 'tabular-nums',
      }}
    >
      {value}
    </span>
  </div>
);

const Toggle: React.FC<{
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}> = ({ label, checked, onChange }) => (
  <label
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      cursor: 'pointer',
      userSelect: 'none',
      fontSize: 13,
      color: '#cbd5e1',
    }}
  >
    <span
      onClick={() => onChange(!checked)}
      style={{
        width: 34,
        height: 18,
        borderRadius: 9,
        background: checked ? '#0ea5e9' : '#334155',
        position: 'relative',
        transition: 'background 0.2s',
        flexShrink: 0,
      }}
    >
      <span
        style={{
          position: 'absolute',
          top: 2,
          left: checked ? 18 : 2,
          width: 14,
          height: 14,
          borderRadius: '50%',
          background: '#f8fafc',
          transition: 'left 0.2s',
        }}
      />
    </span>
    {label}
  </label>
);

const Slider: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}> = ({ label, value, min, max, step, onChange, format }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 12,
        color: '#94a3b8',
      }}
    >
      <span>{label}</span>
      <span style={{ color: '#e2e8f0', fontVariantNumeric: 'tabular-nums' }}>
        {format ? format(value) : value.toFixed(2)}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      style={{
        width: '100%',
        accentColor: '#0ea5e9',
        cursor: 'pointer',
      }}
    />
  </div>
);

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

const CosmicRayLacosmicView: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [config, setConfig] = useState<SimConfig>({
    particleCount: 200,
    speed: 1.0,
    fieldStrength: 0.5,
    turbulence: 0.3,
    showTrails: true,
    showField: true,
    showEvents: true,
    paused: false,
  });

  const [stats, setStats] = useState<Stats>({
    fps: 0,
    activeParticles: 0,
    eventsThisSecond: 0,
    meanEnergy: 0,
    maxEnergy: 0,
  });

  const [hoveredParticle, setHoveredParticle] = useState<Particle | null>(null);
  const [selectedParticle, setSelectedParticle] = useState<Particle | null>(null);
  const [showInfoPanel, setShowInfoPanel] = useState(true);

  // Mutable refs for animation loop
  const particlesRef = useRef<Particle[]>([]);
  const eventsRef = useRef<Event[]>([]);
  const configRef = useRef<SimConfig>(config);
  const mouseRef = useRef<Vec2>({ x: -1, y: -1 });
  const sizeRef = useRef<{ w: number; h: number }>({ w: 800, h: 600 });
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const fpsCounterRef = useRef<{ frames: number; last: