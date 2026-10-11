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

interface WcsCard {
  key: string;
  value: string;
  comment: string;
  raw: string;
}

interface WcsAxis {
  index: number;
  ctype: string;
  crval: number;
  crpix: number;
  cdelt: number;
  cunit: string;
  ctypeLabel: string;
}

interface ParsedHeader {
  cards: WcsCard[];
  wcsAxes: WcsAxis[];
  naxis: number;
  naxis1: number;
  naxis2: number;
  date: string;
  telescope: string;
  instrument: string;
  filter: string;
  exposure: number;
  hasWcs: boolean;
}

interface ParserState {
  status: 'idle' | 'parsing' | 'done' | 'error';
  error?: string;
}

interface ViewMode {
  id: 'grid' | 'axes' | 'raw';
  label: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VIEW_MODES: ViewMode[] = [
  { id: 'grid', label: 'Grid' },
  { id: 'axes', label: 'Axes' },
  { id: 'raw', label: 'Raw' },
];

const CTYPE_LABELS: Record<string, string> = {
  RA---TAN: 'Right Ascension (TAN)',
  DEC--TAN: 'Declination (TAN)',
  RA---SIN: 'Right Ascension (SIN)',
  DEC--SIN: 'Declination (SIN)',
  HPLN-TAN: 'Helioprojective Longitude',
  HPLT-TAN: 'Helioprojective Latitude',
  FREQ--LIN: 'Frequency (Linear)',
  STOKES-LIN: 'Stokes Parameter',
  TIME--LIN: 'Time (Linear)',
  VEL--LIN: 'Velocity (Linear)',
  ICRS--TAN: 'ICRS (TAN)',
  GAL---TAN: 'Galactic (TAN)',
  ECLI--TAN: 'Ecliptic (TAN)',
  EQJ---TAN: 'Equatorial J2000 (TAN)',
  EQE---TAN: 'Equatorial Epoch (TAN)',
};

const SAMPLE_HEADER = `SIMPLE  =                    T / conforms to FITS standard
BITPIX  =                  -32 / array data type
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                 2048 / length of data axis 1
NAXIS2  =                 2048 / length of data axis 2
DATE    = '2024-03-15T12:30:45.123' / UTC date of observation
TELESCOP= 'HST       ' / telescope name
INSTRUME= 'WFC3      ' / instrument
FILTER  = 'F606W     ' / filter name
EXPTIME =              300.0 / exposure time (s)
CTYPE1  = 'RA---TAN ' / coordinate type for axis 1
CRVAL1  =           180.12345 / reference value for axis 1
CRPIX1  =              1024.0 / reference pixel for axis 1
CDELT1  =          -0.0000988 / coordinate increment for axis 1
CUNIT1  = 'deg     ' / coordinate units for axis 1
CTYPE2  = 'DEC--TAN ' / coordinate type for axis 2
CRVAL2  =           -39.67890 / reference value for axis 2
CRPIX2  =              1024.0 / reference pixel for axis 2
CDELT2  =           0.0000988 / coordinate increment for axis 2
CUNIT2  = 'deg     ' / coordinate units for axis 2
WCSAXES =                    2 / number of WCS axes
END
`;

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

function parseFitsHeader(text: string): ParsedHeader {
  const lines = text.split('\n').filter((l) => l.trim().length > 0);
  const cards: WcsCard[] = [];
  const header: Record<string, string> = {};

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === 'END') break;

    // FITS 80-char card format
    const key = trimmed.substring(0, 8).trim();
    const rest = trimmed.substring(8);

    if (rest.startsWith('=')) {
      const valuePart = rest.substring(1).trim();
      let value = '';
      let comment = '';

      // Check for comment after value
      const slashIdx = valuePart.indexOf('/');
      if (slashIdx > 0) {
        value = valuePart.substring(0, slashIdx).trim();
        comment = valuePart.substring(slashIdx + 1).trim();
      } else {
        value = valuePart;
      }

      // Strip quotes
      if (value.startsWith("'") && value.endsWith("'")) {
        value = value.substring(1, value.length - 1);
      }

      cards.push({ key, value, comment, raw: trimmed });
      header[key] = value;
    }
  }

  const naxis = parseInt(header['NAXIS'] || '0', 10);
  const naxis1 = parseInt(header['NAXIS1'] || '0', 10);
  const naxis2 = parseInt(header['NAXIS2'] || '0', 10);

  const wcsAxes: WcsAxis[] = [];
  for (let i = 1; i <= naxis; i++) {
    const ctype = header[`CTYPE${i}`] || '';
    const crval = parseFloat(header[`CRVAL${i}`] || '0');
    const crpix = parseFloat(header[`CRPIX${i}`] || '0');
    const cdelt = parseFloat(header[`CDELT${i}`] || '1');
    const cunit = header[`CUNIT${i}`] || '';
    const ctypeLabel = CTYPE_LABELS[ctype] || ctype || 'Unknown';

    wcsAxes.push({
      index: i,
      ctype,
      crval,
      crpix,
      cdelt,
      cunit,
      ctypeLabel,
    });
  }

  return {
    cards,
    wcsAxes,
    naxis,
    naxis1,
    naxis2,
    date: header['DATE'] || '',
    telescope: header['TELESCOP'] || '',
    instrument: header['INSTRUME'] || '',
    filter: header['FILTER'] || '',
    exposure: parseFloat(header['EXPTIME'] || '0'),
    hasWcs: wcsAxes.length > 0,
  };
}

// ---------------------------------------------------------------------------
// Canvas Drawing Helpers
// ---------------------------------------------------------------------------

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  parsed: ParsedHeader,
  hoverX: number,
  hoverY: number,
): void {
  ctx.clearRect(0, 0, width, height);

  // Background
  ctx.fillStyle = '#0a0e1a';
  ctx.fillRect(0, 0, width, height);

  const { naxis1, naxis2, wcsAxes } = parsed;
  if (naxis1 === 0 || naxis2 === 0) {
    ctx.fillStyle = '#4a5568';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No image dimensions found', width / 2, height / 2);
    return;
  }

  const margin = 60;
  const plotW = width - margin * 2;
  const plotH = height - margin * 2;

  // Draw grid lines
  const gridStep = 100;
  ctx.strokeStyle = 'rgba(99, 179, 237, 0.15)';
  ctx.lineWidth = 1;

  for (let x = 0; x <= naxis1; x += gridStep) {
    const px = margin + (x / naxis1) * plotW;
    ctx.beginPath