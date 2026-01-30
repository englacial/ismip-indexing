// Colormap utilities for data visualization

export type RGB = [number, number, number];

// Colormap definitions - each is an array of RGB values [0-255]
const COLORMAPS: Record<string, RGB[]> = {
  viridis: [
    [68, 1, 84],
    [72, 35, 116],
    [64, 67, 135],
    [52, 94, 141],
    [41, 120, 142],
    [32, 144, 140],
    [34, 167, 132],
    [68, 190, 112],
    [121, 209, 81],
    [189, 222, 38],
    [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135],
    [75, 3, 161],
    [125, 3, 168],
    [168, 34, 150],
    [203, 70, 121],
    [229, 107, 93],
    [248, 148, 65],
    [253, 195, 40],
    [240, 249, 33],
  ],
  inferno: [
    [0, 0, 4],
    [40, 11, 84],
    [101, 21, 110],
    [159, 42, 99],
    [212, 72, 66],
    [245, 125, 21],
    [250, 193, 39],
    [252, 255, 164],
  ],
  magma: [
    [0, 0, 4],
    [28, 16, 68],
    [79, 18, 123],
    [129, 37, 129],
    [181, 54, 122],
    [229, 80, 100],
    [251, 135, 97],
    [254, 194, 135],
    [252, 253, 191],
  ],
  cividis: [
    [0, 32, 77],
    [0, 67, 106],
    [54, 97, 109],
    [111, 120, 113],
    [161, 142, 110],
    [208, 166, 96],
    [253, 199, 67],
  ],
  turbo: [
    [48, 18, 59],
    [86, 91, 213],
    [36, 162, 255],
    [35, 220, 186],
    [114, 252, 100],
    [199, 237, 50],
    [255, 186, 43],
    [244, 105, 37],
    [181, 36, 21],
    [122, 4, 3],
  ],
  coolwarm: [
    [59, 76, 192],
    [98, 130, 234],
    [141, 176, 254],
    [184, 208, 249],
    [221, 221, 221],
    [245, 196, 173],
    [244, 154, 123],
    [222, 96, 77],
    [180, 4, 38],
  ],
  RdBu: [
    [103, 0, 31],
    [178, 24, 43],
    [214, 96, 77],
    [244, 165, 130],
    [253, 219, 199],
    [247, 247, 247],
    [209, 229, 240],
    [146, 197, 222],
    [67, 147, 195],
    [33, 102, 172],
    [5, 48, 97],
  ],
  gray: [
    [0, 0, 0],
    [255, 255, 255],
  ],
};

export const COLORMAP_NAMES = Object.keys(COLORMAPS);

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function interpolateColor(colors: RGB[], t: number): RGB {
  const clampedT = Math.max(0, Math.min(1, t));
  const n = colors.length - 1;
  const idx = clampedT * n;
  const i = Math.floor(idx);
  const frac = idx - i;

  if (i >= n) return colors[n];
  if (i < 0) return colors[0];

  const c1 = colors[i];
  const c2 = colors[i + 1];

  return [
    Math.round(lerp(c1[0], c2[0], frac)),
    Math.round(lerp(c1[1], c2[1], frac)),
    Math.round(lerp(c1[2], c2[2], frac)),
  ];
}

export function applyColormap(
  value: number,
  vmin: number,
  vmax: number,
  colormapName: string
): RGB {
  const colors = COLORMAPS[colormapName] || COLORMAPS.viridis;
  const range = vmax - vmin;
  const t = range !== 0 ? (value - vmin) / range : 0.5;
  return interpolateColor(colors, t);
}

export function createColormapTexture(
  colormapName: string,
  size: number = 256
): Uint8Array {
  const colors = COLORMAPS[colormapName] || COLORMAPS.viridis;
  const data = new Uint8Array(size * 4);

  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    const [r, g, b] = interpolateColor(colors, t);
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }

  return data;
}

export function dataToRGBA(
  data: Float32Array,
  width: number,
  height: number,
  vmin: number,
  vmax: number,
  colormapName: string,
  fillValue?: number | null,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const colors = COLORMAPS[colormapName] || COLORMAPS.viridis;

  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    const pixelIdx = i * 4;

    // Treat NaN, Infinity, and fill values as transparent
    // Use approximate comparison for fill value because float32 data
    // loses precision vs float64 metadata fill value
    const isFill = fillValue != null
      ? Math.abs(value) > 1e10 || Math.abs(value - fillValue) < Math.abs(fillValue) * 1e-6
      : Math.abs(value) > 1e10;

    if (isNaN(value) || !isFinite(value) || isFill) {
      rgba[pixelIdx] = 0;
      rgba[pixelIdx + 1] = 0;
      rgba[pixelIdx + 2] = 0;
      rgba[pixelIdx + 3] = 0;
    } else {
      const range = vmax - vmin;
      const t = range !== 0 ? (value - vmin) / range : 0.5;
      const [r, g, b] = interpolateColor(colors, t);
      rgba[pixelIdx] = r;
      rgba[pixelIdx + 1] = g;
      rgba[pixelIdx + 2] = b;
      rgba[pixelIdx + 3] = 255;
    }
  }

  return rgba;
}
