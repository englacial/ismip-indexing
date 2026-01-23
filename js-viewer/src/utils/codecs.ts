/**
 * Register additional codecs with zarrita's registry.
 *
 * The ISMIP6 data uses numcodecs.zlib and numcodecs.shuffle codecs
 * which need to be registered with their full names.
 */

import { registry } from "zarrita";
import Zlib from "numcodecs/zlib";

/**
 * Simple shuffle codec implementation.
 *
 * Shuffle is a filter that reorders bytes to group similar values together,
 * which typically improves compression ratios. It's commonly used with
 * NetCDF data before zlib compression.
 */
class ShuffleCodec {
  readonly kind = "bytes_to_bytes" as const;
  private elementSize: number;

  constructor(config: { elementsize?: number } = {}) {
    this.elementSize = config.elementsize ?? 4; // Default to 4 bytes (float32)
    console.log(`[ShuffleCodec] Created with elementSize=${this.elementSize}`);
  }

  static fromConfig(
    config: { elementsize?: number },
    meta: { codecs?: Array<{ name: string; configuration?: unknown }> },
  ): ShuffleCodec {
    console.log(`[ShuffleCodec.fromConfig] config=`, config);
    console.log(`[ShuffleCodec.fromConfig] Full codec chain:`, meta?.codecs);
    return new ShuffleCodec(config);
  }

  async encode(data: Uint8Array): Promise<Uint8Array> {
    const n = data.length;
    const elemSize = this.elementSize;
    const numElems = Math.floor(n / elemSize);
    const result = new Uint8Array(n);

    // Shuffle: interleave bytes by position within element
    for (let i = 0; i < numElems; i++) {
      for (let j = 0; j < elemSize; j++) {
        result[j * numElems + i] = data[i * elemSize + j];
      }
    }

    // Copy any remaining bytes
    const remainder = n % elemSize;
    for (let i = 0; i < remainder; i++) {
      result[numElems * elemSize + i] = data[numElems * elemSize + i];
    }

    return result;
  }

  async decode(data: Uint8Array): Promise<Uint8Array> {
    const n = data.length;
    const elemSize = this.elementSize;
    const numElems = Math.floor(n / elemSize);
    const result = new Uint8Array(n);

    console.log(
      `[ShuffleCodec.decode] n=${n}, elemSize=${elemSize}, numElems=${numElems}`,
    );

    // Debug: show first 20 input bytes
    console.log(
      `[ShuffleCodec.decode] First 20 input bytes:`,
      Array.from(data.slice(0, 20)),
    );

    // Debug: show what first float would be WITHOUT shuffle (just interpreting bytes directly)
    const directView = new DataView(
      data.buffer,
      data.byteOffset,
      data.byteLength,
    );
    const directVals = [];
    for (let i = 0; i < Math.min(5, numElems); i++) {
      directVals.push(directView.getFloat32(i * 4, true));
    }
    console.log(
      `[ShuffleCodec.decode] First 5 float32 WITHOUT unshuffle:`,
      directVals,
    );

    // Unshuffle: reverse the interleaving
    for (let i = 0; i < numElems; i++) {
      for (let j = 0; j < elemSize; j++) {
        result[i * elemSize + j] = data[j * numElems + i];
      }
    }

    // Copy any remaining bytes
    const remainder = n % elemSize;
    for (let i = 0; i < remainder; i++) {
      result[numElems * elemSize + i] = data[numElems * elemSize + i];
    }

    // Debug: show first few values as float32
    const view = new DataView(result.buffer, result.byteOffset, result.byteLength);
    const firstVals = [];
    for (let i = 0; i < Math.min(5, numElems); i++) {
      firstVals.push(view.getFloat32(i * 4, true)); // little-endian
    }
    console.log(`[ShuffleCodec.decode] First 5 float32 values:`, firstVals);

    return result;
  }
}

/**
 * Wrapper around Zlib codec to add debugging.
 */
class DebugZlibCodec {
  readonly kind = "bytes_to_bytes" as const;
  private inner: InstanceType<typeof Zlib>;

  constructor(level = 1) {
    this.inner = new Zlib(level);
  }

  static fromConfig(config: { level?: number }): DebugZlibCodec {
    console.log(`[DebugZlibCodec.fromConfig] config=`, config);
    return new DebugZlibCodec(config?.level ?? 1);
  }

  async encode(data: Uint8Array): Promise<Uint8Array> {
    return this.inner.encode(data);
  }

  async decode(data: Uint8Array): Promise<Uint8Array> {
    console.log(`[DebugZlibCodec.decode] Input size: ${data.length}`);
    console.log(
      `[DebugZlibCodec.decode] First 20 input bytes:`,
      Array.from(data.slice(0, 20)),
    );

    const result = this.inner.decode(data);

    console.log(`[DebugZlibCodec.decode] Output size: ${result.length}`);
    console.log(
      `[DebugZlibCodec.decode] First 20 output bytes:`,
      Array.from(result.slice(0, 20)),
    );

    // Check bytes at different positions (each section is one byte-position of all elements)
    const numElems = Math.floor(result.length / 4);
    console.log(`[DebugZlibCodec.decode] Shuffled sections (numElems=${numElems}):`);
    console.log(
      `  - byte0 section (0-20):`,
      Array.from(result.slice(0, 20)),
    );
    console.log(
      `  - byte1 section (${numElems}-${numElems + 20}):`,
      Array.from(result.slice(numElems, numElems + 20)),
    );
    console.log(
      `  - byte2 section (${numElems * 2}-${numElems * 2 + 20}):`,
      Array.from(result.slice(numElems * 2, numElems * 2 + 20)),
    );
    console.log(
      `  - byte3 section (${numElems * 3}-${numElems * 3 + 20}):`,
      Array.from(result.slice(numElems * 3, numElems * 3 + 20)),
    );

    return result;
  }
}

/**
 * Register codecs with zarrita.
 * Call this once at app startup before loading any zarr data.
 */
export function registerCodecs(): void {
  // Register numcodecs.zlib with debug wrapper
  registry.set("numcodecs.zlib", () => DebugZlibCodec);

  // Register numcodecs.shuffle
  registry.set("numcodecs.shuffle", () => ShuffleCodec);

  console.log("[codecs] Registered numcodecs.zlib and numcodecs.shuffle");
}
