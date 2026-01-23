import { create } from "zustand";
import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";

// ISMIP6 icechunk store URL
// Use proxy in development to avoid CORS issues
const ICECHUNK_URL = import.meta.env.DEV
  ? "/gcs-proxy/ismip6-icechunk/12-07-2025/"
  : "https://storage.googleapis.com/ismip6-icechunk/12-07-2025/";

interface ViewerState {
  // Connection state
  isLoading: boolean;
  error: string | null;
  store: IcechunkStore | null;

  // Available data
  models: string[];
  experiments: Map<string, string[]>;
  variables: string[];

  // Selection state
  selectedModel: string | null;
  selectedExperiment: string | null;
  selectedVariable: string | null;
  timeIndex: number;
  maxTimeIndex: number;

  // Visualization settings
  colormap: string;
  vmin: number;
  vmax: number;
  autoRange: boolean;

  // Data
  currentData: Float32Array | null;
  dataShape: number[] | null;

  // Actions
  initialize: () => Promise<void>;
  setSelectedModel: (model: string) => void;
  setSelectedExperiment: (experiment: string) => void;
  setSelectedVariable: (variable: string) => void;
  setTimeIndex: (index: number) => void;
  setColormap: (colormap: string) => void;
  setColorRange: (vmin: number, vmax: number) => void;
  setAutoRange: (auto: boolean) => void;
  loadData: () => Promise<void>;
}

// Common ISMIP6 variables (2D gridded)
const ISMIP6_VARIABLES = [
  "lithk", // ice thickness
  "orog", // surface elevation
  "base", // base elevation
  "topg", // bedrock elevation
  "velsurf_mag", // surface velocity magnitude
  "acabf", // surface mass balance flux
  "libmassbfgr", // basal mass balance flux grounded
  "libmassbffl", // basal mass balance flux floating
  "licalvf", // calving flux
  "ligroundf", // grounding line flux
  "sftgif", // grounded ice fraction
  "sftgrf", // grounded ice area
  "sftflf", // floating ice fraction
  "dlithkdt", // ice thickness change rate
];

// Common experiments
const COMMON_EXPERIMENTS = [
  "ctrl_proj_std",
  "hist_std",
  "exp01",
  "exp02",
  "exp03",
  "exp04",
  "exp05",
  "exp06",
  "exp07",
  "exp08",
  "exp09",
  "exp10",
  "exp11",
  "exp12",
  "exp13",
];

export const useViewerStore = create<ViewerState>((set, get) => ({
  // Initial state
  isLoading: false,
  error: null,
  store: null,
  models: [],
  experiments: new Map(),
  variables: ISMIP6_VARIABLES,
  selectedModel: null,
  selectedExperiment: null,
  selectedVariable: "lithk",
  timeIndex: 0,
  maxTimeIndex: 0,
  colormap: "viridis",
  vmin: 0,
  vmax: 4000,
  autoRange: true,
  currentData: null,
  dataShape: null,

  initialize: async () => {
    set({ isLoading: true, error: null });
    try {
      // Open the icechunk store
      const store = await IcechunkStore.open(ICECHUNK_URL, { ref: "main" });

      // List models from the store (top-level groups)
      const models = store.listChildren("");

      // Build experiments map by querying each model's subgroups
      const experiments = new Map<string, string[]>();
      for (const model of models) {
        const modelExps = store.listChildren(model);
        experiments.set(model, modelExps);
      }

      // Get first model's first experiment
      const firstModel = models.length > 0 ? models[0] : null;
      const firstModelExps = firstModel ? experiments.get(firstModel) : null;
      const firstExp = firstModelExps && firstModelExps.length > 0 ? firstModelExps[0] : null;

      set({
        store,
        models,
        experiments,
        selectedModel: firstModel,
        selectedExperiment: firstExp,
        isLoading: false,
      });
    } catch (err) {
      console.error("Failed to initialize:", err);
      set({
        error: err instanceof Error ? err.message : "Failed to initialize",
        isLoading: false,
      });
    }
  },

  setSelectedModel: (model) => {
    const experiments = get().experiments.get(model) || [];
    const currentExp = get().selectedExperiment;
    // Reset experiment if not available for this model
    const newExp = experiments.includes(currentExp || "")
      ? currentExp
      : (experiments[0] || null);
    set({ selectedModel: model, selectedExperiment: newExp });
  },

  setSelectedExperiment: (experiment) => {
    set({ selectedExperiment: experiment });
  },

  setSelectedVariable: (variable) => {
    set({ selectedVariable: variable });
  },

  setTimeIndex: (index) => {
    set({ timeIndex: Math.max(0, Math.min(index, get().maxTimeIndex)) });
  },

  setColormap: (colormap) => {
    set({ colormap });
  },

  setColorRange: (vmin, vmax) => {
    set({ vmin, vmax, autoRange: false });
  },

  setAutoRange: (auto) => {
    set({ autoRange: auto });
  },

  loadData: async () => {
    const { store, selectedModel, selectedExperiment, selectedVariable, timeIndex } =
      get();
    if (!store || !selectedModel || !selectedExperiment || !selectedVariable) {
      return;
    }

    set({ isLoading: true, error: null });

    try {
      // Build the path to the variable array
      // Structure is: model/experiment/variable_group/variable_array
      const arrayPath = `${selectedModel}/${selectedExperiment}/${selectedVariable}/${selectedVariable}`;
      console.log(`Loading: ${arrayPath}`);

      // Get a store resolved to this path
      const arrayStore = store.resolve(arrayPath);

      // Open the zarr array
      const arr = await zarr.open(arrayStore, { kind: "array" });

      // Get array shape and determine time dimension
      const shape = arr.shape;
      console.log(`Array shape: ${shape}`);

      let maxTime = 0;
      let dataShape: number[];

      // ISMIP6 arrays are typically (time, y, x) or (y, x)
      if (shape.length === 3) {
        maxTime = shape[0] - 1;
        dataShape = [shape[1], shape[2]];
      } else if (shape.length === 2) {
        dataShape = [shape[0], shape[1]];
      } else {
        throw new Error(`Unexpected array shape: ${shape}`);
      }

      set({ maxTimeIndex: maxTime });

      // Determine slice based on dimensions
      let slice: (number | null)[];
      if (shape.length === 3) {
        // (time, y, x) - get single time slice
        const t = Math.min(timeIndex, maxTime);
        slice = [t, null, null];
      } else {
        // (y, x) - get all
        slice = [null, null];
      }

      // Fetch the data
      const result = await zarr.get(arr, slice);
      console.log("Zarr result:", result);
      console.log("Result data type:", result.data.constructor.name);
      console.log("Result shape:", result.shape);

      // Convert to Float32Array
      let data: Float32Array;
      if (result.data instanceof Float32Array) {
        data = result.data;
      } else if (result.data instanceof Float64Array) {
        data = new Float32Array(result.data);
      } else if (ArrayBuffer.isView(result.data)) {
        data = new Float32Array(result.data as ArrayLike<number>);
      } else {
        throw new Error(`Unexpected data type: ${typeof result.data}`);
      }

      // Debug: check data statistics
      let min = Infinity, max = -Infinity, nonZeroCount = 0, nanCount = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (isNaN(v)) { nanCount++; continue; }
        if (v !== 0) nonZeroCount++;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      console.log(`Loaded ${data.length} values, min=${min}, max=${max}, nonZero=${nonZeroCount}, NaN=${nanCount}`);

      set({
        currentData: data,
        dataShape,
        isLoading: false,
      });

      // Auto-range if enabled
      if (get().autoRange) {
        // Filter valid values and compute percentiles
        const validValues: number[] = [];
        for (let i = 0; i < data.length; i++) {
          const v = data[i];
          if (!isNaN(v) && isFinite(v) && v !== 0) {
            validValues.push(v);
          }
        }

        if (validValues.length > 0) {
          validValues.sort((a, b) => a - b);
          const p5 = validValues[Math.floor(validValues.length * 0.05)];
          const p95 = validValues[Math.floor(validValues.length * 0.95)];
          set({ vmin: p5, vmax: p95 });
        }
      }
    } catch (err) {
      console.error("Failed to load data:", err);
      set({
        error: err instanceof Error ? err.message : "Failed to load data",
        isLoading: false,
      });
    }
  },
}));
