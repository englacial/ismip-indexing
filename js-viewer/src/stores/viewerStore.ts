import { create } from "zustand";
import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";
import { registerCodecs } from "../utils/codecs";

// Register numcodecs codecs at module load time
registerCodecs();

// ISMIP6 icechunk store URL
// Use proxy in development to avoid CORS issues
const ICECHUNK_URL = import.meta.env.DEV
  ? "/gcs-proxy/ismip6-icechunk/12-07-2025/"
  : "https://ismip6-icechunk.s3.us-west-2.amazonaws.com/combined-variables-v3/";

// Panel represents a single visualization panel
export interface Panel {
  id: string;
  selectedModel: string | null;
  selectedExperiment: string | null;
  currentData: Float32Array | null;
  dataShape: number[] | null;
  isLoading: boolean;
  error: string | null;
  maxTimeIndex: number;
}

interface ViewerState {
  // Connection state
  isInitializing: boolean;
  initError: string | null;
  store: IcechunkStore | null;

  // Available data
  models: string[];
  experiments: Map<string, string[]>;
  variables: string[];

  // Panels
  panels: Panel[];
  activePanelId: string | null;

  // Shared settings (apply to all panels)
  selectedVariable: string | null;
  timeIndex: number;
  colormap: string;
  vmin: number;
  vmax: number;
  autoRange: boolean;

  // Shared view state for linked zoom/pan
  viewState: {
    target: [number, number, number];
    zoom: number;
  } | null;

  // Shared hover state for cross-panel comparison
  hoverGridPosition: { gridX: number; gridY: number } | null;
  hoveredPanelId: string | null;

  // Actions
  initialize: () => Promise<void>;
  addPanel: () => void;
  removePanel: (panelId: string) => void;
  setActivePanel: (panelId: string) => void;
  setPanelModel: (panelId: string, model: string) => void;
  setPanelExperiment: (panelId: string, experiment: string) => void;
  setSelectedVariable: (variable: string) => void;
  setTimeIndex: (index: number) => void;
  setColormap: (colormap: string) => void;
  setColorRange: (vmin: number, vmax: number) => void;
  setAutoRange: (auto: boolean) => void;
  setViewState: (viewState: { target: [number, number, number]; zoom: number }) => void;
  setHoverGridPosition: (position: { gridX: number; gridY: number } | null, panelId?: string | null) => void;
  getValueAtGridPosition: (panelId: string, gridX: number, gridY: number) => number | null;
  loadPanelData: (panelId: string) => Promise<void>;
  loadAllPanels: () => Promise<void>;
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

let panelIdCounter = 0;
function generatePanelId(): string {
  return `panel-${++panelIdCounter}`;
}

function createEmptyPanel(): Panel {
  return {
    id: generatePanelId(),
    selectedModel: null,
    selectedExperiment: null,
    currentData: null,
    dataShape: null,
    isLoading: false,
    error: null,
    maxTimeIndex: 0,
  };
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  // Initial state
  isInitializing: false,
  initError: null,
  store: null,
  models: [],
  experiments: new Map(),
  variables: ISMIP6_VARIABLES,

  // Start with one empty panel
  panels: [createEmptyPanel()],
  activePanelId: null,

  // Shared settings
  selectedVariable: "lithk",
  timeIndex: 0,
  colormap: "viridis",
  vmin: 0,
  vmax: 4000,
  autoRange: true,
  viewState: null,
  hoverGridPosition: null,
  hoveredPanelId: null,

  initialize: async () => {
    set({ isInitializing: true, initError: null });
    try {
      // Open the icechunk store
      // In dev mode, route virtual chunk URLs through the proxy
      const virtualUrlTransformer = import.meta.env.DEV
        ? (url: string) => {
            // Transform gs://ismip6/... to /ismip6-proxy/...
            if (url.startsWith("gs://ismip6/")) {
              return url.replace("gs://ismip6/", "/ismip6-proxy/");
            }
            return url;
          }
        : undefined;

      const store = await IcechunkStore.open(ICECHUNK_URL, {
        ref: "main",
        virtualUrlTransformer,
      });

      // List models from the store (top-level groups)
      const models = store.listChildren("");

      // Build experiments map by querying each model's subgroups
      const experiments = new Map<string, string[]>();
      for (const model of models) {
        const modelExps = store.listChildren(model);
        experiments.set(model, modelExps);
      }

      // Get first model's first experiment for the initial panel
      const firstModel = models.length > 0 ? models[0] : null;
      const firstModelExps = firstModel ? experiments.get(firstModel) : null;
      const firstExp = firstModelExps && firstModelExps.length > 0 ? firstModelExps[0] : null;

      // Initialize first panel with defaults
      const initialPanel = createEmptyPanel();
      initialPanel.selectedModel = firstModel;
      initialPanel.selectedExperiment = firstExp;

      set({
        store,
        models,
        experiments,
        panels: [initialPanel],
        activePanelId: initialPanel.id,
        isInitializing: false,
      });
    } catch (err) {
      console.error("Failed to initialize:", err);
      set({
        initError: err instanceof Error ? err.message : "Failed to initialize",
        isInitializing: false,
      });
    }
  },

  addPanel: () => {
    const { panels, models, experiments } = get();
    const newPanel = createEmptyPanel();

    // Initialize with first available model/experiment
    if (models.length > 0) {
      newPanel.selectedModel = models[0];
      const modelExps = experiments.get(models[0]);
      if (modelExps && modelExps.length > 0) {
        newPanel.selectedExperiment = modelExps[0];
      }
    }

    set({
      panels: [...panels, newPanel],
      activePanelId: newPanel.id,
    });
  },

  removePanel: (panelId: string) => {
    const { panels, activePanelId } = get();
    if (panels.length <= 1) return; // Keep at least one panel

    const newPanels = panels.filter((p) => p.id !== panelId);
    const newActiveId =
      activePanelId === panelId
        ? newPanels[0]?.id || null
        : activePanelId;

    set({ panels: newPanels, activePanelId: newActiveId });
  },

  setActivePanel: (panelId: string) => {
    set({ activePanelId: panelId });
  },

  setPanelModel: (panelId: string, model: string) => {
    const { panels, experiments } = get();
    const newPanels = panels.map((p) => {
      if (p.id !== panelId) return p;

      const modelExps = experiments.get(model) || [];
      const currentExp = p.selectedExperiment;
      // Reset experiment if not available for this model
      const newExp = modelExps.includes(currentExp || "")
        ? currentExp
        : modelExps[0] || null;

      return {
        ...p,
        selectedModel: model,
        selectedExperiment: newExp,
      };
    });
    set({ panels: newPanels });
  },

  setPanelExperiment: (panelId: string, experiment: string) => {
    const { panels } = get();
    const newPanels = panels.map((p) =>
      p.id === panelId ? { ...p, selectedExperiment: experiment } : p
    );
    set({ panels: newPanels });
  },

  setSelectedVariable: (variable: string) => {
    set({ selectedVariable: variable });
  },

  setTimeIndex: (index: number) => {
    // Find max time index across all panels
    const maxTime = Math.max(...get().panels.map((p) => p.maxTimeIndex), 0);
    set({ timeIndex: Math.max(0, Math.min(index, maxTime)) });
  },

  setColormap: (colormap: string) => {
    set({ colormap });
  },

  setColorRange: (vmin: number, vmax: number) => {
    set({ vmin, vmax, autoRange: false });
  },

  setAutoRange: (auto: boolean) => {
    set({ autoRange: auto });
  },

  setViewState: (viewState) => {
    set({ viewState });
  },

  setHoverGridPosition: (position, panelId = null) => {
    set({ hoverGridPosition: position, hoveredPanelId: panelId });
  },

  getValueAtGridPosition: (panelId: string, gridX: number, gridY: number) => {
    const { panels } = get();
    const panel = panels.find((p) => p.id === panelId);
    if (!panel?.currentData || !panel?.dataShape) return null;

    const [, width] = panel.dataShape;
    const idx = gridY * width + gridX;
    const value = panel.currentData[idx];

    if (isNaN(value) || !isFinite(value) || value > 10000) return null;
    return value;
  },

  loadPanelData: async (panelId: string) => {
    const { store, panels, selectedVariable, timeIndex } = get();
    const panel = panels.find((p) => p.id === panelId);

    if (!store || !panel || !panel.selectedModel || !panel.selectedExperiment || !selectedVariable) {
      return;
    }

    // Update panel loading state
    set({
      panels: panels.map((p) =>
        p.id === panelId ? { ...p, isLoading: true, error: null } : p
      ),
    });

    try {
      // Build the path to the variable array
      const arrayPath = `${panel.selectedModel}/${panel.selectedExperiment}/${selectedVariable}/${selectedVariable}`;
      console.log(`[Panel ${panelId}] Loading: ${arrayPath}`);

      // Get a store resolved to this path
      const arrayStore = store.resolve(arrayPath);

      // Open the zarr array
      const arr = await zarr.open(arrayStore, { kind: "array" });

      // Get array shape and determine time dimension
      const shape = arr.shape;
      console.log(`[Panel ${panelId}] Array shape: ${shape}`);

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
      console.log(`[Panel ${panelId}] Zarr result shape:`, result.shape);

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
      let min = Infinity,
        max = -Infinity,
        nonZeroCount = 0,
        nanCount = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (isNaN(v)) {
          nanCount++;
          continue;
        }
        if (v !== 0) nonZeroCount++;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      console.log(
        `[Panel ${panelId}] Loaded ${data.length} values, min=${min}, max=${max}, nonZero=${nonZeroCount}, NaN=${nanCount}`
      );

      // Update panel with loaded data
      set({
        panels: get().panels.map((p) =>
          p.id === panelId
            ? { ...p, currentData: data, dataShape, maxTimeIndex: maxTime, isLoading: false }
            : p
        ),
      });

      // Auto-range if enabled (compute across all loaded panels)
      if (get().autoRange) {
        computeAutoRange(get);
      }
    } catch (err) {
      console.error(`[Panel ${panelId}] Failed to load data:`, err);
      set({
        panels: get().panels.map((p) =>
          p.id === panelId
            ? {
                ...p,
                error: err instanceof Error ? err.message : "Failed to load data",
                isLoading: false,
              }
            : p
        ),
      });
    }
  },

  loadAllPanels: async () => {
    const { panels, loadPanelData } = get();
    // Load all panels in parallel
    await Promise.all(panels.map((p) => loadPanelData(p.id)));
  },
}));

// Helper to compute auto range across all loaded panels
function computeAutoRange(get: () => ViewerState) {
  const { panels } = get();
  const MAX_VALID_VALUE = 10000;
  const allValidValues: number[] = [];

  for (const panel of panels) {
    if (!panel.currentData) continue;
    for (let i = 0; i < panel.currentData.length; i++) {
      const v = panel.currentData[i];
      if (!isNaN(v) && isFinite(v) && v > 0 && v < MAX_VALID_VALUE) {
        allValidValues.push(v);
      }
    }
  }

  if (allValidValues.length > 0) {
    allValidValues.sort((a, b) => a - b);
    const p5 = allValidValues[Math.floor(allValidValues.length * 0.05)];
    const p95 = allValidValues[Math.floor(allValidValues.length * 0.95)];
    console.log(`[autoRange] Combined p5=${p5}, p95=${p95} from ${allValidValues.length} values`);
    useViewerStore.setState({ vmin: p5, vmax: p95 });
  }
}
