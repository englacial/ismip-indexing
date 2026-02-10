import { create } from "zustand";
import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";
import { registerCodecs } from "../utils/codecs";
import { parseUrlParams, EmbedConfig } from "../utils/urlParams";
import { parseTimeUnits, decodeTimeArray, yearFromLabel, findIndexForYear, yearRange } from "../utils/cftime";

// Register numcodecs codecs at module load time
registerCodecs();

// Default ISMIP6 icechunk store URL (source.coop, unified store)
// Use proxy in development to avoid CORS issues
const DEFAULT_STORE_URL = import.meta.env.DEV
  ? "/s3-proxy/"
  : "https://data.source.coop/englacial/ismip6/icechunk-ais/";

const DEFAULT_GROUP_PATH = "combined";

// Grid configuration derived from coordinate arrays or URL params
export interface GridConfig {
  width: number;
  height: number;
  cellSize: number;
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

// Default grid for ISMIP6 Antarctic (EPSG:3031)
const DEFAULT_GRID: GridConfig = {
  width: 761,
  height: 761,
  cellSize: 8000,
  xMin: -3040000,
  yMin: -3040000,
  xMax: -3040000 + 761 * 8000,
  yMax: -3040000 + 761 * 8000,
};

// Metadata for the currently loaded variable (from zarr array attrs)
export interface VariableMetadata {
  units: string | null;
  standardName: string | null;
}

// Metadata for a group (model/experiment) from zarr group attrs
export interface GroupMetadata {
  title: string | null;
  institution: string | null;
  source: string | null;
  contact: string | null;
  references: string | null;
  comment: string | null;
  [key: string]: string | null;
}

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
  groupMetadata: GroupMetadata | null;
  timeLabels: string[] | null;
  resolvedTimeIndex: number | null;
  allNaN: boolean;
}

// Known coordinate/dimension variable names to exclude from data variables
const COORD_NAMES = new Set([
  "x", "y", "lat", "lon", "latitude", "longitude",
  "time", "t", "bnds", "bounds", "time_bnds", "time_bounds",
  "x_bnds", "y_bnds", "lat_bnds", "lon_bnds",
  "mapping", "crs", "spatial_ref",
]);

export type DataView = "combined" | "state" | "flux";

interface ViewerState {
  // Connection state
  isInitializing: boolean;
  initError: string | null;
  store: IcechunkStore | null;

  // Embed configuration (null = standalone mode)
  embedConfig: EmbedConfig | null;

  // Data view (top-level group in unified store)
  dataView: DataView;

  // Grid configuration (derived from store or URL params)
  gridConfig: GridConfig;

  // Available data
  models: string[];
  experiments: Map<string, string[]>;
  variables: string[];

  // Hierarchy depth (how many group levels before arrays)
  hierarchyDepth: number;

  // Fill value for masking (from zarr metadata or default)
  fillValue: number | null;

  // Variable metadata (units, standard_name) for the current variable
  variableMetadata: VariableMetadata | null;

  // Panels
  panels: Panel[];
  activePanelId: string | null;

  // Shared settings (apply to all panels)
  selectedVariable: string | null;
  timeIndex: number;
  targetYear: number | null; // target year for cross-model time alignment
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
  setDataView: (view: DataView) => Promise<void>;
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
    groupMetadata: null,
    timeLabels: null,
    resolvedTimeIndex: 0,
    allNaN: false,
  };
}

/**
 * Walk the store hierarchy to discover structure.
 * Returns { models, experiments, variables, hierarchyDepth, arrayGroupPath }
 *
 * Supports:
 * - 2-level: model/experiment/variable_arrays  (hierarchyDepth=2)
 * - 1-level: experiment/variable_arrays         (hierarchyDepth=1)
 * - 0-level: variable_arrays at root            (hierarchyDepth=0)
 */
async function discoverHierarchy(
  store: IcechunkStore,
  groupPathOverride?: string,
): Promise<{
  models: string[];
  experiments: Map<string, string[]>;
  variables: string[];
  hierarchyDepth: number;
}> {
  const basePath = groupPathOverride || "";

  // Check if the base path itself contains arrays (depth 0)
  const rootChildren = store.listChildren(basePath);

  // Try to open first child as array to determine if these are arrays or groups
  if (rootChildren.length === 0) {
    return { models: [], experiments: new Map(), variables: [], hierarchyDepth: 0 };
  }

  // Test first child: is it a group or an array?
  const testPath = basePath ? `${basePath}/${rootChildren[0]}` : rootChildren[0];
  let firstChildIsGroup = false;
  try {
    const testChildren = store.listChildren(testPath);
    // If it has children, it's a group
    firstChildIsGroup = testChildren.length > 0;
  } catch {
    // If listChildren fails, it's likely an array
    firstChildIsGroup = false;
  }

  if (!firstChildIsGroup) {
    // Depth 0: arrays at root/basePath
    const variables = rootChildren.filter((name) => !COORD_NAMES.has(name.toLowerCase()));
    return { models: [], experiments: new Map(), variables, hierarchyDepth: 0 };
  }

  // First level is groups. Check if second level is also groups or arrays.
  const firstGroup = rootChildren[0];
  const firstGroupPath = basePath ? `${basePath}/${firstGroup}` : firstGroup;
  const secondLevelChildren = store.listChildren(firstGroupPath);

  if (secondLevelChildren.length === 0) {
    return { models: [], experiments: new Map(), variables: [], hierarchyDepth: 1 };
  }

  const testPath2 = `${firstGroupPath}/${secondLevelChildren[0]}`;
  let secondChildIsGroup = false;
  try {
    const testChildren2 = store.listChildren(testPath2);
    secondChildIsGroup = testChildren2.length > 0;
  } catch {
    secondChildIsGroup = false;
  }

  if (!secondChildIsGroup) {
    // Depth 1: one level of groups, then arrays
    // Treat rootChildren as "experiments" (no model level)
    const experiments = new Map<string, string[]>();
    // Discover variables from first group's arrays
    const variables = secondLevelChildren.filter((name) => !COORD_NAMES.has(name.toLowerCase()));
    // No model grouping — put all under a single dummy model key
    experiments.set("_root", rootChildren);
    return { models: [], experiments, variables, hierarchyDepth: 1 };
  }

  // Depth 2: model/experiment/arrays (ISMIP6 pattern)
  const models = rootChildren;
  const experiments = new Map<string, string[]>();
  for (const model of models) {
    const modelPath = basePath ? `${basePath}/${model}` : model;
    const modelExps = store.listChildren(modelPath);
    experiments.set(model, modelExps);
  }

  // Discover variables from first model's first experiment
  // Filter to only spatial arrays (2D or 3D), excluding 1D time-series
  const firstModel = models[0];
  const firstModelExps = experiments.get(firstModel) || [];
  let variables: string[] = [];
  if (firstModelExps.length > 0) {
    const samplePath = basePath
      ? `${basePath}/${firstModel}/${firstModelExps[0]}`
      : `${firstModel}/${firstModelExps[0]}`;
    const arrayNames = store.listChildren(samplePath)
      .filter((name) => !COORD_NAMES.has(name.toLowerCase()));

    // Check dimensionality of each candidate variable
    for (const name of arrayNames) {
      try {
        const arrStore = store.resolve(`${samplePath}/${name}`);
        const arr = await zarr.open(arrStore, { kind: "array" });
        if (arr.shape.length >= 2) {
          variables.push(name);
        }
      } catch {
        // Skip arrays we can't open
      }
    }
  }

  return { models, experiments, variables, hierarchyDepth: 2 };
}

/**
 * Try to read grid config from coordinate arrays (x, y) in the store.
 * Falls back to URL param overrides, then to defaults.
 */
async function discoverGridConfig(
  store: IcechunkStore,
  sampleGroupPath: string,
  embedConfig: EmbedConfig | null,
): Promise<GridConfig> {
  // Try reading x and y coordinate arrays
  try {
    const xStore = store.resolve(`${sampleGroupPath}/x`);
    const yStore = store.resolve(`${sampleGroupPath}/y`);

    const xArr = await zarr.open(xStore, { kind: "array" });
    const yArr = await zarr.open(yStore, { kind: "array" });

    const xData = await zarr.get(xArr);
    const yData = await zarr.get(yArr);

    const xValues = new Float64Array(xData.data as unknown as ArrayBuffer);
    const yValues = new Float64Array(yData.data as unknown as ArrayBuffer);

    if (xValues.length > 1 && yValues.length > 1) {
      const xMin = xValues[0];
      const yMin = yValues[0];
      const cellSizeX = Math.abs(xValues[1] - xValues[0]);
      const cellSizeY = Math.abs(yValues[1] - yValues[0]);
      const cellSize = Math.max(cellSizeX, cellSizeY);
      const width = xValues.length;
      const height = yValues.length;

      // Validate: cellSize must be positive and coordinates must be finite numbers
      if (cellSize > 0 && isFinite(xMin) && isFinite(yMin) && isFinite(cellSize)) {
        const grid: GridConfig = {
          width,
          height,
          cellSize,
          xMin,
          yMin,
          xMax: xMin + width * cellSize,
          yMax: yMin + height * cellSize,
        };
        console.log("[gridConfig] Derived from coordinate arrays:", grid);
        return grid;
      }
      console.warn("[gridConfig] Invalid coordinate values (cellSize=%f, xMin=%f, yMin=%f), falling back", cellSize, xMin, yMin);
    }
  } catch (err) {
    console.warn("[gridConfig] Could not read coordinate arrays, trying fallbacks:", err);
  }

  // Fallback to URL param overrides
  if (embedConfig?.grid_width && embedConfig?.grid_height && embedConfig?.cell_size != null) {
    const width = embedConfig.grid_width;
    const height = embedConfig.grid_height;
    const cellSize = embedConfig.cell_size;
    const xMin = embedConfig.x_min ?? 0;
    const yMin = embedConfig.y_min ?? 0;
    const grid: GridConfig = {
      width,
      height,
      cellSize,
      xMin,
      yMin,
      xMax: xMin + width * cellSize,
      yMax: yMin + height * cellSize,
    };
    console.log("[gridConfig] From URL params:", grid);
    return grid;
  }

  // Final fallback: ISMIP6 defaults
  console.log("[gridConfig] Using ISMIP6 defaults");
  return DEFAULT_GRID;
}

/**
 * Try to read the fill_value from zarr array metadata.
 */
async function discoverFillValue(
  store: IcechunkStore,
  arrayPath: string,
): Promise<number | null> {
  try {
    const arrayStore = store.resolve(arrayPath);
    const arr = await zarr.open(arrayStore, { kind: "array" });
    const attrs = arr.attrs as Record<string, unknown>;
    // zarrita exposes fill_value on the array object
    const fv = (arr as unknown as Record<string, unknown>).fill_value;
    if (typeof fv === "number" && isFinite(fv)) {
      console.log(`[fillValue] From metadata: ${fv}`);
      return fv;
    }
    // Also check _FillValue attribute
    if (typeof attrs?._FillValue === "number") {
      console.log(`[fillValue] From _FillValue attr: ${attrs._FillValue}`);
      return attrs._FillValue;
    }
  } catch (err) {
    console.warn("[fillValue] Could not read fill value:", err);
  }
  return null;
}

/**
 * Read variable-level metadata (units, standard_name) from zarr array attrs.
 */
async function discoverVariableMetadata(
  store: IcechunkStore,
  arrayPath: string,
): Promise<VariableMetadata> {
  try {
    const arrayStore = store.resolve(arrayPath);
    const arr = await zarr.open(arrayStore, { kind: "array" });
    const attrs = arr.attrs as Record<string, unknown>;
    return {
      units: typeof attrs?.units === "string" ? attrs.units : null,
      standardName: typeof attrs?.standard_name === "string" ? attrs.standard_name : null,
    };
  } catch (err) {
    console.warn("[variableMetadata] Could not read:", err);
    return { units: null, standardName: null };
  }
}

/**
 * Read group-level metadata (title, institution, etc.) from zarr group attrs.
 */
async function discoverGroupMetadata(
  store: IcechunkStore,
  groupPath: string,
): Promise<GroupMetadata | null> {
  try {
    const groupStore = store.resolve(groupPath);
    const group = await zarr.open(groupStore, { kind: "group" });
    const attrs = group.attrs as Record<string, unknown>;
    if (!attrs || Object.keys(attrs).length === 0) return null;

    const meta: GroupMetadata = {
      title: null, institution: null, source: null,
      contact: null, references: null, comment: null,
    };
    for (const key of Object.keys(attrs)) {
      const val = attrs[key];
      if (typeof val === "string") {
        const lk = key.toLowerCase();
        if (lk in meta) {
          meta[lk] = val;
        } else {
          meta[key] = val;
        }
      }
    }
    return meta;
  } catch (err) {
    console.warn("[groupMetadata] Could not read:", err);
    return null;
  }
}

/**
 * Read the time coordinate array and decode values to date strings.
 */
async function discoverTimeLabels(
  store: IcechunkStore,
  groupPath: string,
): Promise<string[] | null> {
  try {
    const timeStore = store.resolve(`${groupPath}/time`);
    const timeArr = await zarr.open(timeStore, { kind: "array" });
    const attrs = timeArr.attrs as Record<string, unknown>;

    const units = typeof attrs?.units === "string" ? attrs.units : null;
    const calendar = typeof attrs?.calendar === "string" ? attrs.calendar : null;

    const encoding = parseTimeUnits(units, calendar);

    const timeData = await zarr.get(timeArr);
    const values = new Float64Array(timeData.data as unknown as ArrayBuffer);

    if (encoding) {
      const labels = decodeTimeArray(values, encoding);
      console.log(`[timeLabels] Decoded ${labels.length} time labels, first=${labels[0]}, last=${labels[labels.length - 1]}`);
      return labels;
    }

    // Fallback: if raw values look like years (all finite, in [1000, 3000]),
    // use them directly as synthetic date labels.  This handles time arrays
    // with non-CF units (e.g. bare "years") or missing units attributes.
    if (values.length > 0) {
      let allYearLike = true;
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (!isFinite(v) || v < 1000 || v > 3000) {
          allYearLike = false;
          break;
        }
      }
      if (allYearLike) {
        const labels = Array.from(values, (v) => `${Math.round(v)}-01-01`);
        console.log(`[timeLabels] Fallback: raw values as years, ${labels.length} labels, first=${labels[0]}, last=${labels[labels.length - 1]}`);
        return labels;
      }
    }

    console.warn("[timeLabels] Could not parse time encoding and values don't look like years, units=", units, "calendar=", calendar);
    return null;
  } catch (err) {
    console.warn("[timeLabels] Could not read time array:", err);
    return null;
  }
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  // Initial state
  isInitializing: false,
  initError: null,
  store: null,
  embedConfig: null,
  dataView: "combined" as DataView,
  gridConfig: DEFAULT_GRID,
  models: [],
  experiments: new Map(),
  variables: [],
  hierarchyDepth: 2,
  fillValue: null,
  variableMetadata: null,

  // Start with one empty panel
  panels: [createEmptyPanel()],
  activePanelId: null,

  // Shared settings
  selectedVariable: null,
  timeIndex: 0,
  targetYear: null,
  colormap: "viridis",
  vmin: 0,
  vmax: 4000,
  autoRange: true,
  viewState: null,
  hoverGridPosition: null,
  hoveredPanelId: null,

  initialize: async () => {
    const embedConfig = parseUrlParams();
    const initialDataView: DataView = (embedConfig?.data_view as DataView) || "combined";
    set({ isInitializing: true, initError: null, embedConfig, dataView: initialDataView });
    try {
      // Determine store URL (configurable via embed param)
      const storeUrl = embedConfig?.store_url || DEFAULT_STORE_URL;

      // In dev mode, route virtual chunk URLs through the proxy
      const virtualUrlTransformer = import.meta.env.DEV
        ? (url: string) => {
            if (url.startsWith("s3://us-west-2.opendata.source.coop/englacial/ismip6/")) {
              return url.replace("s3://us-west-2.opendata.source.coop/englacial/ismip6/", "/ismip6-proxy/");
            }
            if (url.startsWith("gs://ismip6/")) {
              return url.replace("gs://ismip6/", "/ismip6-proxy/");
            }
            return url;
          }
        : undefined;

      // Determine store ref: branch, tag, or snapshot ID
      const storeRef = embedConfig?.store_ref || "main";

      // Open store — if ref looks like a snapshot ID (20-char base32), use it directly
      const isSnapshotId = /^[0-9A-Z]{20}$/.test(storeRef);
      const store = await IcechunkStore.open(storeUrl, {
        ...(isSnapshotId ? { snapshot: storeRef } : { ref: storeRef }),
        virtualUrlTransformer,
      });

      // Discover hierarchy (models/experiments/variables)
      const groupPath = embedConfig?.group_path || initialDataView || DEFAULT_GROUP_PATH;
      const { models, experiments, variables, hierarchyDepth } =
        await discoverHierarchy(store, groupPath);

      // Discover grid config from coordinate arrays
      let sampleGroupPath = "";
      if (hierarchyDepth === 2 && models.length > 0) {
        const firstExps = experiments.get(models[0]) || [];
        if (firstExps.length > 0) {
          sampleGroupPath = `${groupPath}/${models[0]}/${firstExps[0]}`;
        }
      } else if (hierarchyDepth === 1) {
        const rootExps = experiments.get("_root") || [];
        sampleGroupPath = `${groupPath}/${rootExps[0]}`;
      } else {
        sampleGroupPath = groupPath;
      }

      const gridConfig = await discoverGridConfig(store, sampleGroupPath, embedConfig);

      // Discover fill value from first available variable
      let fillValue: number | null = null;
      if (variables.length > 0 && sampleGroupPath) {
        fillValue = await discoverFillValue(store, `${sampleGroupPath}/${variables[0]}`);
      }

      // Select initial variable
      const defaultVariable = variables.length > 0 ? variables[0] : null;

      // Build panels from embed config or defaults
      let initialPanels: Panel[];

      if (embedConfig?.panels && embedConfig.panels.length > 0) {
        initialPanels = embedConfig.panels.map((pc) => {
          const p = createEmptyPanel();
          p.selectedModel = pc.model;
          p.selectedExperiment = pc.experiment;
          return p;
        });
      } else if (embedConfig?.model) {
        const p = createEmptyPanel();
        p.selectedModel = embedConfig.model;
        p.selectedExperiment = embedConfig.experiment || (
          experiments.get(embedConfig.model)?.[0] ?? null
        );
        initialPanels = [p];
      } else if (hierarchyDepth === 2) {
        // Don't pre-select model/experiment - let user choose
        initialPanels = [createEmptyPanel()];
      } else {
        // For flat stores, create a panel without model/experiment
        initialPanels = [createEmptyPanel()];
      }

      // Apply embed overrides to shared settings
      const overrides: Partial<ViewerState> = {};
      if (embedConfig?.variable && variables.includes(embedConfig.variable)) {
        overrides.selectedVariable = embedConfig.variable;
      }
      if (embedConfig?.time !== undefined) overrides.timeIndex = embedConfig.time;
      if (embedConfig?.colormap) overrides.colormap = embedConfig.colormap;
      if (embedConfig?.vmin !== undefined) {
        overrides.vmin = embedConfig.vmin;
        overrides.autoRange = false;
      }
      if (embedConfig?.vmax !== undefined) {
        overrides.vmax = embedConfig.vmax;
        overrides.autoRange = false;
      }

      set({
        store,
        models,
        experiments,
        variables,
        hierarchyDepth,
        gridConfig,
        fillValue,
        selectedVariable: defaultVariable,
        panels: initialPanels,
        activePanelId: initialPanels[0]?.id || null,
        isInitializing: false,
        ...overrides,
      });

      // Auto-load if embed config requests it
      if (embedConfig?.autoload) {
        setTimeout(() => get().loadAllPanels(), 0);
      }
    } catch (err) {
      console.error("Failed to initialize:", err);
      set({
        initError: err instanceof Error ? err.message : "Failed to initialize",
        isInitializing: false,
      });
    }
  },

  setDataView: async (view: DataView) => {
    const { store, embedConfig } = get();
    if (!store) return;

    set({ isInitializing: true, dataView: view });

    try {
      const groupPath = embedConfig?.group_path || view;
      const { models, experiments, variables, hierarchyDepth } =
        await discoverHierarchy(store, groupPath);

      // Discover grid config from first sample group
      let sampleGroupPath = "";
      if (hierarchyDepth === 2 && models.length > 0) {
        const firstExps = experiments.get(models[0]) || [];
        if (firstExps.length > 0) {
          sampleGroupPath = `${groupPath}/${models[0]}/${firstExps[0]}`;
        }
      } else if (hierarchyDepth === 1) {
        const rootExps = experiments.get("_root") || [];
        sampleGroupPath = `${groupPath}/${rootExps[0]}`;
      } else {
        sampleGroupPath = groupPath;
      }

      const gridConfig = await discoverGridConfig(store, sampleGroupPath, embedConfig);

      let fillValue: number | null = null;
      if (variables.length > 0 && sampleGroupPath) {
        fillValue = await discoverFillValue(store, `${sampleGroupPath}/${variables[0]}`);
      }

      const defaultVariable = variables.length > 0 ? variables[0] : null;
      const initialPanel = createEmptyPanel();

      set({
        models,
        experiments,
        variables,
        hierarchyDepth,
        gridConfig,
        fillValue,
        selectedVariable: defaultVariable,
        panels: [initialPanel],
        activePanelId: initialPanel.id,
        timeIndex: 0,
        targetYear: null,
        variableMetadata: null,
        isInitializing: false,
      });
    } catch (err) {
      console.error("Failed to switch data view:", err);
      set({
        initError: err instanceof Error ? err.message : "Failed to switch data view",
        isInitializing: false,
      });
    }
  },

  addPanel: () => {
    const { panels, models, experiments } = get();
    const newPanel = createEmptyPanel();

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
    if (panels.length <= 1) return;

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
    const { panels, experiments, hoverGridPosition, hoveredPanelId, selectedVariable, loadPanelData } = get();
    const newPanels = panels.map((p) => {
      if (p.id !== panelId) return p;

      const modelExps = experiments.get(model) || [];
      const currentExp = p.selectedExperiment;
      const newExp = modelExps.includes(currentExp || "")
        ? currentExp
        : modelExps[0] || null;

      // Clear data when model changes
      return {
        ...p,
        selectedModel: model,
        selectedExperiment: newExp,
        currentData: null,
        dataShape: null,
        timeLabels: null,
        groupMetadata: null,
        resolvedTimeIndex: null,
        allNaN: false,
        error: null,
      };
    });

    const updates: Partial<ViewerState> = { panels: newPanels };
    if (hoveredPanelId === panelId && hoverGridPosition) {
      updates.hoverGridPosition = null;
      updates.hoveredPanelId = null;
    }
    set(updates);

    // Auto-load if we have a valid selection and instant_load is enabled
    const updatedPanel = newPanels.find((p) => p.id === panelId);
    const { embedConfig } = get();
    if (embedConfig?.instant_load !== false && model && updatedPanel?.selectedExperiment && selectedVariable) {
      loadPanelData(panelId);
    }
  },

  setPanelExperiment: (panelId: string, experiment: string) => {
    const { panels, hoverGridPosition, hoveredPanelId, selectedVariable, loadPanelData, embedConfig } = get();
    const newPanels = panels.map((p) =>
      p.id === panelId
        ? {
            ...p,
            selectedExperiment: experiment,
            currentData: null,
            dataShape: null,
            timeLabels: null,
            groupMetadata: null,
            resolvedTimeIndex: null,
            allNaN: false,
            error: null,
          }
        : p
    );

    const updates: Partial<ViewerState> = { panels: newPanels };
    if (hoveredPanelId === panelId && hoverGridPosition) {
      updates.hoverGridPosition = null;
      updates.hoveredPanelId = null;
    }
    set(updates);

    // Auto-load if we have a valid selection and instant_load is enabled
    const updatedPanel = newPanels.find((p) => p.id === panelId);
    if (embedConfig?.instant_load !== false && updatedPanel?.selectedModel && experiment && selectedVariable) {
      loadPanelData(panelId);
    }
  },

  setSelectedVariable: (variable: string) => {
    set({ selectedVariable: variable });
  },

  setTimeIndex: (index: number) => {
    // The slider now drives targetYear directly.
    // `index` is the year offset from minYear (slider min=0, max=maxYear-minYear).
    const { panels } = get();
    const range = yearRange(panels.map((p) => p.timeLabels));
    if (!range) {
      set({ timeIndex: index });
      return;
    }
    const targetYear = range.minYear + index;
    // Update each panel's resolvedTimeIndex (null = out of range)
    const updatedPanels = panels.map((p) => {
      if (!p.timeLabels || p.timeLabels.length === 0) {
        // No time labels but has a time dimension — can't resolve year,
        // so mark as out-of-range (loadPanelData will handle properly).
        if (p.maxTimeIndex > 0 && p.resolvedTimeIndex !== null) {
          return { ...p, resolvedTimeIndex: null };
        }
        return p;
      }
      const resolved = findIndexForYear(p.timeLabels, targetYear);
      return resolved !== p.resolvedTimeIndex ? { ...p, resolvedTimeIndex: resolved } : p;
    });
    set({ timeIndex: index, targetYear, panels: updatedPanels });
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
    const { panels, fillValue } = get();
    const panel = panels.find((p) => p.id === panelId);
    if (!panel?.currentData || !panel?.dataShape) return null;

    const [, width] = panel.dataShape;
    const idx = gridY * width + gridX;
    const value = panel.currentData[idx];

    if (isNaN(value) || !isFinite(value)) return null;
    // Use approximate comparison: float32 data loses precision vs float64 fill value
    if (Math.abs(value) > 1e10) return null;
    if (fillValue !== null && Math.abs(value - fillValue) < Math.abs(fillValue) * 1e-6) return null;
    return value;
  },

  loadPanelData: async (panelId: string) => {
    const { store, panels, selectedVariable, timeIndex, targetYear, hierarchyDepth, embedConfig } = get();
    console.log(`[Panel ${panelId}] loadPanelData called: timeIndex=${timeIndex}, targetYear=${targetYear}`);
    const panel = panels.find((p) => p.id === panelId);

    if (!store || !panel || !selectedVariable) {
      return;
    }

    // For depth-2 hierarchy, require model and experiment
    if (hierarchyDepth === 2 && (!panel.selectedModel || !panel.selectedExperiment)) {
      return;
    }

    set({
      panels: panels.map((p) =>
        p.id === panelId ? { ...p, isLoading: true, error: null } : p
      ),
    });

    try {
      // Build path based on hierarchy depth
      let arrayPath: string;
      const basePath = embedConfig?.group_path || get().dataView;

      if (hierarchyDepth === 2) {
        arrayPath = basePath
          ? `${basePath}/${panel.selectedModel}/${panel.selectedExperiment}/${selectedVariable}`
          : `${panel.selectedModel}/${panel.selectedExperiment}/${selectedVariable}`;
      } else if (hierarchyDepth === 1) {
        const group = panel.selectedExperiment || panel.selectedModel || "";
        arrayPath = basePath
          ? `${basePath}/${group}/${selectedVariable}`
          : `${group}/${selectedVariable}`;
      } else {
        arrayPath = basePath
          ? `${basePath}/${selectedVariable}`
          : selectedVariable;
      }

      console.log(`[Panel ${panelId}] Loading: ${arrayPath}`);

      const arrayStore = store.resolve(arrayPath);
      const arr = await zarr.open(arrayStore, { kind: "array" });

      const shape = arr.shape;
      console.log(`[Panel ${panelId}] Array shape: ${shape}`);

      let maxTime = 0;
      let dataShape: number[];

      if (shape.length === 3) {
        maxTime = shape[0] - 1;
        dataShape = [shape[1], shape[2]];
      } else if (shape.length === 2) {
        dataShape = [shape[0], shape[1]];
      } else {
        throw new Error(`Variable "${selectedVariable}" is not a spatial grid (shape: [${shape}]). Only 2D and 3D arrays are supported.`);
      }

      // Discover group-level metadata and time labels BEFORE fetching data
      // so we can resolve the correct per-panel time index by year
      let groupMeta: GroupMetadata | null = null;
      let timeLabels: string[] | null = null;
      if (hierarchyDepth === 2 && panel.selectedModel && panel.selectedExperiment) {
        const groupPath = `${basePath}/${panel.selectedModel}/${panel.selectedExperiment}`;
        groupMeta = await discoverGroupMetadata(store, groupPath);
        if (maxTime > 0) {
          timeLabels = await discoverTimeLabels(store, groupPath);
        }
      }

      // Resolve time index: if we have a target year and this panel has
      // time labels, find the index whose year best matches the target.
      // This aligns panels that have different time ranges.
      let resolvedTimeIndex: number | null = Math.min(timeIndex, maxTime);
      if (targetYear !== null && timeLabels && timeLabels.length > 0) {
        resolvedTimeIndex = findIndexForYear(timeLabels, targetYear);
        console.log(`[Panel ${panelId}] Year-aligned: targetYear=${targetYear}, resolvedIndex=${resolvedTimeIndex}${resolvedTimeIndex !== null ? ` (${timeLabels[resolvedTimeIndex]})` : ' (out of range)'}`);
      } else if (targetYear !== null && !timeLabels && maxTime > 0) {
        // Time dimension exists but labels couldn't be decoded — can't
        // determine if targetYear is in range.  Show out-of-range rather
        // than displaying data from an arbitrary time step.
        console.warn(`[Panel ${panelId}] Time dimension (maxTime=${maxTime}) but no time labels; treating as out-of-range`);
        resolvedTimeIndex = null;
      }

      // If target year is out of this panel's range, store null data
      if (resolvedTimeIndex === null) {
        set({
          panels: get().panels.map((p) =>
            p.id === panelId
              ? { ...p, currentData: null, dataShape: null, maxTimeIndex: maxTime, isLoading: false, groupMetadata: groupMeta, timeLabels, resolvedTimeIndex: null, allNaN: false }
              : p
          ),
        });
        return;
      }

      let slice: (number | null)[];
      if (shape.length === 3) {
        slice = [resolvedTimeIndex, null, null];
      } else {
        slice = [null, null];
      }

      const result = await zarr.get(arr, slice);
      console.log(`[Panel ${panelId}] Zarr result shape:`, result.shape);

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

      // Debug stats
      let min = Infinity,
        max = -Infinity,
        nonZeroCount = 0,
        nanCount = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (isNaN(v)) { nanCount++; continue; }
        if (v !== 0) nonZeroCount++;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const dataAllNaN = nanCount === data.length || (min === Infinity && max === -Infinity);
      console.log(
        `[Panel ${panelId}] Loaded ${data.length} values, min=${min}, max=${max}, nonZero=${nonZeroCount}, NaN=${nanCount}, allNaN=${dataAllNaN}`
      );

      // Re-discover fill value and variable metadata for current variable
      const newFillValue = await discoverFillValue(store, arrayPath);
      const varMeta = await discoverVariableMetadata(store, arrayPath);

      // If targetYear hasn't been set yet, derive it from this panel's time labels
      const updates: Record<string, unknown> = {
        fillValue: newFillValue,
        variableMetadata: varMeta,
        panels: get().panels.map((p) =>
          p.id === panelId
            ? { ...p, currentData: data, dataShape, maxTimeIndex: maxTime, isLoading: false, groupMetadata: groupMeta, timeLabels, resolvedTimeIndex, allNaN: dataAllNaN }
            : p
        ),
      };
      if (get().targetYear === null && resolvedTimeIndex !== null && timeLabels && timeLabels[resolvedTimeIndex]) {
        const yr = yearFromLabel(timeLabels[resolvedTimeIndex]);
        console.log(`[Panel ${panelId}] Setting targetYear=${yr} from label=${timeLabels[resolvedTimeIndex]}`);
        updates.targetYear = yr;
        // Also set timeIndex as year-offset once we know the range
        const allLabels = (updates.panels as Panel[]).map((p: Panel) => p.timeLabels);
        const range = yearRange(allLabels);
        if (range) {
          updates.timeIndex = yr - range.minYear;
        }
      }
      set(updates as Partial<ViewerState>);
      console.log(`[Panel ${panelId}] After set: targetYear=${get().targetYear}`);

      if (get().autoRange) {
        computeAutoRange(get);
      }
    } catch (err) {
      console.error(`[Panel ${panelId}] Failed to load data:`, err);
      const errMsg = err instanceof Error ? err.message : "Failed to load data";
      // Friendly message for missing variables
      const isNotFound = errMsg.includes("Node not found") || errMsg.includes("Missing key");
      const displayError = isNotFound
        ? `Variable "${selectedVariable}" not available for ${panel.selectedModel}/${panel.selectedExperiment}`
        : errMsg;
      set({
        panels: get().panels.map((p) =>
          p.id === panelId
            ? {
                ...p,
                currentData: null,
                dataShape: null,
                error: displayError,
                isLoading: false,
                allNaN: false,
              }
            : p
        ),
      });
    }
  },

  loadAllPanels: async () => {
    const { panels, activePanelId, loadPanelData } = get();
    if (panels.length === 0) return;

    // Mark all loadable panels as loading upfront so they all show
    // the indicator immediately, rather than waiting for the sequential
    // active-panel load to finish before the rest even start.
    set({
      panels: get().panels.map((p) =>
        p.selectedModel && p.selectedExperiment
          ? { ...p, isLoading: true, error: null }
          : p
      ),
    });

    // Load the active panel first to establish targetYear from its time
    // labels (the slider is based on the active panel), then load the
    // rest in parallel so year-based alignment resolves correctly.
    const activeId = activePanelId || panels[0].id;
    await loadPanelData(activeId);
    const rest = panels.filter((p) => p.id !== activeId);
    if (rest.length > 0) {
      await Promise.all(rest.map((p) => loadPanelData(p.id)));
    }
  },
}));

// Helper to compute auto range across all loaded panels
function computeAutoRange(get: () => ViewerState) {
  const { panels, fillValue } = get();
  const allValidValues: number[] = [];

  for (const panel of panels) {
    if (!panel.currentData) continue;
    for (let i = 0; i < panel.currentData.length; i++) {
      const v = panel.currentData[i];
      if (isNaN(v) || !isFinite(v)) continue;
      if (Math.abs(v) > 1e10) continue;
      if (fillValue !== null && Math.abs(v - fillValue) < Math.abs(fillValue) * 1e-6) continue;
      allValidValues.push(v);
    }
  }

  if (allValidValues.length === 0) {
    console.log("[autoRange] No valid values found, setting range to 0-1");
    useViewerStore.setState({ vmin: 0, vmax: 1 });
    return;
  }

  allValidValues.sort((a, b) => a - b);
  let p5 = allValidValues[Math.floor(allValidValues.length * 0.05)];
  let p95 = allValidValues[Math.floor(allValidValues.length * 0.95)];

  // Guard against degenerate range (p5 === p95)
  if (p5 === p95) {
    const v = p5;
    if (v === 0) {
      p5 = -1;
      p95 = 1;
    } else {
      // Expand by 10% around the single value
      const margin = Math.abs(v) * 0.1;
      p5 = v - margin;
      p95 = v + margin;
    }
  }

  console.log(`[autoRange] Combined p5=${p5}, p95=${p95} from ${allValidValues.length} values`);
  useViewerStore.setState({ vmin: p5, vmax: p95 });
}
