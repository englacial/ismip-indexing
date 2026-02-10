/**
 * URL parameter parsing for embed mode.
 *
 * When the viewer is embedded via iframe, query parameters configure
 * the initial state (model, experiment, variable, etc.) and control
 * which UI elements are visible.
 */

export interface EmbedConfig {
  /** Pre-select model for single-panel mode */
  model?: string;
  /** Pre-select experiment for single-panel mode */
  experiment?: string;
  /** Pre-select variable */
  variable?: string;
  /** Initial time index */
  time?: number;
  /** Colormap name */
  colormap?: string;
  /** Fixed color range min */
  vmin?: number;
  /** Fixed color range max */
  vmax?: number;
  /** Multi-panel config: [{model, experiment}, ...] */
  panels?: Array<{ model: string; experiment: string }>;
  /** Which controls to show: "all" | "time" | "none" */
  controls?: "all" | "time" | "none";
  /** Auto-load data on init */
  autoload?: boolean;
  /** Auto-load when dropdowns/slider change (default: true) */
  instant_load?: boolean;
  /** icechunk store URL */
  store_url?: string;
  /** Store ref: branch name, tag name, or snapshot ID (default: "main") */
  store_ref?: string;
  /** Override group path within store (e.g., "model/experiment" or just "experiment") */
  group_path?: string;
  /** Data view: "combined" (default), "state", or "flux" */
  data_view?: "combined" | "state" | "flux";
  /** Grid parameter overrides (fallback if coordinate arrays not found) */
  grid_width?: number;
  grid_height?: number;
  cell_size?: number;
  x_min?: number;
  y_min?: number;
}

export function parseUrlParams(): EmbedConfig | null {
  const params = new URLSearchParams(window.location.search);
  if (params.toString() === "") return null;

  const config: EmbedConfig = {};

  const str = (key: string) => params.get(key) || undefined;
  const num = (key: string) => {
    const v = params.get(key);
    return v ? parseFloat(v) : undefined;
  };
  const int = (key: string) => {
    const v = params.get(key);
    return v ? parseInt(v, 10) : undefined;
  };

  config.model = str("model");
  config.experiment = str("experiment");
  config.variable = str("variable");
  config.time = int("time");
  config.colormap = str("colormap");
  config.vmin = num("vmin");
  config.vmax = num("vmax");
  config.store_url = str("store_url");
  config.store_ref = str("store_ref");
  config.group_path = str("group_path");

  const dataView = str("data_view");
  if (dataView === "combined" || dataView === "state" || dataView === "flux") {
    config.data_view = dataView;
  }

  // Grid overrides
  config.grid_width = int("grid_width");
  config.grid_height = int("grid_height");
  config.cell_size = num("cell_size");
  config.x_min = num("x_min");
  config.y_min = num("y_min");

  const panels = params.get("panels");
  if (panels) {
    try {
      config.panels = JSON.parse(panels);
    } catch {
      console.warn("Failed to parse panels parameter:", panels);
    }
  }

  const controls = params.get("controls");
  if (controls === "all" || controls === "time" || controls === "none") {
    config.controls = controls;
  }

  const autoload = params.get("autoload");
  if (autoload === "true" || autoload === "1") {
    config.autoload = true;
  }

  const instantLoad = params.get("instant_load");
  if (instantLoad === "false" || instantLoad === "0") {
    config.instant_load = false;
  } else {
    config.instant_load = true; // default to true
  }

  return config;
}
