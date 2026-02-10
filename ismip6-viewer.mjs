/**
 * MyST directive for embedding the ISMIP6 viewer.
 *
 * Usage:
 *   ```{ismip6-viewer}
 *   :store_url: https://example.com/my-icechunk-store/
 *   :model: DOE_MALI
 *   :experiment: ctrl_proj_std
 *   :variable: lithk
 *   :controls: time
 *   :height: 600px
 *   ```
 *
 * Multi-panel:
 *   ```{ismip6-viewer}
 *   :store_url: https://example.com/my-icechunk-store/
 *   :panels: [{"model": "DOE_MALI", "experiment": "exp05"}, {"model": "JPL1_ISSM", "experiment": "exp05"}]
 *   :variable: lithk
 *   :controls: time
 *   ```
 *
 * Pinned to a specific store version:
 *   ```{ismip6-viewer}
 *   :store_url: https://example.com/my-icechunk-store/
 *   :store_ref: 0KJ35GFRGPGJ2DYWYWNG
 *   :variable: lithk
 *   ```
 *
 * Register in myst.yml:
 *   project:
 *     plugins:
 *       - ismip6-viewer.mjs
 */

const DEFAULT_VIEWER_URL = 'https://models.englacial.org/';

const ismip6ViewerDirective = {
  name: 'ismip6-viewer',
  doc: 'Embed an interactive ISMIP6 ice sheet model viewer with configurable panels, variables, and controls.',
  arg: {
    type: String,
    doc: 'Optional override for the viewer base URL',
    required: false,
  },
  options: {
    model: {
      type: String,
      doc: 'Model name (e.g., "DOE_MALI", "JPL1_ISSM")',
    },
    experiment: {
      type: String,
      doc: 'Experiment name (e.g., "ctrl_proj_std", "exp05")',
    },
    variable: {
      type: String,
      doc: 'Variable to display (e.g., "lithk", "acabf", "orog")',
    },
    time: {
      type: Number,
      doc: 'Initial time index',
    },
    colormap: {
      type: String,
      doc: 'Colormap name (viridis, plasma, inferno, magma, cividis, turbo, coolwarm, RdBu, gray)',
    },
    vmin: {
      type: Number,
      doc: 'Color scale minimum value',
    },
    vmax: {
      type: Number,
      doc: 'Color scale maximum value',
    },
    panels: {
      type: String,
      doc: 'JSON array of panel configs: [{"model": "X", "experiment": "Y"}, ...]',
    },
    controls: {
      type: String,
      doc: 'Which controls to show: "all", "time", or "none"',
    },
    store_url: {
      type: String,
      doc: 'icechunk store URL (required)',
      required: true,
    },
    store_ref: {
      type: String,
      doc: 'Store version: branch name, tag, or snapshot ID (default: "main")',
    },
    group_path: {
      type: String,
      doc: 'Override group path within store (e.g., "model/experiment")',
    },
    data_view: {
      type: String,
      doc: 'Data view: "combined" (default), "state", or "flux"',
    },
    grid_width: {
      type: Number,
      doc: 'Grid width in cells (fallback if coordinate arrays not found)',
    },
    grid_height: {
      type: Number,
      doc: 'Grid height in cells (fallback if coordinate arrays not found)',
    },
    cell_size: {
      type: Number,
      doc: 'Cell size in coordinate units (fallback if coordinate arrays not found)',
    },
    x_min: {
      type: Number,
      doc: 'Grid origin X coordinate (fallback if coordinate arrays not found)',
    },
    y_min: {
      type: Number,
      doc: 'Grid origin Y coordinate (fallback if coordinate arrays not found)',
    },
    width: {
      type: String,
      doc: 'iframe width in CSS units (default: "100%")',
    },
    height: {
      type: String,
      doc: 'iframe height in CSS units (default: "700px")',
    },
    class: {
      type: String,
      doc: 'Space-delimited CSS class names',
    },
  },
  run(data) {
    const { arg: src, options = {} } = data;
    const baseUrl = src || DEFAULT_VIEWER_URL;

    const params = new URLSearchParams();
    params.set('autoload', 'true');

    if (options.model) params.set('model', options.model);
    if (options.experiment) params.set('experiment', options.experiment);
    if (options.variable) params.set('variable', options.variable);
    if (options.time !== undefined) params.set('time', options.time.toString());
    if (options.colormap) params.set('colormap', options.colormap);
    if (options.vmin !== undefined) params.set('vmin', options.vmin.toString());
    if (options.vmax !== undefined) params.set('vmax', options.vmax.toString());
    if (options.controls) params.set('controls', options.controls);
    if (options.store_url) params.set('store_url', options.store_url);
    if (options.store_ref) params.set('store_ref', options.store_ref);
    if (options.group_path) params.set('group_path', options.group_path);
    if (options.data_view) params.set('data_view', options.data_view);
    if (options.grid_width !== undefined) params.set('grid_width', options.grid_width.toString());
    if (options.grid_height !== undefined) params.set('grid_height', options.grid_height.toString());
    if (options.cell_size !== undefined) params.set('cell_size', options.cell_size.toString());
    if (options.x_min !== undefined) params.set('x_min', options.x_min.toString());
    if (options.y_min !== undefined) params.set('y_min', options.y_min.toString());

    if (options.panels) {
      try {
        const parsed = JSON.parse(options.panels.replace(/'/g, '"'));
        params.set('panels', JSON.stringify(parsed));
      } catch (e) {
        console.warn('ismip6-viewer: failed to parse panels option:', e);
      }
    }

    const finalSrc = `${baseUrl}?${params.toString()}`;

    const iframeNode = {
      type: 'iframe',
      src: finalSrc,
      width: options.width || '100%',
      height: options.height || '700px',
      frameborder: '0',
      style: 'border: 1px solid #ccc; border-radius: 5px;',
      loading: 'lazy',
    };

    if (options.class) {
      iframeNode.class = options.class;
    }

    iframeNode.id = `ismip6-viewer-${Math.random().toString(36).substr(2, 9)}`;

    return [iframeNode];
  },
};

const plugin = {
  name: 'ISMIP6 Viewer Plugin',
  directives: [ismip6ViewerDirective],
};

export default plugin;
