# Embedding the ISMIP6 Viewer

The ISMIP6 viewer can be embedded into scientific documents authored with
[MyST Markdown](https://mystmd.org) or [Quarto](https://quarto.org). Both
platforms use a thin directive/shortcode that renders an `<iframe>` pointing at
the hosted viewer with URL parameters that configure its initial state.

The viewer is a client-side React application backed by an
[icechunk](https://icechunk.io) store containing Zarr v3 arrays. While it was
designed for ISMIP6 ice sheet model output, it works with any icechunk store
that follows a compatible hierarchy. The only **required** option is the store
URL — everything else is either auto-discovered or optional.

## How the viewer initializes

When the page loads, the viewer performs the following steps in order:

1. **Connects to the icechunk store** at the given `store_url` and reads the
   snapshot identified by `store_ref` (default: the `main` branch head). Using
   a specific snapshot ID pins the viewer to an exact version of the data,
   which is useful for reproducible figures in publications.

2. **Discovers the data hierarchy** automatically — walking the group tree to
   find models, experiments, and variables. The viewer supports three levels of
   nesting: flat stores (arrays at root), single-level grouping
   (experiment/arrays), and two-level grouping (model/experiment/arrays, the
   ISMIP6 pattern).

3. **Reads coordinate arrays** (`x`, `y`) from the store to derive the spatial
   grid (extent, cell size, dimensions). If the coordinates can't be read
   (e.g., inline chunks not yet supported by the store backend), it falls back
   to URL parameter overrides, then to the built-in ISMIP6 Antarctic defaults
   (761 &times; 761 cells, 8 km resolution, EPSG:3031).

4. **Reads the fill value** from Zarr array metadata so masked pixels render as
   transparent. The fill value is re-read each time the variable changes, since
   different variables may use different fill values.

5. **Computes a color range** automatically from the 5th and 95th percentiles
   of the loaded data, handling negative values, near-zero values, and
   degenerate ranges (all-identical or all-fill data).

Any of these automatically discovered values can be overridden via directive
options.

---

## MyST Markdown

### Setup

Register the plugin in your `myst.yml`:

```yaml
project:
  plugins:
    - ismip6-viewer.mjs
```

### Usage

The `store_url` option is **required** — it tells the viewer which icechunk
store to read from:

````markdown
```{ismip6-viewer}
:store_url: https://data.source.coop/englacial/ismip6/icechunk-ais/
:model: DOE_MALI
:experiment: ctrl_proj_std
:variable: lithk
```
````

Multi-panel comparison (linked zoom/pan):

````markdown
```{ismip6-viewer}
:store_url: https://data.source.coop/englacial/ismip6/icechunk-ais/
:panels: [{"model": "DOE_MALI", "experiment": "exp05"}, {"model": "JPL1_ISSM", "experiment": "exp05"}]
:variable: lithk
:controls: time
```
````

Pinned to a specific store version for reproducibility:

````markdown
```{ismip6-viewer}
:store_url: https://data.source.coop/englacial/ismip6/icechunk-ais/
:store_ref: 0KJ35GFRGPGJ2DYWYWNG
:model: DOE_MALI
:variable: lithk
```
````

The directive also accepts an optional positional argument to override the
viewer base URL (the web application itself, not the data store):

````markdown
```{ismip6-viewer} https://my-custom-viewer.example.com/
:store_url: https://example.com/my-store/
:variable: orog
```
````

---

## Quarto

### Setup

Install the extension into your Quarto project:

```bash
quarto add _extensions/ismip6-viewer
```

Or copy the `_extensions/ismip6-viewer/` directory into your project.

### Usage

```markdown
{{< ismip6-viewer store_url="https://data.source.coop/englacial/ismip6/icechunk-ais/" model="DOE_MALI" experiment="ctrl_proj_std" variable="lithk" >}}
```

Multi-panel:

```markdown
{{< ismip6-viewer store_url="https://data.source.coop/englacial/ismip6/icechunk-ais/" panels='[{"model":"DOE_MALI","experiment":"exp05"},{"model":"JPL1_ISSM","experiment":"exp05"}]' variable="lithk" controls="time" >}}
```

Pinned to a specific snapshot:

```markdown
{{< ismip6-viewer store_url="https://data.source.coop/englacial/ismip6/icechunk-ais/" store_ref="0KJ35GFRGPGJ2DYWYWNG" variable="lithk" >}}
```

Non-HTML outputs (PDF, LaTeX) render a placeholder message instead of the
interactive viewer.

---

## Options Reference

### Store configuration (required)

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `store_url` | string | **yes** | icechunk store URL. This is the only required option. |
| `store_ref` | string | no | Store version to read. Can be a **branch name** (e.g., `main`), a **tag**, or a **snapshot ID** (20-character Crockford Base32 string). Defaults to `main`. Use a snapshot ID for reproducible, immutable references. |
| `group_path` | string | no | Base group path within the store. Use when data sits under a sub-group rather than at the store root. |
| `data_view` | string | no | Data view to display: `combined` (default, year-binned, all variables), `state` (original timestamps, state variables only), or `flux` (original timestamps, flux variables only). Maps to the top-level group in the unified store. |

### Data selection

All data selection options are **optional**. When omitted, the viewer presents
dropdown menus for the user to choose interactively.

| Option | Type | Description |
|--------|------|-------------|
| `model` | string | Pre-select a model (e.g., `DOE_MALI`, `JPL1_ISSM`). |
| `experiment` | string | Pre-select an experiment (e.g., `ctrl_proj_std`, `exp05`). |
| `variable` | string | Pre-select a variable (e.g., `lithk`, `acabf`, `orog`). Only 2D and 3D spatial arrays are listed; 1D time-series variables are filtered out automatically. |
| `time` | integer | Initial time step index (0-based). Defaults to `0`. |
| `panels` | JSON string | Configure multiple panels for side-by-side comparison. Each entry needs `model` and `experiment` fields. Panels share zoom/pan state and hover tooltips. Overrides the single `model`/`experiment` options. |

### Display

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `colormap` | string | `viridis` | Colormap name. Available: `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `turbo`, `coolwarm`, `RdBu`, `gray`. |
| `vmin` | number | *auto* | Fix the color scale minimum. When set, disables auto-range. |
| `vmax` | number | *auto* | Fix the color scale maximum. When set, disables auto-range. |
| `controls` | string | `all` | Which UI controls to show: `all` (full sidebar), `time` (time slider only), or `none` (static view). |
| `width` | string | `100%` | iframe width in CSS units. |
| `height` | string | `700px` | iframe height in CSS units (MyST) or pixels (Quarto). |
| `class` | string | | CSS class names for the iframe (MyST only). |

### Grid overrides

The viewer reads `x` and `y` coordinate arrays from the store to derive the
spatial grid automatically. If those arrays can't be read (for example, because
they use inline chunks that the store backend doesn't yet support), the viewer
falls back to these parameters, and finally to the built-in ISMIP6 Antarctic
defaults.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `grid_width` | integer | *auto* / 761 | Number of grid cells in the x-direction. |
| `grid_height` | integer | *auto* / 761 | Number of grid cells in the y-direction. |
| `cell_size` | number | *auto* / 8000 | Cell size in the coordinate system's units (meters for EPSG:3031). |
| `x_min` | number | *auto* / -3040000 | X-coordinate of the grid origin (lower-left corner). |
| `y_min` | number | *auto* / -3040000 | Y-coordinate of the grid origin (lower-left corner). |

---

## Auto-discovery behavior

The viewer automatically discovers as much as possible from the store metadata,
falling back through a chain of defaults:

| Property | Discovery method | Fallback |
|----------|-----------------|----------|
| **Models & experiments** | Walk the store group hierarchy (supports 0, 1, or 2 levels of nesting) | Empty list; user selects manually |
| **Variables** | List arrays under a sample model/experiment group, filtering out coordinate variables (`x`, `y`, `lat`, `lon`, `time`, etc.) and 1D arrays | Empty list |
| **Grid geometry** | Read `x` and `y` coordinate arrays from the store | URL parameter overrides &rarr; ISMIP6 defaults (761 &times; 761, 8 km, origin at &minus;3,040,000) |
| **Fill value** | Read `fill_value` from Zarr array metadata, or `_FillValue` attribute | Heuristic: values with |v| > 10^10 treated as fill |
| **Color range** | 5th and 95th percentile of valid (non-fill, finite) data values | 0&ndash;1 if no valid values exist; expanded by &plusmn;10% if all values are identical |

Setting `vmin`/`vmax` explicitly disables auto-range. All other overrides
are applied on top of discovery results — for example, setting `model` still
allows the viewer to discover experiments and variables from the store.

---

## Data views

The unified ISMIP6 icechunk store contains three top-level groups, each
providing a different view of the same underlying data:

| View | Group | Time handling | Variables |
|------|-------|---------------|-----------|
| **Combined** | `combined/` | Year-binned (one time step per year) | All variables (state + flux) |
| **State** | `state/` | Original model timestamps | State variables only (lithk, orog, xvelsurf, ...) |
| **Flux** | `flux/` | Original model timestamps | Flux variables only (acabf, libmassbfgr, ...) |

The viewer includes an interactive toggle in the sidebar to switch between
views. You can also set the initial view via the `data_view` option:

**MyST:**

````markdown
```{ismip6-viewer}
:store_url: https://data.source.coop/englacial/ismip6/icechunk-ais/
:data_view: state
:model: DOE_MALI
:experiment: ctrl_proj_std
:variable: lithk
```
````

**Quarto:**

```markdown
{{< ismip6-viewer store_url="https://data.source.coop/englacial/ismip6/icechunk-ais/" data_view="state" model="DOE_MALI" experiment="ctrl_proj_std" variable="lithk" >}}
```

## Reproducibility with `store_ref`

icechunk stores are versioned. Every write operation creates a new snapshot,
and branches (like `main`) point to the latest snapshot on that branch. By
default, the viewer reads from the head of `main`, which means the displayed
data always reflects the most recent version of the store.

For published figures or archived notebooks, use `store_ref` with a **snapshot
ID** to pin the viewer to an exact, immutable version of the data:

````markdown
```{ismip6-viewer}
:store_url: https://data.source.coop/englacial/ismip6/icechunk-ais/
:store_ref: 0KJ35GFRGPGJ2DYWYWNG
:variable: lithk
:model: DOE_MALI
:experiment: ctrl_proj_std
```
````

Snapshot IDs are 20-character Crockford Base32 strings (uppercase letters and
digits, e.g., `0KJ35GFRGPGJ2DYWYWNG`). You can also use a **tag name** if the
store maintainer has created tags for specific releases. The viewer
auto-detects whether the ref is a snapshot ID or a branch/tag name.
