import { useViewerStore, type DataView } from "../stores/viewerStore";
import { COLORMAP_NAMES } from "../utils/colormap";
import { formatValue } from "../utils/format";
import { yearRange } from "../utils/cftime";

const DATA_VIEWS: { value: DataView; label: string }[] = [
  { value: "combined", label: "Combined" },
  { value: "state", label: "State" },
  { value: "flux", label: "Flux" },
];

export function Controls() {
  const {
    panels,
    variables,
    selectedVariable,
    timeIndex,
    colormap,
    vmin,
    vmax,
    autoRange,
    variableMetadata,
    dataView,
    isInitializing,
    setSelectedVariable,
    setTimeIndex,
    setColormap,
    setColorRange,
    setAutoRange,
    setDataView,
    addPanel,
    loadAllPanels,
  } = useViewerStore();

  const unitsLabel = variableMetadata?.units || null;
  const standardName = variableMetadata?.standardName || null;

  // Compute year range (union of all panels)
  const range = yearRange(panels.map((p) => p.timeLabels));
  const maxSlider = range ? range.maxYear - range.minYear : 0;
  const currentYear = range ? range.minYear + timeIndex : null;

  // Check if any panel is loading
  const anyLoading = panels.some((p) => p.isLoading);

  // Check if all panels have model/experiment selected
  const allPanelsConfigured = panels.every(
    (p) => p.selectedModel && p.selectedExperiment
  );

  return (
    <div
      style={{
        width: "280px",
        padding: "16px",
        borderRight: "1px solid #e0e0e0",
        overflowY: "auto",
        backgroundColor: "#fafafa",
      }}
    >
      <h2 style={{ margin: "0 0 16px 0", fontSize: "18px", fontWeight: 600 }}>
        ISMIP6 Comparison
      </h2>

      {/* Data View Toggle */}
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            marginBottom: "4px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Data View
        </label>
        <div style={{ display: "flex", borderRadius: "4px", overflow: "hidden", border: "1px solid #ccc" }}>
          {DATA_VIEWS.map((dv) => (
            <button
              key={dv.value}
              onClick={() => setDataView(dv.value)}
              disabled={isInitializing}
              style={{
                flex: 1,
                padding: "6px 0",
                fontSize: "12px",
                fontWeight: dataView === dv.value ? 600 : 400,
                backgroundColor: dataView === dv.value ? "#1976d2" : "#fff",
                color: dataView === dv.value ? "#fff" : "#333",
                border: "none",
                borderRight: dv.value !== "flux" ? "1px solid #ccc" : "none",
                cursor: isInitializing ? "not-allowed" : "pointer",
                opacity: isInitializing ? 0.6 : 1,
              }}
            >
              {dv.label}
            </button>
          ))}
        </div>
        <div style={{ fontSize: "11px", color: "#666", marginTop: "4px" }}>
          Combined: year-binned, all vars. State/Flux: original timestamps.
        </div>
      </div>

      {/* Panel Management */}
      <div style={{ marginBottom: "16px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "8px",
          }}
        >
          <label
            style={{
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            Panels: {panels.length}
          </label>
          <button
            onClick={addPanel}
            style={{
              padding: "4px 12px",
              backgroundColor: "#4caf50",
              color: "white",
              border: "none",
              borderRadius: "4px",
              fontSize: "12px",
              cursor: "pointer",
            }}
          >
            + Add Panel
          </button>
        </div>
        <div style={{ fontSize: "11px", color: "#666" }}>
          Click a panel to select it. Each panel can show a different model/experiment.
        </div>
      </div>

      {/* Variable Selection (shared across all panels) */}
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            marginBottom: "4px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Variable (all panels)
        </label>
        <select
          value={selectedVariable || ""}
          onChange={(e) => setSelectedVariable(e.target.value)}
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "14px",
          }}
        >
          {variables.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
        {(standardName || unitsLabel) && (
          <div style={{ marginTop: "4px", fontSize: "11px", color: "#666" }}>
            {standardName && <div>{standardName}</div>}
            {unitsLabel && <div>Units: {unitsLabel}</div>}
          </div>
        )}
      </div>

      {/* Load All Data Button */}
      <button
        onClick={loadAllPanels}
        disabled={anyLoading || !allPanelsConfigured}
        style={{
          width: "100%",
          padding: "10px",
          marginBottom: "24px",
          backgroundColor:
            anyLoading || !allPanelsConfigured ? "#ccc" : "#1976d2",
          color: "white",
          border: "none",
          borderRadius: "4px",
          fontSize: "14px",
          fontWeight: 500,
          cursor: anyLoading || !allPanelsConfigured ? "not-allowed" : "pointer",
        }}
      >
        {anyLoading ? "Loading..." : "Load All Panels"}
      </button>

      {/* Time Slider (year-based, union of all panels) */}
      {maxSlider > 0 && (
        <div style={{ marginBottom: "16px" }}>
          <label
            style={{
              display: "block",
              marginBottom: "4px",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            Year: {currentYear ?? timeIndex}
          </label>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#888", marginBottom: "2px" }}>
            <span>{range!.minYear}</span>
            <span>{range!.maxYear}</span>
          </div>
          <input
            type="range"
            min={0}
            max={maxSlider}
            value={timeIndex}
            onChange={(e) => setTimeIndex(parseInt(e.target.value, 10))}
            style={{ width: "100%" }}
          />
        </div>
      )}

      <hr
        style={{ margin: "16px 0", border: "none", borderTop: "1px solid #e0e0e0" }}
      />

      {/* Visualization Settings */}
      <h3 style={{ margin: "0 0 12px 0", fontSize: "14px", fontWeight: 600 }}>
        Visualization
      </h3>

      {/* Colormap Selection */}
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            marginBottom: "4px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Colormap
        </label>
        <select
          value={colormap}
          onChange={(e) => setColormap(e.target.value)}
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "14px",
          }}
        >
          {COLORMAP_NAMES.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      {/* Auto Range Toggle */}
      <div style={{ marginBottom: "12px" }}>
        <label style={{ fontSize: "13px", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={autoRange}
            onChange={(e) => setAutoRange(e.target.checked)}
            style={{ marginRight: "8px" }}
          />
          Auto color range
        </label>
      </div>

      {/* Color Range Inputs */}
      {!autoRange && (
        <div style={{ marginBottom: "16px" }}>
          <div style={{ display: "flex", gap: "8px" }}>
            <div style={{ flex: 1 }}>
              <label
                style={{
                  display: "block",
                  marginBottom: "4px",
                  fontSize: "12px",
                }}
              >
                Min
              </label>
              <input
                type="number"
                value={vmin}
                onChange={(e) => setColorRange(parseFloat(e.target.value), vmax)}
                style={{
                  width: "100%",
                  padding: "6px",
                  borderRadius: "4px",
                  border: "1px solid #ccc",
                  fontSize: "13px",
                }}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label
                style={{
                  display: "block",
                  marginBottom: "4px",
                  fontSize: "12px",
                }}
              >
                Max
              </label>
              <input
                type="number"
                value={vmax}
                onChange={(e) => setColorRange(vmin, parseFloat(e.target.value))}
                style={{
                  width: "100%",
                  padding: "6px",
                  borderRadius: "4px",
                  border: "1px solid #ccc",
                  fontSize: "13px",
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Current Range Display */}
      <div style={{ fontSize: "12px", color: "#666" }}>
        Current range: {formatValue(vmin)} - {formatValue(vmax)}{unitsLabel ? ` ${unitsLabel}` : ""}
      </div>

      <hr
        style={{ margin: "16px 0", border: "none", borderTop: "1px solid #e0e0e0" }}
      />

      {/* Colorbar */}
      <div>
        <label
          style={{
            display: "block",
            marginBottom: "8px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Color Scale
        </label>
        {unitsLabel && (
          <div style={{ fontSize: "11px", color: "#666", marginBottom: "4px", textAlign: "center" }}>
            {unitsLabel}
          </div>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{ fontSize: "11px", width: "40px", textAlign: "right" }}>
            {formatValue(vmin)}
          </div>
          <div
            style={{
              flex: 1,
              height: "20px",
              background: `linear-gradient(to right, ${getColormapGradient(colormap)})`,
              borderRadius: "2px",
            }}
          />
          <div style={{ fontSize: "11px", width: "40px" }}>{formatValue(vmax)}</div>
        </div>
      </div>
    </div>
  );
}

function getColormapGradient(colormap: string): string {
  const gradients: Record<string, string> = {
    viridis: "#440154, #3b528b, #21918c, #5ec962, #fde725",
    plasma: "#0d0887, #7e03a8, #cc4778, #f89540, #f0f921",
    inferno: "#000004, #57106e, #bc3754, #f98e09, #fcffa4",
    magma: "#000004, #51127c, #b73779, #fc8961, #fcfdbf",
    cividis: "#1f9e89, #35b779, #6ece58, #a5db36, #fde725",
    turbo: "#30123b, #23bdd8, #d9f537, #f4650b, #7a0403",
    coolwarm: "#3b4cc0, #819fce, #dddddd, #f4987a, #b40426",
    RdBu: "#67001f, #d6604d, #f7f7f7, #4393c3, #053061",
    gray: "#000000, #ffffff",
  };
  return gradients[colormap] || gradients.viridis;
}
