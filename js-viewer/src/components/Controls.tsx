import { useViewerStore } from "../stores/viewerStore";
import { COLORMAP_NAMES } from "../utils/colormap";

export function Controls() {
  const {
    models,
    experiments,
    variables,
    selectedModel,
    selectedExperiment,
    selectedVariable,
    timeIndex,
    maxTimeIndex,
    colormap,
    vmin,
    vmax,
    autoRange,
    isLoading,
    setSelectedModel,
    setSelectedExperiment,
    setSelectedVariable,
    setTimeIndex,
    setColormap,
    setColorRange,
    setAutoRange,
    loadData,
  } = useViewerStore();

  const availableExperiments = selectedModel
    ? experiments.get(selectedModel) || []
    : [];

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
        ISMIP6 Viewer
      </h2>

      {/* Model Selection */}
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            marginBottom: "4px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Model
        </label>
        <select
          value={selectedModel || ""}
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "14px",
          }}
        >
          <option value="">Select model...</option>
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      {/* Experiment Selection */}
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            marginBottom: "4px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Experiment
        </label>
        <select
          value={selectedExperiment || ""}
          onChange={(e) => setSelectedExperiment(e.target.value)}
          disabled={!selectedModel}
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "14px",
          }}
        >
          <option value="">Select experiment...</option>
          {availableExperiments.map((exp) => (
            <option key={exp} value={exp}>
              {exp}
            </option>
          ))}
        </select>
      </div>

      {/* Variable Selection */}
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            marginBottom: "4px",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
          Variable
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
      </div>

      {/* Load Data Button */}
      <button
        onClick={loadData}
        disabled={isLoading || !selectedModel || !selectedExperiment}
        style={{
          width: "100%",
          padding: "10px",
          marginBottom: "24px",
          backgroundColor:
            isLoading || !selectedModel || !selectedExperiment
              ? "#ccc"
              : "#1976d2",
          color: "white",
          border: "none",
          borderRadius: "4px",
          fontSize: "14px",
          fontWeight: 500,
          cursor:
            isLoading || !selectedModel || !selectedExperiment
              ? "not-allowed"
              : "pointer",
        }}
      >
        {isLoading ? "Loading..." : "Load Data"}
      </button>

      {/* Time Slider */}
      {maxTimeIndex > 0 && (
        <div style={{ marginBottom: "16px" }}>
          <label
            style={{
              display: "block",
              marginBottom: "4px",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            Time: {timeIndex} / {maxTimeIndex}
          </label>
          <input
            type="range"
            min={0}
            max={maxTimeIndex}
            value={timeIndex}
            onChange={(e) => setTimeIndex(parseInt(e.target.value, 10))}
            style={{ width: "100%" }}
          />
        </div>
      )}

      <hr style={{ margin: "16px 0", border: "none", borderTop: "1px solid #e0e0e0" }} />

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
                onChange={(e) =>
                  setColorRange(parseFloat(e.target.value), vmax)
                }
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
                onChange={(e) =>
                  setColorRange(vmin, parseFloat(e.target.value))
                }
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
        Current range: {vmin.toFixed(1)} - {vmax.toFixed(1)}
      </div>
    </div>
  );
}
