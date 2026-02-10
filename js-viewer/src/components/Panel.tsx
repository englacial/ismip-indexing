import { useMemo, useCallback, useEffect, useRef, useState } from "react";
import { DeckGL } from "@deck.gl/react";
import { OrthographicView, PickingInfo } from "@deck.gl/core";
import { BitmapLayer } from "@deck.gl/layers";
import { useViewerStore, Panel as PanelType } from "../stores/viewerStore";
import { dataToRGBA } from "../utils/colormap";
import { formatValue } from "../utils/format";

interface PanelProps {
  panel: PanelType;
  isActive: boolean;
  canRemove: boolean;
}

export function Panel({ panel, isActive, canRemove }: PanelProps) {
  const {
    colormap,
    vmin,
    vmax,
    viewState,
    setViewState,
    setActivePanel,
    removePanel,
    setPanelModel,
    setPanelExperiment,
    loadPanelData,
    models,
    experiments,
    panels,
    hoverGridPosition,
    hoveredPanelId,
    setHoverGridPosition,
    getValueAtGridPosition,
    gridConfig,
    fillValue,
    variableMetadata,
    selectedVariable,
    targetYear,
  } = useViewerStore();

  const unitsLabel = variableMetadata?.units || null;
  const standardName = variableMetadata?.standardName || null;

  // Derive grid geometry from store config
  const { width: GRID_WIDTH, height: GRID_HEIGHT, cellSize: CELL_SIZE, xMin: X_MIN, yMin: Y_MIN, xMax: X_MAX, yMax: Y_MAX } = gridConfig;
  const CENTER_X = (X_MIN + X_MAX) / 2;
  const CENTER_Y = (Y_MIN + Y_MAX) / 2;
  const INITIAL_VIEW_STATE = useMemo(() => ({
    target: [CENTER_X, CENTER_Y, 0] as [number, number, number],
    zoom: -13,
    minZoom: -16,
    maxZoom: 0,
  }), [CENTER_X, CENTER_Y]);

  const { currentData, dataShape, selectedModel, selectedExperiment, isLoading, error, groupMetadata, allNaN } = panel;
  const [showInfo, setShowInfo] = useState(false);

  const availableExperiments = selectedModel
    ? experiments.get(selectedModel) || []
    : [];

  // Build an ImageBitmap from the data. ImageBitmap is immutable and
  // GPU-transferable, so deck.gl can upload it as a texture without
  // racing against canvas mutations or garbage collection.
  const [bitmap, setBitmap] = useState<ImageBitmap | null>(null);
  const bitmapGenRef = useRef(0);

  useEffect(() => {
    if (!currentData || !dataShape) {
      setBitmap(null);
      return;
    }

    const gen = ++bitmapGenRef.current;
    const [height, width] = dataShape;
    const rgba = dataToRGBA(currentData, width, height, vmin, vmax, colormap, fillValue);

    const imgData = new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength), width, height);

    createImageBitmap(imgData, { imageOrientation: "flipY" }).then(
      (bmp) => {
        // Only apply if this is still the latest generation
        if (bitmapGenRef.current === gen) {
          setBitmap((prev) => {
            prev?.close(); // release old GPU resource
            return bmp;
          });
        } else {
          bmp.close(); // stale, discard
        }
      },
      (err) => console.error("Failed to create ImageBitmap:", err)
    );

    return () => {
      // If the effect re-runs before the promise resolves, the gen
      // check above will discard the stale bitmap.
    };
  }, [currentData, dataShape, colormap, vmin, vmax, fillValue]);

  const onHover = useCallback(
    (info: PickingInfo) => {
      const { coordinate } = info;
      if (!coordinate) {
        setHoverGridPosition(null, null);
        return;
      }

      const [worldX, worldY] = coordinate;
      const gridX = Math.floor((worldX - X_MIN) / CELL_SIZE);
      const gridY = Math.floor((worldY - Y_MIN) / CELL_SIZE);

      if (gridX < 0 || gridX >= GRID_WIDTH || gridY < 0 || gridY >= GRID_HEIGHT) {
        setHoverGridPosition(null, null);
        return;
      }

      setHoverGridPosition({ gridX, gridY }, panel.id);
    },
    [setHoverGridPosition, panel.id]
  );

  // Get values from all panels at the current hover position
  const hoverValues = useMemo(() => {
    if (!hoverGridPosition) return null;
    const { gridX, gridY } = hoverGridPosition;

    return panels
      .filter((p) => p.currentData && p.selectedModel && p.selectedExperiment)
      .map((p) => ({
        panelId: p.id,
        model: p.selectedModel!,
        experiment: p.selectedExperiment!,
        value: getValueAtGridPosition(p.id, gridX, gridY),
        timeLabel: p.resolvedTimeIndex !== null ? (p.timeLabels?.[p.resolvedTimeIndex] ?? null) : null,
      }));
  }, [hoverGridPosition, panels, getValueAtGridPosition]);

  // Hide data when the target year is outside this panel's time range
  // (also triggers when time labels couldn't be decoded but the array has a time dimension)
  const outOfRange = panel.resolvedTimeIndex === null && panel.maxTimeIndex > 0;

  const layers = useMemo(() => {
    if (!bitmap || outOfRange) return [];

    return [
      new BitmapLayer({
        id: `data-layer-${panel.id}`,
        bounds: [X_MIN, Y_MIN, X_MAX, Y_MAX],
        image: bitmap,
        pickable: false,
      }),
    ];
  }, [bitmap, panel.id, outOfRange]);

  const views = new OrthographicView({
    id: "ortho",
    flipY: false,
  });

  // Use shared view state for linked zoom/pan
  const currentViewState = viewState || INITIAL_VIEW_STATE;

  const onViewStateChange = useCallback(
    ({ viewState: newViewState }: { viewState: typeof INITIAL_VIEW_STATE }) => {
      setViewState({
        target: newViewState.target,
        zoom: newViewState.zoom,
      });
    },
    [setViewState]
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        border: isActive ? "2px solid #1976d2" : "1px solid #e0e0e0",
        borderRadius: "4px",
        overflow: "hidden",
      }}
      onClick={() => setActivePanel(panel.id)}
    >
      {/* Panel header with model/experiment selectors */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 100,
          background: "rgba(255,255,255,0.95)",
          padding: "8px 12px",
          borderBottom: "1px solid #e0e0e0",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <select
          value={selectedModel || ""}
          onChange={(e) => setPanelModel(panel.id, e.target.value)}
          style={{
            padding: "4px 8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "12px",
            flex: 1,
            minWidth: 0,
          }}
        >
          <option value="">Model...</option>
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>

        <select
          value={selectedExperiment || ""}
          onChange={(e) => setPanelExperiment(panel.id, e.target.value)}
          disabled={!selectedModel}
          style={{
            padding: "4px 8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "12px",
            flex: 1,
            minWidth: 0,
          }}
        >
          <option value="">Experiment...</option>
          {availableExperiments.map((exp) => (
            <option key={exp} value={exp}>
              {exp}
            </option>
          ))}
        </select>

        <button
          onClick={(e) => {
            e.stopPropagation();
            loadPanelData(panel.id);
          }}
          disabled={isLoading || !selectedModel || !selectedExperiment}
          style={{
            padding: "4px 12px",
            backgroundColor:
              isLoading || !selectedModel || !selectedExperiment
                ? "#ccc"
                : "#1976d2",
            color: "white",
            border: "none",
            borderRadius: "4px",
            fontSize: "12px",
            cursor:
              isLoading || !selectedModel || !selectedExperiment
                ? "not-allowed"
                : "pointer",
          }}
        >
          {isLoading ? "..." : "Load"}
        </button>

        {canRemove && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              removePanel(panel.id);
            }}
            style={{
              padding: "4px 8px",
              backgroundColor: "#f44336",
              color: "white",
              border: "none",
              borderRadius: "4px",
              fontSize: "12px",
              cursor: "pointer",
            }}
          >
            X
          </button>
        )}
      </div>

      {/* Map view */}
      <div
        style={{ position: "absolute", top: "48px", left: 0, right: 0, bottom: 0 }}
        onMouseLeave={() => setHoverGridPosition(null, null)}
      >
        <DeckGL
          views={views}
          viewState={currentViewState}
          onViewStateChange={onViewStateChange}
          controller={true}
          layers={layers}
          style={{ background: "#1a1a2e" }}
          onHover={onHover}
        />

        {/* Loading overlay */}
        {isLoading && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              background: "rgba(255,255,255,0.9)",
              padding: "12px 20px",
              borderRadius: "4px",
              fontSize: "14px",
              zIndex: 160,
            }}
          >
            Loading...
          </div>
        )}

        {/* Error overlay */}
        {error && (
          <div
            style={{
              position: "absolute",
              bottom: "50px",
              left: "10px",
              right: "10px",
              background: "#ffebee",
              color: "#c62828",
              padding: "8px",
              borderRadius: "4px",
              fontSize: "12px",
            }}
          >
            {error}
          </div>
        )}

        {/* Out of range overlay */}
        {outOfRange && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              color: "white",
            }}
          >
            <div style={{ fontSize: "14px", opacity: 0.7 }}>
              No data for selected year
            </div>
          </div>
        )}

        {/* All-NaN data overlay — persists through re-loads; cleared on selection change */}
        {allNaN && currentData && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              background: "rgba(0, 0, 0, 0.75)",
              color: "white",
              padding: "16px 24px",
              borderRadius: "6px",
              maxWidth: "80%",
              zIndex: 150,
            }}
          >
            <div style={{ fontSize: "14px", fontWeight: 500, marginBottom: "6px" }}>
              No data available
            </div>
            <div style={{ fontSize: "11px", opacity: 0.7 }}>
              {selectedVariable ? `"${selectedVariable}"` : "Variable"} contains only
              NaN values for {selectedModel}/{selectedExperiment}
              {targetYear != null && !isNaN(targetYear) ? ` at year ${targetYear}` : ""}
            </div>
          </div>
        )}

        {/* No data message */}
        {!currentData && !isLoading && !outOfRange && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center",
              color: "white",
            }}
          >
            <div style={{ fontSize: "14px", opacity: 0.7 }}>
              Select model/experiment and click Load
            </div>
          </div>
        )}

        {/* Hover tooltip - show values from all panels (only on hovered panel) */}
        {hoveredPanelId === panel.id && hoverGridPosition && hoverValues && hoverValues.length > 0 && (
          <div
            style={{
              position: "absolute",
              left: "50%",
              bottom: "50px",
              transform: "translateX(-50%)",
              background: "rgba(0, 0, 0, 0.9)",
              color: "white",
              padding: "8px 12px",
              borderRadius: "4px",
              fontSize: "11px",
              pointerEvents: "none",
              whiteSpace: "nowrap",
              zIndex: 200,
            }}
          >
            <div style={{ marginBottom: "4px", opacity: 0.7, fontSize: "10px" }}>
              Grid: ({hoverGridPosition.gridX}, {hoverGridPosition.gridY})
            </div>
            {hoverValues.map((hv) => (
              <div
                key={hv.panelId}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "12px",
                  padding: "2px 0",
                  borderBottom: hv.panelId === panel.id ? "1px solid #666" : "none",
                  fontWeight: hv.panelId === panel.id ? "bold" : "normal",
                }}
              >
                <span style={{ opacity: 0.8 }}>
                  {hv.model}/{hv.experiment}{hv.timeLabel ? ` [${hv.timeLabel}]` : ""}:
                </span>
                <span>
                  {hv.value !== null ? `${formatValue(hv.value)}${unitsLabel ? ` ${unitsLabel}` : ""}` : "N/A"}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Panel label with info button */}
        {currentData && (
          <div
            style={{
              position: "absolute",
              bottom: "8px",
              left: "8px",
              background: "rgba(255,255,255,0.9)",
              padding: "4px 8px",
              borderRadius: "4px",
              fontSize: "11px",
              maxWidth: "calc(100% - 16px)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ fontWeight: 500 }}>{selectedModel} / {selectedExperiment}</span>
              {groupMetadata && (
                <button
                  onClick={(e) => { e.stopPropagation(); setShowInfo(!showInfo); }}
                  style={{
                    background: "none", border: "1px solid #999", borderRadius: "50%",
                    width: "16px", height: "16px", fontSize: "10px", lineHeight: "14px",
                    cursor: "pointer", padding: 0, color: "#666",
                  }}
                  title="Show metadata"
                >
                  i
                </button>
              )}
            </div>
            {standardName && (
              <div style={{ fontSize: "10px", color: "#666", marginTop: "2px" }}>
                {standardName}{unitsLabel ? ` (${unitsLabel})` : ""}
              </div>
            )}
          </div>
        )}

        {/* Group metadata info panel */}
        {showInfo && groupMetadata && (
          <div
            style={{
              position: "absolute",
              bottom: "48px",
              left: "8px",
              right: "8px",
              background: "rgba(255,255,255,0.95)",
              padding: "10px 12px",
              borderRadius: "4px",
              fontSize: "11px",
              zIndex: 300,
              maxHeight: "200px",
              overflowY: "auto",
              boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
              <span style={{ fontWeight: 600, fontSize: "12px" }}>Metadata</span>
              <button
                onClick={() => setShowInfo(false)}
                style={{ background: "none", border: "none", cursor: "pointer", fontSize: "14px", color: "#666" }}
              >
                x
              </button>
            </div>
            {Object.entries(groupMetadata)
              .filter(([, v]) => v !== null)
              .map(([key, value]) => (
                <div key={key} style={{ marginBottom: "4px" }}>
                  <span style={{ fontWeight: 500, color: "#444" }}>{key}: </span>
                  <span style={{ color: "#666", wordBreak: "break-word" }}>{value}</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
