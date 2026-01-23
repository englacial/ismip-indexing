import { useMemo } from "react";
import { DeckGL } from "@deck.gl/react";
import { OrthographicView } from "@deck.gl/core";
import { BitmapLayer } from "@deck.gl/layers";
import { useViewerStore } from "../stores/viewerStore";
import { dataToRGBA } from "../utils/colormap";

// ISMIP6 Antarctic data grid parameters (EPSG:3031)
// Standard grid: 761x761 at 8km resolution
const GRID_WIDTH = 761;
const GRID_HEIGHT = 761;
const CELL_SIZE = 8000; // 8km in meters

// Grid extent in EPSG:3031 coordinates
const X_MIN = -3040000;
const Y_MIN = -3040000;
const X_MAX = X_MIN + GRID_WIDTH * CELL_SIZE;
const Y_MAX = Y_MIN + GRID_HEIGHT * CELL_SIZE;

// Center of the grid
const CENTER_X = (X_MIN + X_MAX) / 2;
const CENTER_Y = (Y_MIN + Y_MAX) / 2;

const INITIAL_VIEW_STATE = {
  target: [CENTER_X, CENTER_Y, 0] as [number, number, number],
  zoom: -13, // ~6 million meters / ~1000 pixels = need zoom around -12 to -13
  minZoom: -16,
  maxZoom: 0,
};

export function Viewer() {
  const { currentData, dataShape, colormap, vmin, vmax } = useViewerStore();

  const imageData = useMemo(() => {
    if (!currentData || !dataShape) return null;

    const [height, width] = dataShape;
    const rgba = dataToRGBA(currentData, width, height, vmin, vmax, colormap);

    // Create ImageData for BitmapLayer
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    // Put pixel data directly onto canvas
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(rgba);
    ctx.putImageData(imgData, 0, 0);

    return canvas.toDataURL();
  }, [currentData, dataShape, colormap, vmin, vmax]);

  const layers = useMemo(() => {
    if (!imageData) return [];

    return [
      new BitmapLayer({
        id: "data-layer",
        bounds: [X_MIN, Y_MIN, X_MAX, Y_MAX],
        image: imageData,
        pickable: false,
      }),
    ];
  }, [imageData]);

  const views = new OrthographicView({
    id: "ortho",
    flipY: false,
  });

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <DeckGL
        views={views}
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
        style={{ background: "#1a1a2e" }}
      />

      {/* Scale bar and info overlay */}
      <div
        style={{
          position: "absolute",
          bottom: "16px",
          left: "16px",
          background: "rgba(255,255,255,0.9)",
          padding: "8px 12px",
          borderRadius: "4px",
          fontSize: "12px",
        }}
      >
        <div>Grid: {GRID_WIDTH} x {GRID_HEIGHT} @ 8km</div>
        <div>Projection: EPSG:3031</div>
        {currentData && <div>Color range: {vmin.toFixed(1)} - {vmax.toFixed(1)}</div>}
        {dataShape && <div>Data shape: {dataShape.join(" x ")}</div>}
      </div>

      {/* Colorbar */}
      {currentData && (
        <div
          style={{
            position: "absolute",
            right: "16px",
            top: "50%",
            transform: "translateY(-50%)",
            background: "rgba(255,255,255,0.9)",
            padding: "8px",
            borderRadius: "4px",
          }}
        >
          <div
            style={{
              fontSize: "11px",
              textAlign: "center",
              marginBottom: "4px",
            }}
          >
            {vmax.toFixed(0)}
          </div>
          <div
            style={{
              width: "20px",
              height: "200px",
              background: `linear-gradient(to bottom,
                ${getColormapGradient(colormap)})`,
              borderRadius: "2px",
            }}
          />
          <div
            style={{
              fontSize: "11px",
              textAlign: "center",
              marginTop: "4px",
            }}
          >
            {vmin.toFixed(0)}
          </div>
        </div>
      )}

      {/* No data message */}
      {!currentData && (
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
          <div style={{ fontSize: "18px", marginBottom: "8px" }}>
            No data loaded
          </div>
          <div style={{ fontSize: "14px", opacity: 0.7 }}>
            Select a model, experiment, and click "Load Data"
          </div>
        </div>
      )}
    </div>
  );
}

function getColormapGradient(colormap: string): string {
  // Simple gradient stops for colorbar visualization
  const gradients: Record<string, string> = {
    viridis: "#fde725, #5ec962, #21918c, #3b528b, #440154",
    plasma: "#f0f921, #f89540, #cc4778, #7e03a8, #0d0887",
    inferno: "#fcffa4, #f98e09, #bc3754, #57106e, #000004",
    magma: "#fcfdbf, #fc8961, #b73779, #51127c, #000004",
    cividis: "#fde725, #a5db36, #6ece58, #35b779, #1f9e89",
    turbo: "#7a0403, #f4650b, #d9f537, #23bdd8, #30123b",
    coolwarm: "#b40426, #f4987a, #dddddd, #819fce, #3b4cc0",
    RdBu: "#053061, #4393c3, #f7f7f7, #d6604d, #67001f",
    gray: "#ffffff, #000000",
  };
  return gradients[colormap] || gradients.viridis;
}
