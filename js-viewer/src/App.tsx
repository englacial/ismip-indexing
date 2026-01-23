import { useEffect } from "react";
import { Panel } from "./components/Panel";
import { Controls } from "./components/Controls";
import { useViewerStore } from "./stores/viewerStore";

export default function App() {
  const { initialize, isInitializing, initError, panels, activePanelId } =
    useViewerStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  // Calculate grid layout based on number of panels
  const getGridStyle = (count: number): React.CSSProperties => {
    if (count === 1) {
      return { gridTemplateColumns: "1fr", gridTemplateRows: "1fr" };
    } else if (count === 2) {
      return { gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr" };
    } else if (count <= 4) {
      return { gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr" };
    } else if (count <= 6) {
      return { gridTemplateColumns: "1fr 1fr 1fr", gridTemplateRows: "1fr 1fr" };
    } else {
      // For more than 6, use 3 columns with as many rows as needed
      const rows = Math.ceil(count / 3);
      return {
        gridTemplateColumns: "1fr 1fr 1fr",
        gridTemplateRows: Array(rows).fill("1fr").join(" "),
      };
    }
  };

  return (
    <div style={{ display: "flex", width: "100%", height: "100%" }}>
      <Controls />
      <div style={{ flex: 1, position: "relative" }}>
        {/* Initialization loading */}
        {isInitializing && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              zIndex: 1000,
              background: "rgba(255,255,255,0.9)",
              padding: "20px",
              borderRadius: "8px",
              boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
            }}
          >
            Connecting to data store...
          </div>
        )}

        {/* Initialization error */}
        {initError && (
          <div
            style={{
              position: "absolute",
              top: "10px",
              right: "10px",
              zIndex: 1000,
              background: "#ffebee",
              color: "#c62828",
              padding: "10px 20px",
              borderRadius: "4px",
              maxWidth: "400px",
            }}
          >
            {initError}
          </div>
        )}

        {/* Panel grid */}
        <div
          style={{
            display: "grid",
            ...getGridStyle(panels.length),
            gap: "8px",
            padding: "8px",
            width: "100%",
            height: "100%",
            boxSizing: "border-box",
          }}
        >
          {panels.map((panel) => (
            <Panel
              key={panel.id}
              panel={panel}
              isActive={panel.id === activePanelId}
              canRemove={panels.length > 1}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
