import { useEffect } from "react";
import { Viewer } from "./components/Viewer";
import { Controls } from "./components/Controls";
import { useViewerStore } from "./stores/viewerStore";

export default function App() {
  const { initialize, isLoading, error } = useViewerStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  return (
    <div style={{ display: "flex", width: "100%", height: "100%" }}>
      <Controls />
      <div style={{ flex: 1, position: "relative" }}>
        {isLoading && (
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
            Loading...
          </div>
        )}
        {error && (
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
            {error}
          </div>
        )}
        <Viewer />
      </div>
    </div>
  );
}
