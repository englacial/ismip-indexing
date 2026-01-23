import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  base: "/viewer/",
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  optimizeDeps: {
    include: ["@deck.gl/core", "@deck.gl/layers", "@deck.gl/react"],
  },
  server: {
    fs: {
      // Allow serving files from icechunk-js parent directory
      allow: [
        // Search up from project root
        path.resolve(__dirname, ".."),
        // Explicitly allow icechunk-js
        path.resolve(__dirname, "../../icechunk-js"),
      ],
    },
    proxy: {
      "/gcs-proxy": {
        target: "https://storage.googleapis.com",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/gcs-proxy/, ""),
      },
      // Proxy for virtual chunk data from ismip6 bucket
      "/ismip6-proxy": {
        target: "https://storage.googleapis.com",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ismip6-proxy/, "/ismip6"),
      },
    },
  },
});
