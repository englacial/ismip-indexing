import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  base: "/static/models/",
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
      // Proxy for icechunk store on S3
      "/s3-proxy": {
        target: "https://ismip6-icechunk.s3.us-west-2.amazonaws.com",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/s3-proxy/, ""),
      },
      // Proxy for virtual chunk data from ismip6 GCS bucket
      "/ismip6-proxy": {
        target: "https://storage.googleapis.com",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ismip6-proxy/, "/ismip6"),
      },
    },
  },
});
