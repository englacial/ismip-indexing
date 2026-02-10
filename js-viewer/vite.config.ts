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
      // Proxy for icechunk store on source.coop
      "/s3-proxy": {
        target: "https://data.source.coop",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/s3-proxy/, "/englacial/ismip6/icechunk-ais"),
      },
      // Proxy for virtual chunk data from source.coop S3 bucket
      "/ismip6-proxy": {
        target: "https://data.source.coop",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ismip6-proxy/, "/englacial/ismip6"),
      },
    },
  },
});
