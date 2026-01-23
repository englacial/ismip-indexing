import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

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
    proxy: {
      "/gcs-proxy": {
        target: "https://storage.googleapis.com",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/gcs-proxy/, ""),
      },
    },
  },
});
