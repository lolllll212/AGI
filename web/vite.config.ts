/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the console runs on its own port and forwards API calls to the Python server,
// so the browser only ever talks to one origin. Point AGI_API at the server if it is not local.
const api = process.env.AGI_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,
    proxy: {
      "/v1": { target: api, changeOrigin: true },
      "/health": { target: api, changeOrigin: true },
      "/metrics": { target: api, changeOrigin: true },
    },
  },
  build: {
    // The Python server serves these files itself, so a deployment needs no Node.
    outDir: "../src/agi/infrastructure/api/static",
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    globals: false,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
