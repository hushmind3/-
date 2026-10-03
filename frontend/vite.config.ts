import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8766" },
  },
  build: { outDir: "../src/stockrl/web/dist", emptyOutDir: true },
  test: { environment: "jsdom", globals: true, restoreMocks: true },
});
