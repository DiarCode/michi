import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    host: true,
    // Proxy /api and /ws to the FastAPI backend so the dev server
    // works out of the box without a .env override. Production
    // (Docker) uses the nginx proxy baked into the dashboard image.
    proxy: {
      "/api": {
        target: "http://localhost:8100",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8100",
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
