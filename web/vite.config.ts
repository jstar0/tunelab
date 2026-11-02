import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Local builds keep Vite's default output (web/dist), which `tunelab serve` mounts.
// The GitHub Pages replay build sets PAGES_BASE and publishes web/dist as well.
export default defineConfig({
  base: process.env.PAGES_BASE ?? "/",
  plugins: [react()],
  build: process.env.PAGES_BASE ? { outDir: "dist" } : undefined,
});
