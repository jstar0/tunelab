import { install } from "./install";

// Imported first from main.tsx. Does nothing unless the replay build is requested.
if (import.meta.env.VITE_REPLAY === "1") install();
