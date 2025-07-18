import Color, { type ColorInstance } from "color";
import iwanthue from "iwanthue";

const palette = iwanthue(100, {
  clustering: "k-means",
  seed: "without",
  quality: 50,
});
let paletteIndex = 0;

const backendColors: Record<string, string> = {};

export function getBackendColor(backend: string): ColorInstance {
  if (!backendColors[backend]) backendColors[backend] = palette[paletteIndex++];
  const colorStr = backendColors[backend];
  return Color(colorStr);
}
