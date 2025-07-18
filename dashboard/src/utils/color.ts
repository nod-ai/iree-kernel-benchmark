import Color from "color";
import iwanthue from "iwanthue";

const palette = iwanthue(100, {
  clustering: "k-means",
  seed: "without",
  quality: 50,
});
let paletteIndex = 0;

const backendColors: Record<string, string> = {};

export function getColor(backend: string) {
  if (!backendColors[backend]) backendColors[backend] = palette[paletteIndex++];
  return backendColors[backend];
}

export function lighten(color: string, factor: number = 0.8): string {
  return Color(color).lighten(factor).rgb().string();
}
