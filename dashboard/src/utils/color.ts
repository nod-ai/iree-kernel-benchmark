export const BACKEND_COLORS: Record<string, string> = {
  wave: "#1f77b4",
  iree: "#ff7f0e",
  hipblaslt: "#2ca02c",
  wavegqa: "#982ca0ff",
};

function hashStringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  // Generate RGB values from the hash
  const r = (hash >> 16) & 0xff;
  const g = (hash >> 8) & 0xff;
  const b = hash & 0xff;

  // Darken the color to ensure contrast against white
  const darken = (value: number) => Math.floor(value * 0.6);

  return `rgb(${darken(r)}, ${darken(g)}, ${darken(b)})`;
}

export function getColor(backend: string) {
  return BACKEND_COLORS[backend] || hashStringToColor(backend);
}
