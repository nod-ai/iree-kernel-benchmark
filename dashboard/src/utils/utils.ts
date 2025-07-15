import type { KernelType } from "../types";

export function toTitleCase(str: string): string {
  return str.replace(
    /\w\S*/g,
    (text) => text.charAt(0).toUpperCase() + text.substring(1).toLowerCase()
  );
}

export const KERNEL_DIMS: Record<KernelType, string[]> = {
  gemm: ["M", "N", "K", "transpose", "dtype"],
  attention: ["B", "M", "N", "K1", "K2", "dtype"],
  conv: ["B", "H", "W", "C", "P", "Q", "F", "S", "dtype"],
};
