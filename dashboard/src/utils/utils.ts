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

export function getTimeStringRelative(time: Date) {
  const currentTime = new Date();

  const diffSeconds = (currentTime.getTime() - time.getTime()) / 1000;
  if (diffSeconds < 0) {
    return "Future";
  }

  if (diffSeconds < 60) {
    return `${Math.floor(diffSeconds)} seconds ago`;
  }

  const diffMinutes = diffSeconds / 60;
  if (diffMinutes < 60) {
    return `${Math.floor(diffMinutes)} minutes ago`;
  }

  const diffHours = diffMinutes / 60;
  if (diffHours < 24) {
    return `${Math.floor(diffHours)} hours ago`;
  }

  const diffDays = diffHours / 24;
  if (diffDays < 7) {
    return `${Math.floor(diffDays)} days ago`;
  }

  return time.toLocaleDateString();
}
