export type KernelType = "gemm" | "attention" | "conv";

export interface KernelBase {
  id: string;
  backend: string;
  kernelType: KernelType;
  name: string;
  tag: string;
  dtype: string;
  meanMicroseconds: number;
  arithmeticIntensity: number;
  tflops: number;
}

export interface GemmKernel extends KernelBase {
  kernelType: "gemm";
  M: number;
  N: number;
  K: number;
}

export interface AttentionKernel extends KernelBase {
  kernelType: "attention";
  B: number;
  M: number;
  N: number;
  K1: number;
  K2: number;
}

export interface ConvKernel extends KernelBase {
  kernelType: "conv";
  B: number;
  H: number;
  W: number;
  C: number;
  P: number;
  Q: number;
  F: number;
  S: number;
}

export type Kernel = GemmKernel | AttentionKernel | ConvKernel;
