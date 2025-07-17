/* Kernels */

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

export type GemmTransposeType = "NN" | "NT" | "TN" | "TT";
export interface GemmKernel extends KernelBase {
  kernelType: "gemm";
  M: number;
  N: number;
  K: number;
  transpose: GemmTransposeType;
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

/* Source Control */
export interface ChangeAuthor {
  name: string;
  profileUrl: string;
}

export type ChangeStats = Record<KernelType, number>;

export interface RepoModification {
  _id: string;
  type: "pr" | "merge";
  timestamp: Date;
  author: ChangeAuthor;
  changeStats: ChangeStats;
}

export interface RepoCommit {
  _id: string;
  title: string;
  author: ChangeAuthor;
  timestamp: Date;
  description?: string;
}

export interface RepoPullRequest extends RepoModification {
  type: "pr";
  title: string;
  description?: string;
  status: "open" | "closed";
  commits: RepoCommit[];
}

export interface RepoMerge extends RepoModification {
  type: "merge";
  prId: string;
}
