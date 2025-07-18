/* Kernels */

export type KernelType = "gemm" | "attention" | "conv";

export interface Kernel {
  id: string;
  backend: string;
  kernelType: KernelType;
  name: string;
  tag: string;
  dtype: string;
  meanMicroseconds: number;
  arithmeticIntensity: number;
  tflops: number;
  shape: Record<string, any>;
}

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
