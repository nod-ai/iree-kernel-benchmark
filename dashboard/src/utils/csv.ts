import { parse } from "papaparse";
import { v4 as uuidv4 } from "uuid";
import type { Kernel, GemmKernel, AttentionKernel, KernelType } from "../types";

export async function loadResultCsv(
  backend: string,
  kernelType: KernelType,
  csvPath: string
): Promise<Kernel[]> {
  const response = await fetch(csvPath);
  const csvText = await response.text();

  return new Promise<Kernel[]>((resolve) => {
    const results: Kernel[] = [];

    parse(csvText, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: ({ data }) => {
        for (const row of data as any[]) {
          const common = {
            id: uuidv4(),
            backend,
            kernelType,
            name: row["name"],
            tag: row["tag"],
            dtype: row["dtype"],
            meanMicroseconds: row["mean_microseconds"],
            arithmeticIntensity: row["arithmetic_intensity"],
            tflops: row["tflops"],
          };

          if (kernelType === "gemm") {
            const kernel: GemmKernel = {
              ...common,
              kernelType: "gemm",
              M: row["M"],
              N: row["N"],
              K: row["K"],
            };
            results.push(kernel);
          } else if (kernelType === "attention") {
            const kernel: AttentionKernel = {
              ...common,
              kernelType: "attention",
              B: row["B"],
              M: row["M"],
              N: row["N"],
              K1: row["K1"],
              K2: row["K2"],
            };
            results.push(kernel);
          }
        }

        resolve(results);
      },
    });
  });
}
