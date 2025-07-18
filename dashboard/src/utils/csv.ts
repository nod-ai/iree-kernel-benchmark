import { parse } from "papaparse";
import { v4 as uuidv4 } from "uuid";
import type { Kernel, KernelType } from "../types";

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
          let shape = {};
          let dtype = "";

          if (kernelType === "gemm") {
            dtype = row["dtype"];
            shape = {
              M: row["M"],
              N: row["N"],
              K: row["K"],
              transpose: row["tA"] + row["tB"],
            };
          } else if (kernelType === "attention") {
            dtype = row["dtype"];
            shape = {
              B: row["B"],
              M: row["M"],
              N: row["N"],
              K1: row["K1"],
              K2: row["K2"],
            };
          } else if (kernelType === "conv") {
            dtype = row["input_dtype"];
            shape = {
              B: row["B"],
              H: row["H"],
              W: row["W"],
              C: row["C"],
              P: row["P"],
              Q: row["Q"],
              F: row["F"],
              S: row["S"],
            };
          }

          const kernel: Kernel = {
            id: uuidv4(),
            backend,
            kernelType,
            dtype,
            shape,
            name: row["name"],
            tag: row["tag"],
            meanMicroseconds: row["mean_microseconds"],
            arithmeticIntensity: row["arithmetic_intensity"],
            tflops: row["tflops"],
          };
          results.push(kernel);
        }

        resolve(results);
      },
    });
  });
}

export async function fetchData() {
  const dataConfigs = [
    ["IREE MHA", "attention", "/results/attention/attention_iree.csv"],
    ["Wave MHA", "attention", "/results/attention/attention_wave.csv"],
    ["Wave GQA", "attention", "/results/attention/attention_wavegqa.csv"],
    // [
    //   "Wave GQA New",
    //   "attention",
    //   "/results/attention/attention_wavegqanew.csv",
    // ],
    ["IREE", "conv", "/results/conv/conv_iree.csv"],
    ["Wave", "conv", "/results/conv/conv_wave.csv"],
    ["IREE", "gemm", "/results/gemm/gemm_iree.csv"],
    ["Wave", "gemm", "/results/gemm/gemm_wave.csv"],
  ];
  const kernelRequests = dataConfigs.map(
    async ([backend, kernelType, csvPath]) =>
      await loadResultCsv(backend, kernelType as KernelType, csvPath)
  );
  const kernels = (await Promise.all(kernelRequests)).flat();
  return kernels;
}
