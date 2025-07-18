import { useEffect, useRef } from "react";
import {
  Chart,
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from "chart.js";
import type { Kernel } from "../types";
import { getBackendColor } from "../utils/color";
import { KERNEL_DIMS } from "../utils/utils";

Chart.register(
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title
);

interface BarComparisonPlotProps {
  kernels: Kernel[];
}

export function BarComparisonPlot({ kernels }: BarComparisonPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) chartRef.current.destroy();

    const hashKernel = (kernel: Kernel) =>
      `${kernel.kernelType}_` +
      KERNEL_DIMS[kernel.kernelType]
        .map((dimName) => `${dimName}${(kernel as any)[dimName]}`)
        .join("_");

    const backendShapes: Record<string, Set<string>> = {};
    for (const kernel of kernels) {
      const kernelHash = hashKernel(kernel);
      if (!backendShapes[kernel.backend])
        backendShapes[kernel.backend] = new Set<string>();
      backendShapes[kernel.backend].add(kernelHash);
    }

    const commonShapes =
      kernels.length > 0
        ? Object.values(backendShapes).reduce(
            (prev, curr) => new Set([...prev].filter((hash) => curr.has(hash)))
          )
        : new Set<string>();

    const backendGroups: Record<string, number[]> = {};

    for (const kernel of kernels) {
      if (!commonShapes.has(hashKernel(kernel))) continue;
      if (!backendGroups[kernel.backend]) backendGroups[kernel.backend] = [];
      backendGroups[kernel.backend].push(kernel.meanMicroseconds);
    }

    const labels = Object.keys(backendGroups);
    const data = labels.map((label) => {
      const values = backendGroups[label];
      return values.reduce((a, b) => a + b, 0) / values.length;
    });

    chartRef.current = new Chart(canvasRef.current, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Avg Mean Time (μs)",
            data,
            backgroundColor: labels.map((label) =>
              getBackendColor(label).string()
            ),
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: {
            title: {
              display: true,
              text: "Mean Time (μs)",
            },
            beginAtZero: true,
          },
        },
        plugins: {
          legend: {
            display: false,
          },
        },
      },
    });
  }, [kernels]);

  return <canvas ref={canvasRef} className="w-full h-[500px]" />;
}
