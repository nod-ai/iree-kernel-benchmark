import { useEffect, useRef } from "react";
import { Chart, BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend, Title } from "chart.js";
import type { Kernel } from "../types";
import { BACKEND_COLORS } from "./RooflinePlot";

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

interface BarComparisonPlotProps {
  kernels: Kernel[];
}

export function BarComparisonPlot({ kernels }: BarComparisonPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) chartRef.current.destroy();

    const backendGroups: Record<string, number[]> = {};

    for (const kernel of kernels) {
      if (!backendGroups[kernel.backend]) backendGroups[kernel.backend] = [];
      backendGroups[kernel.backend].push(kernel.meanMicroseconds);
    }

    const labels = Object.keys(backendGroups);
    const data = labels.map(label => {
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
            backgroundColor: labels.map(label => BACKEND_COLORS[label] || "#888"),
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