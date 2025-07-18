import { useEffect, useRef } from "react";
import {
  Chart,
  ScatterController,
  LinearScale,
  LogarithmicScale,
  PointElement,
  Title,
  Tooltip,
  Legend,
  LineController,
  LineElement,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import type { Kernel } from "../types";
import { getBackendColor } from "../utils/color";

Chart.register(
  ScatterController,
  LinearScale,
  LogarithmicScale,
  PointElement,
  Title,
  Tooltip,
  Legend,
  LineController,
  LineElement,
  zoomPlugin
);

interface RooflinePlotProps {
  kernels: Kernel[];
  selectedKernel?: Kernel;
  setSelected: (kernelId: string | null) => void;
}

export default function RooflinePlot({
  kernels,
  setSelected,
  selectedKernel,
}: RooflinePlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) chartRef.current.destroy();

    const grouped = kernels.reduce<
      Record<string, { x: number; y: number; id: string; name: string }[]>
    >((acc, kernel) => {
      if (!acc[kernel.backend]) acc[kernel.backend] = [];
      acc[kernel.backend].push({
        x: kernel.arithmeticIntensity,
        y: kernel.tflops,
        id: kernel.id,
        name: kernel.name,
      });
      return acc;
    }, {});

    const datasets = Object.entries(grouped).map(([backend, points]) => ({
      label: backend,
      data: points.filter((point) => point.id !== selectedKernel?.id),
      borderColor: getBackendColor(backend).string(),
      backgroundColor: selectedKernel
        ? "rgba(200, 200, 200, 0.3)"
        : getBackendColor(backend).string(),
      showLine: false,
      pointRadius: 5,
    }));

    if (selectedKernel) {
      datasets.push({
        label: selectedKernel.backend,
        data: [
          {
            x: selectedKernel.arithmeticIntensity,
            y: selectedKernel.tflops,
            id: selectedKernel.id,
            name: selectedKernel.name,
          },
        ],
        borderColor: getBackendColor(selectedKernel.backend).string(),
        backgroundColor: getBackendColor(selectedKernel.backend).string(),
        showLine: false,
        pointRadius: 5,
      });
    }

    const xMin = Math.max(
      0.01,
      Math.min(...kernels.map((k) => k.arithmeticIntensity))
    );
    const xMax = Math.max(...kernels.map((k) => k.arithmeticIntensity)) * 2;

    const peakMemoryBandwidth = 5.3;
    const peakCompute = 1307.4;
    const xRoofline = Array.from(
      { length: 100 },
      (_, i) => xMin * Math.pow(xMax / xMin, i / 99)
    );

    const yMemory = xRoofline.map((x) => x * peakMemoryBandwidth);
    const yCompute = xRoofline.map(() => peakCompute);

    // Append memory bound line
    datasets.push({
      label: "Memory Bound",
      // type: "line",
      data: xRoofline.map((x, i) => ({ x, y: yMemory[i], id: "", name: "" })),
      borderColor: "#d62728", // red
      showLine: true,
      backgroundColor: "#d62728",
      // borderWidth: 2,
      // fill: false,
      pointRadius: 0,
    });

    // Append compute bound line
    datasets.push({
      label: "Compute Bound",
      // type: "line",
      data: xRoofline.map((x, i) => ({ x, y: yCompute[i], id: "", name: "" })),
      borderColor: "#2ca02c", // green
      showLine: true,
      backgroundColor: "#2ca02c",
      // borderWidth: 2,
      // fill: false,
      pointRadius: 0,
    });

    chartRef.current = new Chart(canvasRef.current, {
      type: "scatter",
      data: {
        datasets,
      },
      options: {
        responsive: true,
        scales: {
          x: {
            type: "logarithmic",
            title: {
              display: true,
              text: "Arithmetic Intensity (FLOP/byte)",
            },
          },
          y: {
            type: "logarithmic",
            title: {
              display: true,
              text: "Performance (TFLOP/s)",
            },
          },
        },
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const point = ctx.raw as any;
                return `${point.name}: (${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
              },
            },
          },
          zoom: {
            zoom: {
              wheel: {
                enabled: true,
              },
              pinch: {
                enabled: true,
              },
              mode: "xy",
            },
            pan: {
              enabled: true,
              mode: "xy",
            },
            limits: {
              x: { min: "original", max: "original" },
              y: { min: "original", max: "original" },
            },
          },
        },
        onClick: (event, elements) => {
          if (elements.length > 0) {
            const datasetIndex = elements[0].datasetIndex;
            const index = elements[0].index;
            const point = (
              chartRef.current?.data.datasets[datasetIndex].data as any[]
            )[index];
            if (point?.id) setSelected(point.id);
          } else {
            setSelected(null);
          }
        },
      },
    });
  }, [kernels, selectedKernel]);

  return <canvas ref={canvasRef} className="w-full h-[500px]" />;
}
