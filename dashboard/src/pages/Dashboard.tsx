import RooflinePlot from "../components/RooflinePlot";
import { BarComparisonPlot } from "../components/BarPlot";
import FilterControls from "../components/FilterControls";
import { useEffect, useMemo, useState } from "react";
import type {
  AttentionKernel,
  ConvKernel,
  GemmKernel,
  Kernel,
  KernelType,
} from "../types";
import { loadResultCsv } from "../utils/csv";
import KernelView from "../components/KernelView";

export default function Dashboard() {
  const [kernels, setKernels] = useState<Kernel[]>([]);
  const [selectedKernelId, setSelectedKernelId] = useState<string | null>(null);
  const [kernelType, setKernelType] = useState<KernelType>("attention");
  const [selectedBackends, setSelectedBackends] = useState<string[]>([]);
  const [selectedDtypes, setSelectedDtypes] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  useEffect(() => {
    async function fetchData() {
      const ireeKernels = await loadResultCsv(
        "iree",
        "attention",
        "/results/attention_iree.csv"
      );
      const waveKernels = await loadResultCsv(
        "wave",
        "attention",
        "/results/attention_wave.csv"
      );
      const ireeConvKernels = await loadResultCsv(
        "iree",
        "conv",
        "/results/conv_iree.csv"
      );
      const waveConvKernels = await loadResultCsv(
        "wave",
        "conv",
        "/results/conv_wave.csv"
      );
      // console.log(waveConvKernels);
      const kernels = ireeKernels.concat(
        waveKernels,
        ireeConvKernels,
        waveConvKernels
      );
      setKernels(kernels);
    }
    fetchData();
  }, []);

  useEffect(() => {
    const uniqueBackends = Array.from(new Set(kernels.map((k) => k.backend)));
    const uniqueDtypes = Array.from(new Set(kernels.map((k) => k.dtype)));
    const uniqueTags = Array.from(new Set(kernels.map((k) => k.tag)));

    setSelectedBackends(uniqueBackends);
    setSelectedDtypes(uniqueDtypes);
    setSelectedTags(uniqueTags);
  }, [kernels]);

  const filteredKernels = useMemo(() => {
    return kernels.filter(
      (k) =>
        selectedBackends.includes(k.backend) &&
        selectedDtypes.includes(k.dtype) &&
        selectedTags.includes(k.tag) &&
        kernelType === k.kernelType
    );
  }, [kernels, kernelType, selectedBackends, selectedDtypes, selectedTags]);

  const selectedKernel = useMemo(
    () => kernels.find((k) => k.id === selectedKernelId),
    [kernels, selectedKernelId]
  );

  const sameShapeKernels = useMemo(() => {
    if (!selectedKernel) return [];
    return kernels.filter((k) => {
      if (k.kernelType !== selectedKernel.kernelType) return false;
      if (k.kernelType === "gemm") {
        const gk = selectedKernel as GemmKernel;
        return k.M === gk.M && k.N === gk.N && k.K === gk.K;
      } else if (k.kernelType === "attention") {
        const ak = selectedKernel as AttentionKernel;
        return (
          k.B === ak.B &&
          k.M === ak.M &&
          k.N === ak.N &&
          k.K1 === ak.K1 &&
          k.K2 === ak.K2
        );
      } else if (k.kernelType === "conv") {
        const ck = selectedKernel as ConvKernel;
        return (
          k.B === ck.B &&
          k.H === ck.H &&
          k.W === ck.W &&
          k.C === ck.C &&
          k.P === ck.P &&
          k.Q === ck.Q &&
          k.F === ck.F &&
          k.S === ck.S
        );
      }
    });
  }, [kernels, selectedKernel]);

  return (
    <div className="p-6 space-y-8">
      <h1 className="text-2xl font-bold">Benchmarking Dashboard</h1>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <FilterControls
          kernels={kernels}
          kernelType={kernelType}
          setKernelType={setKernelType}
          selectedBackends={selectedBackends}
          setSelectedBackends={setSelectedBackends}
          selectedDtypes={selectedDtypes}
          setSelectedDtypes={setSelectedDtypes}
          selectedTags={selectedTags}
          setSelectedTags={setSelectedTags}
        />
      </div>
      <div className="flex flex-col lg:flex-row gap-6 items-center">
        <div className="w-full lg:w-[60%] flex flex-col items-center">
          <h2 className="text-xl mb-4 font-bold">Roofline Plot</h2>
          <RooflinePlot
            kernels={filteredKernels}
            setSelected={setSelectedKernelId}
            selectedKernel={selectedKernel}
          />
        </div>
        <div className="w-full lg:w-[40%] flex flex-col items-center">
          <h2 className="text-xl mb-4 font-bold">
            Average runtime
            {selectedKernel ? `: ${selectedKernel.name}` : " (microseconds)"}
          </h2>
          <BarComparisonPlot
            kernels={selectedKernelId ? sameShapeKernels : filteredKernels}
          />
        </div>
      </div>

      {selectedKernel && (
        <KernelView
          selectedKernel={selectedKernel}
          sameShapeKernels={sameShapeKernels}
          kernels={kernels}
          setSelected={setSelectedKernelId}
        />
      )}
    </div>
  );
}
