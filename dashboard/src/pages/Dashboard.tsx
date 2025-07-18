import RooflinePlot from "../components/RooflinePlot";
import { BarComparisonPlot } from "../components/BarPlot";
import FilterControls from "../components/FilterControls";
import { useEffect, useMemo, useState } from "react";
import type { Kernel, KernelType } from "../types";
import { fetchData, loadResultCsv } from "../utils/csv";
import KernelView from "../components/KernelView";
import Navbar from "../components/Navbar";
import PageContainer from "../components/PageContainer";
import { KERNEL_DIMS } from "../utils/utils";

export default function Dashboard() {
  const [kernels, setKernels] = useState<Kernel[]>([]);
  const [selectedKernelId, setSelectedKernelId] = useState<string | null>(null);
  const [kernelType, setKernelType] = useState<KernelType>("attention");
  const [selectedBackends, setSelectedBackends] = useState<string[]>([]);
  const [selectedDtypes, setSelectedDtypes] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  useEffect(() => {
    fetchData().then(setKernels);
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

  const filteredWaveKernels = useMemo(
    () => filteredKernels.filter((k) => k.backend.startsWith("wave")),
    [filteredKernels]
  );

  const selectedKernel = useMemo(
    () => kernels.find((k) => k.id === selectedKernelId),
    [kernels, selectedKernelId]
  );

  const sameShapeKernels = useMemo(() => {
    if (!selectedKernel) return [];
    return kernels.filter((k) => {
      if (k.kernelType !== selectedKernel.kernelType) return false;
      return KERNEL_DIMS[k.kernelType].every(
        (dimName) => k.shape[dimName] === selectedKernel.shape[dimName]
      );
    });
  }, [kernels, selectedKernel]);

  return (
    <PageContainer activePage="dashboard">
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
          {!selectedKernel && filteredWaveKernels.length > 0 && (
            <button className="px-4 mt-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
              Tune {filteredWaveKernels.length} Wave Kernels
            </button>
          )}
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
    </PageContainer>
  );
}
