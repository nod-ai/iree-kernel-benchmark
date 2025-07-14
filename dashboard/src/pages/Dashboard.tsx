import RooflinePlot from "../components/RooflinePlot";
import { BarComparisonPlot } from "../components/BarPlot";
import FilterControls from "../components/FilterControls";
import { useEffect, useMemo, useState } from "react";
import type { AttentionKernel, GemmKernel, Kernel, KernelType } from "../types";
import { loadResultCsv } from "../utils/csv";

export default function Dashboard() {
  const [kernels, setKernels] = useState<Kernel[]>([]);
  const [selectedKernelId, setSelectedKernelId] = useState<string | null>(null);
  const [kernelType, setKernelType] = useState<KernelType>("attention");
  const [selectedBackends, setSelectedBackends] = useState<string[]>([]);
  const [selectedDtypes, setSelectedDtypes] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  useEffect(() => {
    async function fetchData() {
      const ireeKernels = await loadResultCsv('iree', 'attention', '/results/attention_iree.csv');
      const waveKernels = await loadResultCsv('wave', 'attention', '/results/attention_wave.csv');
      const kernels = ireeKernels.concat(waveKernels);
      setKernels(kernels);
    }
    fetchData();
  }, []);

  useEffect(() => {
    const uniqueBackends = Array.from(new Set(kernels.map(k => k.backend)));
    const uniqueDtypes = Array.from(new Set(kernels.map(k => k.dtype)));
    const uniqueTags = Array.from(new Set(kernels.map(k => k.tag)));

    setSelectedBackends(uniqueBackends);
    setSelectedDtypes(uniqueDtypes);
    setSelectedTags(uniqueTags);
  }, [kernels]);

  const filteredKernels = useMemo(() => {
    return kernels.filter(k =>
      selectedBackends.includes(k.backend) &&
      selectedDtypes.includes(k.dtype) &&
      selectedTags.includes(k.tag)
    );
  }, [kernels, selectedBackends, selectedDtypes, selectedTags]);

  const selectedKernel = useMemo(
    () => kernels.find(k => k.id === selectedKernelId),
    [kernels, selectedKernelId]
  );

  const sameShapeKernels = useMemo(() => {
    if (!selectedKernel) return [];
    return kernels.filter(k => {
      if (k.kernelType !== selectedKernel.kernelType) return false;
      if (k.kernelType === "gemm") {
        const gk = selectedKernel as GemmKernel;
        return k.M === gk.M && k.N === gk.N && k.K === gk.K;
      } else {
        const ak = selectedKernel as AttentionKernel;
        return (
          k.B === ak.B &&
          k.M === ak.M &&
          k.N === ak.N &&
          k.K1 === ak.K1 &&
          k.K2 === ak.K2
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
          <h2 className="text-xl mb-4 font-bold">Average runtime{selectedKernel ? `: ${selectedKernel.name}` : ' (microseconds)'}</h2>
          <BarComparisonPlot 
            kernels={selectedKernelId ? sameShapeKernels : filteredKernels} 
          />
        </div>
      </div>

      {selectedKernel && (
        <div className="mt-8 border-t pt-6 space-y-4">
          <h2 className="text-xl font-semibold">Selected Kernel Details</h2>

          <div className="flex flex-wrap gap-4 items-center">
            {selectedKernel.kernelType === "gemm" ? (
              <>
                <div>Type: GEMM</div>
                <div>M: <select defaultValue={selectedKernel.M}><option>{selectedKernel.M}</option></select></div>
                <div>N: <select defaultValue={selectedKernel.N}><option>{selectedKernel.N}</option></select></div>
                <div>K: <select defaultValue={selectedKernel.K}><option>{selectedKernel.K}</option></select></div>
              </>
            ) : (
              <>
                <div>Type: Attention</div>
                <div>B: <select defaultValue={selectedKernel.B}><option>{selectedKernel.B}</option></select></div>
                <div>M: <select defaultValue={selectedKernel.M}><option>{selectedKernel.M}</option></select></div>
                <div>N: <select defaultValue={selectedKernel.N}><option>{selectedKernel.N}</option></select></div>
                <div>K1: <select defaultValue={selectedKernel.K1}><option>{selectedKernel.K1}</option></select></div>
                <div>K2: <select defaultValue={selectedKernel.K2}><option>{selectedKernel.K2}</option></select></div>
              </>
            )}
            <div>dtype: <select defaultValue={selectedKernel.dtype}><option>{selectedKernel.dtype}</option></select></div>
            <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Tune Kernel</button>
          </div>

          <div className="space-y-2">
            <h3 className="font-semibold">Performance Metrics by Backend:</h3>
            <table className="table-auto w-full border text-sm">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border px-2 py-1 text-left">Backend</th>
                  <th className="border px-2 py-1 text-left">Arithmetic Intensity</th>
                  <th className="border px-2 py-1 text-left">Mean Time (μs)</th>
                  <th className="border px-2 py-1 text-left">TFLOP/s</th>
                </tr>
              </thead>
              <tbody>
                {sameShapeKernels.map(k => (
                  <tr key={k.id}>
                    <td className="border px-2 py-1">{k.backend}</td>
                    <td className="border px-2 py-1">{k.arithmeticIntensity.toFixed(2)}</td>
                    <td className="border px-2 py-1">{k.meanMicroseconds.toFixed(2)}</td>
                    <td className="border px-2 py-1">{k.tflops.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}