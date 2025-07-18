import type { Kernel } from "../types";
import { KERNEL_DIMS, toTitleCase } from "../utils/utils";

interface ShapeSelectorProps {
  selectedKernel: Kernel;
  kernels: Kernel[];
  setSelected: (kernelId: string | null) => void;
  dimensions: string[];
}

function ShapeSelector({
  dimensions,
  kernels,
  setSelected,
  selectedKernel,
}: ShapeSelectorProps) {
  const selection = dimensions.map((dimension) => ({
    name: dimension,
    value: selectedKernel.shape[dimension],
  }));

  const uniqueElements = (array: any[]) => {
    return Array.from(new Set(array));
  };

  const filterKernels = (dimName: string) => {
    let filteredKernels = kernels;

    for (let dim of selection) {
      if (dim.name === dimName) {
        return filteredKernels;
      }
      filteredKernels = filteredKernels.filter(
        (kernel) => kernel.shape[dim.name] === dim.value
      );
    }

    return [];
  };

  const filterDim = (dimName: string) => {
    return uniqueElements(
      filterKernels(dimName).map((kernel) =>
        dimName === "dtype" ? kernel.dtype : kernel.shape[dimName]
      )
    );
  };

  const setDim = (dimName: string, dimValue: any) => {
    let filteredKernels = filterKernels(dimName);
    filteredKernels = filteredKernels.filter((kernel) =>
      dimName === "dtype"
        ? kernel.dtype === dimValue
        : kernel.shape[dimName] === parseInt(dimValue)
    );
    if (filteredKernels.length > 0) setSelected(filteredKernels[0].id);
    else setSelected(null);
  };

  return (
    <>
      {selection.map((dim) => (
        <div>
          {dim.name}:
          <select
            value={selectedKernel.shape[dim.name]}
            onInput={(e) => {
              setDim(dim.name, e.currentTarget.value);
            }}
          >
            {filterDim(dim.name).map((dimValue) => (
              <option>{dimValue}</option>
            ))}
          </select>
        </div>
      ))}
    </>
  );
}

interface KernelViewProps {
  selectedKernel: Kernel;
  kernels: Kernel[];
  setSelected: (kernelId: string | null) => void;
  sameShapeKernels: Kernel[];
}

export default function KernelView({
  selectedKernel,
  kernels,
  setSelected,
  sameShapeKernels,
}: KernelViewProps) {
  return (
    <div className="mt-8 border-t pt-6 space-y-4">
      <h2 className="text-xl font-semibold">Selected Kernel Details</h2>

      <div className="flex flex-wrap gap-4 items-center">
        <div>Type: {toTitleCase(selectedKernel.kernelType)}</div>
        <ShapeSelector
          dimensions={KERNEL_DIMS[selectedKernel.kernelType]}
          kernels={kernels.filter(
            (k) => k.kernelType === selectedKernel.kernelType
          )}
          setSelected={setSelected}
          selectedKernel={selectedKernel}
        />
        <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
          Tune Kernel
        </button>
      </div>

      <div className="space-y-2">
        <h3 className="font-semibold">Performance Metrics by Backend:</h3>
        <table className="table-auto w-full border text-sm">
          <thead>
            <tr className="bg-gray-100">
              <th className="border px-2 py-1 text-left">Backend</th>
              <th className="border px-2 py-1 text-left">
                Arithmetic Intensity
              </th>
              <th className="border px-2 py-1 text-left">Mean Time (μs)</th>
              <th className="border px-2 py-1 text-left">TFLOP/s</th>
            </tr>
          </thead>
          <tbody>
            {sameShapeKernels.map((k) => (
              <tr key={k.id}>
                <td className="border px-2 py-1">{k.backend}</td>
                <td className="border px-2 py-1">
                  {k.arithmeticIntensity.toFixed(2)}
                </td>
                <td className="border px-2 py-1">
                  {k.meanMicroseconds.toFixed(2)}
                </td>
                <td className="border px-2 py-1">{k.tflops.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
