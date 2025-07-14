import type { Kernel, KernelType } from "../types";


interface Props {
  kernels: Kernel[];
  kernelType: KernelType;
  setKernelType: (type: KernelType) => void;
  selectedBackends: string[];
  setSelectedBackends: (values: string[]) => void;
  selectedDtypes: string[];
  setSelectedDtypes: (values: string[]) => void;
  selectedTags: string[];
  setSelectedTags: (values: string[]) => void;
}

export default function FilterControls({
  kernels,
  kernelType,
  setKernelType,
  selectedBackends,
  setSelectedBackends,
  selectedDtypes,
  setSelectedDtypes,
  selectedTags,
  setSelectedTags,
}: Props) {
  const backends = Array.from(new Set(kernels.map(k => k.backend)));
  const dtypes = Array.from(new Set(kernels.map(k => k.dtype)));
  const tags = Array.from(new Set(kernels.map(k => k.tag)));

  function handleToggle(value: string, selected: string[], setSelected: (v: string[]) => void) {
    if (selected.includes(value)) {
      setSelected(selected.filter(v => v !== value));
    } else {
      setSelected([...selected, value]);
    }
  }

  return (
    <div className="flex flex-wrap gap-6 items-center">
      <div className="flex gap-2 items-center">
        <span className="font-semibold">Kernel Type:</span>
        {["gemm", "conv", "attention"].map(type => (
          <button
            key={type}
            className={`px-3 py-1 rounded border ${kernelType === type ? "bg-blue-500 text-white" : "bg-white border-gray-300"}`}
            onClick={() => setKernelType(type as KernelType)}
          >
            {type}
          </button>
        ))}
      </div>

      <div className="flex gap-2 items-center">
        <span className="font-semibold">Backends:</span>
        {backends.map(b => (
          <button
            key={b}
            className={`px-2 py-1 rounded border ${selectedBackends.includes(b) ? "bg-blue-100 border-blue-500" : "bg-white border-gray-300"}`}
            onClick={() => handleToggle(b, selectedBackends, setSelectedBackends)}
          >
            {b}
          </button>
        ))}
      </div>

      <div className="flex gap-2 items-center">
        <span className="font-semibold">Data Types:</span>
        {dtypes.map(d => (
          <button
            key={d}
            className={`px-2 py-1 rounded border ${selectedDtypes.includes(d) ? "bg-blue-100 border-blue-500" : "bg-white border-gray-300"}`}
            onClick={() => handleToggle(d, selectedDtypes, setSelectedDtypes)}
          >
            {d}
          </button>
        ))}
      </div>

      <div className="flex gap-2 items-center">
        <span className="font-semibold">Tags:</span>
        {tags.map(t => (
          <button
            key={t}
            className={`px-2 py-1 rounded border ${selectedTags.includes(t) ? "bg-blue-100 border-blue-500" : "bg-white border-gray-300"}`}
            onClick={() => handleToggle(t, selectedTags, setSelectedTags)}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}