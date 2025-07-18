import { useEffect, useState } from "react";
import PageContainer from "../components/PageContainer";
import type { Kernel } from "../types";
import { fetchData } from "../utils/csv";
import { toTitleCase } from "../utils/utils";
import { getColor, lighten } from "../utils/color";

export default function Tuning() {
  const [kernels, setKernels] = useState<Kernel[]>([]);
  const [query, setQuery] = useState<string>("");

  useEffect(() => {
    fetchData().then(setKernels);
  }, []);

  return (
    <PageContainer activePage="tune">
      <div className="flex flex-col w-[100%] gap-1 px-4">
        {kernels.map((k) => (
          <div
            style={{ backgroundColor: lighten(getColor(k.backend), 0.95) }}
            className="flex flex-row justify-between w-[100%] border border-gray-500 rounded-md py-1 px-4"
          >
            <div className="flex flex-row gap-4">
              <div>{k.backend}</div>
              <div>{toTitleCase(k.kernelType)}</div>
            </div>
          </div>
        ))}
      </div>
    </PageContainer>
  );
}
