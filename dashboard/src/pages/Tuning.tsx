import { useEffect, useState } from "react";
import PageContainer from "../components/PageContainer";
import type { Kernel } from "../types";
import { fetchData } from "../utils/csv";
import { toTitleCase } from "../utils/utils";
import { getBackendColor } from "../utils/color";
import type { ColorInstance } from "color";
import Color from "color";

interface ItemTagProps {
  color?: ColorInstance | string;
  colorHash?: string;
  label: string;
}

function ItemTag({ color, colorHash, label }: ItemTagProps) {
  if (!color) {
    color = getBackendColor(colorHash || label);
  }
  const colorStr = Color(color).lighten(0.4).string();

  return (
    <div
      style={{ backgroundColor: colorStr }}
      className="rounded-md px-2 text-black"
    >
      {label}
    </div>
  );
}

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
          <div className="flex flex-row justify-between w-[100%] bg-gray-100 border border-gray-500 rounded-md py-1 px-4">
            <div className="flex flex-row gap-4">
              <ItemTag label={toTitleCase(k.kernelType)} />
              <ItemTag label={k.backend} />
              <>
                {Object.entries(k.shape).map(([dimName, dimValue]) => (
                  <ItemTag
                    label={`${dimName} = ${dimValue}`}
                    colorHash={`dim_${dimName}`}
                  />
                ))}
              </>
              <ItemTag label={`dtype = ${k.dtype}`} />
            </div>
          </div>
        ))}
      </div>
    </PageContainer>
  );
}
