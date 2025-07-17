import React from "react";

interface ChangeStatBarProps {
  kernelType: string;
  change: number;
}

const MAX_BAR_WIDTH = 100; // 100px left/right (200 total bar area)

const ChangeStatBar: React.FC<ChangeStatBarProps> = ({
  kernelType,
  change,
}) => {
  const isPositive = change > 0;
  const pct = Math.min(Math.abs(change), 100);
  const barWidth = (pct / 100) * MAX_BAR_WIDTH;

  return (
    <div className="flex items-center w-[320px] gap-2">
      {/* Kernel name */}
      <div className="w-[60px] text-sm text-gray-700 text-right pr-2">
        {kernelType}
      </div>

      {/* Bar visualization */}
      <div className="relative flex items-center w-[200px] h-4">
        {/* Center anchor */}
        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-300" />

        {/* Red (left) */}
        {!isPositive && (
          <div
            className="bg-red-500 h-full absolute right-1/2"
            style={{ width: `${barWidth}px` }}
          />
        )}

        {/* Green (right) */}
        {isPositive && (
          <div
            className="bg-green-500 h-full absolute left-1/2"
            style={{ width: `${barWidth}px` }}
          />
        )}
      </div>

      {/* Label outside bar */}
      <div className="w-[60px] text-sm font-medium">
        <span
          className={
            isPositive ? "text-green-700 text-left" : "text-red-700 text-right"
          }
        >
          {change.toFixed(1)}%
        </span>
      </div>
    </div>
  );
};

export default ChangeStatBar;
