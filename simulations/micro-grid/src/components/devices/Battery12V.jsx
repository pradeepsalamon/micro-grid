import React, { useState, useEffect } from "react";

export default function Battery12V({
  id,
  x,
  y,
  voltage = 12.4,
  capacity = 100, // percentage
  charging = false,
  onUpdate,
}) {
  const [localVoltage, setLocalVoltage] = useState(voltage);
  const [localCapacity, setLocalCapacity] = useState(capacity);

  // Simulate charge/discharge
  useEffect(() => {
    const timer = setInterval(() => {
      setLocalVoltage((prev) => {
        let newV = prev;
        if (charging) newV = Math.min(14.4, prev + 0.01);
        else newV = Math.max(11.8, prev - 0.003);
        onUpdate?.(id, { voltage: newV });
        return newV;
      });

      setLocalCapacity((prev) => {
        let newC = prev;
        if (charging) newC = Math.min(100, prev + 0.05);
        else newC = Math.max(0, prev - 0.02);
        onUpdate?.(id, { capacity: newC });
        return newC;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [charging]);

  const chargeColor =
    localCapacity > 80
      ? "bg-green-500"
      : localCapacity > 40
      ? "bg-yellow-500"
      : "bg-red-500";

  return (
    <div
      className="relative w-40 h-28 rounded-lg border-2 border-gray-500 bg-gradient-to-b from-gray-800 to-gray-700 shadow-lg shadow-slate-800/40"
      title={`Battery: ${localVoltage.toFixed(2)} V (${localCapacity.toFixed(
        0
      )}%)`}
    >
      {/* Top section with terminals */}
      <div className="absolute -top-3 left-1/2 -translate-x-1/2 flex gap-4">
        <div
          className="w-4 h-2 bg-red-500 border border-red-300 rounded-sm cursor-crosshair hover:scale-110"
          title="Battery +"
        />
        <div
          className="w-4 h-2 bg-black border border-gray-600 rounded-sm cursor-crosshair hover:scale-110"
          title="Battery −"
        />
      </div>

      {/* Label */}
      <div className="text-center mt-6 text-sm text-gray-200 font-semibold">
        🔋 12 V Battery
      </div>

      {/* Charge indicator bar */}
      <div className="absolute bottom-4 left-4 w-32 h-4 border border-gray-400 rounded-sm bg-gray-900 overflow-hidden">
        <div
          className={`${chargeColor} h-full transition-all duration-500`}
          style={{ width: `${localCapacity}%` }}
        ></div>
      </div>

      {/* Voltage + percentage display */}
      <div className="absolute bottom-1 left-0 w-full text-center text-xs text-gray-300 font-mono">
        {localVoltage.toFixed(2)} V — {localCapacity.toFixed(0)}%
      </div>
    </div>
  );
}
