import React, { useState, useEffect } from "react";

export default function PWMChargeController({
  id,
  solarInput = 18,
  batteryVoltage = 12.4,
  loadPower = 25,
  onUpdate,
}) {
  const [charging, setCharging] = useState(true);
  const [dutyCycle, setDutyCycle] = useState(0.6);
  const [temp, setTemp] = useState(32);

  useEffect(() => {
    if (solarInput > batteryVoltage + 1 && batteryVoltage < 14.4) {
      setCharging(true);
      setDutyCycle((prev) => Math.min(1, prev + 0.01));
    } else {
      setCharging(false);
      setDutyCycle((prev) => Math.max(0, prev - 0.02));
    }
  }, [solarInput, batteryVoltage]);

  const displayColor = charging ? "text-green-400" : "text-yellow-400";

  return (
    <div className="relative w-64 h-40 bg-gray-900 rounded-xl border-2 border-gray-600 shadow-lg shadow-slate-800/40">
      <div className="text-center text-sm text-gray-300 font-semibold bg-gray-700 py-1 rounded-t-lg">
        ⚡ PWM Solar Charge Controller
      </div>

      <div className="absolute left-4 top-8 w-56 h-20 bg-black border border-green-500 rounded-md p-2 text-green-400 font-mono text-xs leading-tight">
        <div className="flex justify-between">
          <span>Solar: {solarInput.toFixed(1)} V</span>
          <span>Temp: {temp.toFixed(1)}°C</span>
        </div>
        <div className="flex justify-between mt-1">
          <span>Battery: {batteryVoltage.toFixed(2)} V</span>
          <span>PWM: {(dutyCycle * 100).toFixed(0)}%</span>
        </div>
        <div className="flex justify-between mt-1">
          <span>Load: {loadPower} W</span>
          <span className={`${displayColor} font-semibold ${charging ? "animate-pulse" : ""}`}>
            {charging ? "CHARGING" : "IDLE"}
          </span>
        </div>
      </div>

      {/* Terminals */}
      <div className="absolute -bottom-4 left-4 flex gap-6 text-center text-[10px] text-gray-300">
        {["SOLAR", "BATT", "LOAD"].map((label, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <div className="flex gap-1">
              <div className="w-3 h-3 rounded-full bg-red-500 border border-red-300" />
              <div className="w-3 h-3 rounded-full bg-gray-400 border border-gray-200" />
            </div>
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
