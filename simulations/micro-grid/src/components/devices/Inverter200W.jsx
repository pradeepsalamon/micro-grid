import React, { useState, useEffect } from "react";

export default function Inverter200W({
  id,
  x,
  y,
  batteryVoltage = 12.5,
  loadPower = 0,
  onUpdate,
}) {
  const [isOn, setIsOn] = useState(true);
  const [outputVoltage, setOutputVoltage] = useState(230);
  const [temperature, setTemperature] = useState(32);
  const [fault, setFault] = useState(false);

  // ⚙️ Simulate inverter behavior
  useEffect(() => {
    let interval = setInterval(() => {
      if (!isOn) return;

      // Overload or low voltage protection
      if (loadPower > 200 || batteryVoltage < 10.8) {
        setFault(true);
        setOutputVoltage(0);
      } else {
        setFault(false);
        setOutputVoltage(230);
      }

      // Temperature simulation
      setTemperature((t) =>
        fault ? t + 0.5 : Math.max(28, t - 0.2)
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [isOn, loadPower, batteryVoltage]);

  const handleToggle = () => setIsOn((prev) => !prev);

  const statusColor = fault
    ? "text-red-400"
    : !isOn
    ? "text-gray-400"
    : "text-green-400";

  const caseColor = fault
    ? "from-red-900 to-red-700"
    : isOn
    ? "from-gray-800 to-gray-700"
    : "from-gray-700 to-gray-600";

  return (
    <div
      id={id}
      className={`relative w-64 h-36 rounded-lg border-2 border-gray-600 bg-gradient-to-b ${caseColor} shadow-lg`}
    >
      {/* Header */}
      <div className="text-center text-sm text-gray-300 font-semibold bg-gray-800 py-1 rounded-t-md">
        ⚙️ 200 W DC–AC Inverter
      </div>

      {/* Display Screen */}
      <div className="absolute left-4 top-8 w-56 h-20 bg-black border border-green-500 rounded-md p-2 text-green-400 font-mono text-xs leading-tight">
        <div className="flex justify-between">
          <span>Input: {batteryVoltage.toFixed(1)} V</span>
          <span>Temp: {temperature.toFixed(1)}°C</span>
        </div>
        <div className="flex justify-between mt-1">
          <span>Output: {outputVoltage.toFixed(0)} V AC</span>
          <span>Load: {loadPower.toFixed(0)} W</span>
        </div>
        <div
          className={`text-center mt-2 font-semibold ${statusColor}`}
        >
          {fault
            ? "FAULT / SHUTDOWN"
            : !isOn
            ? "OFF"
            : "RUNNING"}
        </div>
      </div>

      {/* Input Terminals (from battery) */}
      <div className="absolute -bottom-4 left-4 flex gap-4 text-center text-[10px] text-gray-300">
        <div className="flex flex-col items-center">
          <div
            className="w-3 h-3 bg-red-500 rounded-full border border-red-300 hover:scale-125 cursor-crosshair"
            title="Battery +"
          />
          <span>BAT+</span>
        </div>
        <div className="flex flex-col items-center">
          <div
            className="w-3 h-3 bg-gray-400 rounded-full border border-gray-200 hover:scale-125 cursor-crosshair"
            title="Battery −"
          />
          <span>BAT−</span>
        </div>
      </div>

      {/* Output Sockets */}
      <div className="absolute -bottom-4 right-6 flex gap-4 text-center text-[10px] text-gray-300">
        <div className="flex flex-col items-center">
          <div
            className="w-3 h-3 bg-yellow-300 rounded-full border border-yellow-200 hover:scale-125 cursor-crosshair"
            title="AC Live"
          />
          <span>LIVE</span>
        </div>
        <div className="flex flex-col items-center">
          <div
            className="w-3 h-3 bg-blue-400 rounded-full border border-blue-200 hover:scale-125 cursor-crosshair"
            title="AC Neutral"
          />
          <span>NEUT</span>
        </div>
      </div>

      {/* Power Button */}
      <button
        onClick={handleToggle}
        className={`absolute right-3 top-3 px-3 py-1 rounded text-xs font-semibold ${
          isOn
            ? "bg-emerald-600 hover:bg-emerald-500"
            : "bg-gray-600 hover:bg-gray-500"
        }`}
      >
        {isOn ? "Turn Off" : "Turn On"}
      </button>
    </div>
  );
}
