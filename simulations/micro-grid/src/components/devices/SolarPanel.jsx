import React, { useState, useRef, useEffect } from "react";

export default function SolarPanel({ id, irradiance = 0.8, onUpdate }) {
  const [localIrradiance, setLocalIrradiance] = useState(irradiance);
  const [showSlider, setShowSlider] = useState(false);
  const sliderRef = useRef();

  // Keep local value synced with parent
  useEffect(() => {
    setLocalIrradiance(irradiance);
  }, [irradiance]);

  // Update parent after small delay (prevents rapid updates)
  useEffect(() => {
    const timer = setTimeout(() => {
      onUpdate?.(id, { irradiance: localIrradiance });
    }, 100);
    return () => clearTimeout(timer);
  }, [localIrradiance]);

  const voltage = (12 + localIrradiance * 6).toFixed(1);
  const power = (localIrradiance * 100).toFixed(0);

  return (
    <div
      className="relative group select-none"
      onMouseEnter={() => setShowSlider(true)}
      onMouseLeave={() => setShowSlider(false)}
      title={`Irradiance: ${(localIrradiance * 100).toFixed(0)}%`}
    >
      {/* 🌞 Solar Panel visual */}
      <div
        className="relative w-40 h-24 rounded-md overflow-hidden border-2 border-sky-400 shadow-lg shadow-sky-400/20 transition-all duration-300"
        style={{
          background: `linear-gradient(to-br, rgba(0,0,60,${
            0.5 + localIrradiance * 0.5
          }) 30%, rgba(0,80,160,${0.8 + localIrradiance * 0.2}))`,
        }}
      >
        {/* Panel cells */}
        <div className="absolute inset-0 grid grid-cols-4 grid-rows-2 gap-[1px]">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="bg-blue-950/50" />
          ))}
        </div>

        {/* Terminals */}
        <div
          className="absolute -right-3 top-1/3 w-3 h-3 rounded-full bg-red-500 border border-red-300 cursor-crosshair hover:scale-125 transition-transform"
          title="Positive (+)"
        />
        <div
          className="absolute -right-3 bottom-1/3 w-3 h-3 rounded-full bg-gray-300 border border-gray-100 cursor-crosshair hover:scale-125 transition-transform"
          title="Negative (–)"
        />

        {/* Label */}
        <div className="absolute bottom-0 left-0 w-full bg-sky-900/80 text-xs text-center py-1 text-white">
          ☀️ Solar Panel
          <br />
          {voltage} V – {power} W
        </div>
      </div>

      {/* 🌤️ Irradiance control (hover popup) */}
      {showSlider && (
        <div
          ref={sliderRef}
          className="absolute left-1/2 -translate-x-1/2 -top-16 w-44 bg-slate-800/95 border border-slate-600 rounded-lg p-2 text-center shadow-lg z-40"
        >
          <div className="text-xs text-gray-300 mb-1">
            Irradiance: {(localIrradiance * 100).toFixed(0)}%
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={localIrradiance}
            onChange={(e) => setLocalIrradiance(parseFloat(e.target.value))}
            className="w-full accent-yellow-400 cursor-pointer"
          />
        </div>
      )}
    </div>
  );
}
