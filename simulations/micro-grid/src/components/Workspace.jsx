import React, { useEffect, useState } from "react";
import DeviceWrapper from "./DeviceWrapper";
import SolarPanel from "./devices/SolarPanel";
import PWMChargeController from "./devices/PWMChargeController";
import Battery12V from "./devices/Battery12V";
import Inverter200W from "./devices/Inverter200W";



export default function Workspace() {
    const [devices, setDevices] = useState(() => {
        const saved = localStorage.getItem("static-workspace");
        if (saved) {
            try {
                return JSON.parse(saved);
            } catch {
                console.warn("Corrupted workspace data. Resetting...");
            }
        }
        return [
            { id: "solar-1", type: "solar", x: 200, y: 100, rotation: 0, flipped: false, size: 1 },
            { id: "pwm-1", type: "pwm", x: 500, y: 100, rotation: 0, flipped: false, size: 1 },
            { id: "battery-1", type: "battery", x: 600, y: 250, rotation: 0, flipped: false, size: 1 },
            { id: "inverter-1", type: "inverter", x: 850, y: 250 },
        ];
    });

    const [running, setRunning] = useState(false);
    const [speed, setSpeed] = useState(1);

    useEffect(() => {
        localStorage.setItem("static-workspace", JSON.stringify(devices));
    }, [devices]);

    const updateDevice = (id, updates) => {
        setDevices((prev) =>
            prev.map((d) => (d.id === id ? { ...d, ...updates } : d))
        );
    };

    const maxX = Math.max(...devices.map((d) => d.x + 400), 2000);
    const maxY = Math.max(...devices.map((d) => d.y + 400), 1000);

    return (
        <div className="relative h-screen w-screen bg-slate-900 text-white overflow-hidden">
            <div className="absolute inset-0 overflow-auto" style={{ padding: "40px" }}>
                <div
                    className="relative"
                    style={{
                        width: `${maxX}px`,
                        height: `${maxY}px`,
                        backgroundImage:
                            "linear-gradient(to right, rgba(255,255,255,0.07) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.07) 1px, transparent 1px)",
                        backgroundSize: "40px 40px",
                    }}
                >
                    {devices.map((dev) => (
                        <DeviceWrapper
                            key={dev.id}
                            id={dev.id}
                            x={dev.x}
                            y={dev.y}
                            rotation={dev.rotation || 0}
                            flipped={dev.flipped || false}
                            size={dev.size || 1}
                            onUpdate={updateDevice}
                        >
                            {dev.type === "solar" ? (
                                <SolarPanel id={dev.id} irradiance={dev.irradiance || 0.8} onUpdate={updateDevice} />
                            ) : dev.type === "pwm" ? (
                                <PWMChargeController id={dev.id} onUpdate={updateDevice} />
                            ) : dev.type === "battery" ? (
                                <Battery12V id={dev.id} onUpdate={updateDevice} />
                            ) : dev.type === "inverter" ? (
                                <Inverter200W
                                    id={dev.id}
                                    batteryVoltage={dev.batteryVoltage || 12.5}
                                    loadPower={dev.loadPower || 0}
                                    onUpdate={updateDevice}
                                />
                            ) : null}

                        </DeviceWrapper>
                    ))}
                </div>
            </div>

            {/* Control bar */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-800/80 backdrop-blur-lg px-6 py-3 rounded-xl flex items-center gap-4 border border-slate-600 shadow-lg z-50">
                <button
                    onClick={() => setRunning(!running)}
                    className={`px-6 py-2 rounded-md font-semibold ${running ? "bg-red-600 hover:bg-red-500" : "bg-emerald-600 hover:bg-emerald-500"
                        }`}
                >
                    {running ? "⏹ Stop" : "▶️ Start"}
                </button>
                <div className="flex items-center gap-2">
                    <label>Speed:</label>
                    <input
                        type="range"
                        min="0.5"
                        max="2"
                        step="0.1"
                        value={speed}
                        onChange={(e) => setSpeed(e.target.value)}
                        className="accent-emerald-400"
                    />
                    <span>{speed}×</span>
                </div>
            </div>
            <button
                onClick={() => {
                    localStorage.removeItem("static-workspace");
                    window.location.reload();
                }}
                className="absolute top-4 right-4 bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded shadow"
            >
                🗑 Reset Workspace
            </button>

        </div>
    );
}
