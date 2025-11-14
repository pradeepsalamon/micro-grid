import React, { useState, useRef, useEffect } from "react";

export default function DeviceWrapper({
  id,
  x = 0,
  y = 0,
  rotation = 0,
  flipped = false,
  size = 1,
  onUpdate,
  children,
}) {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef();
  const moveStep = 10;

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setShowMenu(false);
      }
    };
    if (showMenu) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showMenu]);

  const handleContextMenu = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setShowMenu((prev) => !prev);
  };

  const safeUpdate = (updates) => {
    if (typeof onUpdate === "function") onUpdate(id, updates);
  };

  return (
    <div
      className="absolute"
      style={{
        left: `${x}px`,
        top: `${y}px`,
        transform: `rotate(${rotation}deg) scaleX(${flipped ? -1 : 1}) scale(${size})`,
        cursor: "default",
        transition: "transform 0.15s ease, left 0.1s linear, top 0.1s linear",
        zIndex: 20,
      }}
      onContextMenu={handleContextMenu}
    >
      {/* Render actual device content */}
      {children}

      {/* Context Menu */}
      {showMenu && (
        <div
          ref={menuRef}
          className="absolute -right-36 top-0 bg-slate-800 border border-slate-600 rounded-md shadow-lg z-50 p-2 text-xs"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-center text-gray-300 mb-1 font-semibold">
            Move
          </div>

          <div className="grid grid-cols-3 gap-1 justify-items-center mb-2">
            <div />
            <button
              onClick={() => safeUpdate({ y: y - moveStep })}
              className="bg-slate-700 hover:bg-slate-600 w-6 h-6 rounded"
            >
              ↑
            </button>
            <div />
            <button
              onClick={() => safeUpdate({ x: x - moveStep })}
              className="bg-slate-700 hover:bg-slate-600 w-6 h-6 rounded"
            >
              ←
            </button>
            <div />
            <button
              onClick={() => safeUpdate({ x: x + moveStep })}
              className="bg-slate-700 hover:bg-slate-600 w-6 h-6 rounded"
            >
              →
            </button>
            <div />
            <button
              onClick={() => safeUpdate({ y: y + moveStep })}
              className="bg-slate-700 hover:bg-slate-600 w-6 h-6 rounded"
            >
              ↓
            </button>
            <div />
          </div>

          <hr className="border-slate-700 my-2" />

          <button
            onClick={() =>
              safeUpdate({ rotation: (rotation + 45) % 360 })
            }
            className="block w-full text-left hover:bg-slate-700 px-2 py-1 rounded mb-1"
          >
            Rotate 45°
          </button>

          <button
            onClick={() => safeUpdate({ flipped: !flipped })}
            className="block w-full text-left hover:bg-slate-700 px-2 py-1 rounded mb-1"
          >
            Flip
          </button>

          <button
            onClick={() => safeUpdate({ size: Math.max(0.5, size - 0.1) })}
            className="block w-full text-left hover:bg-slate-700 px-2 py-1 rounded mb-1"
          >
            Size −
          </button>

          <button
            onClick={() => safeUpdate({ size: Math.min(1.5, size + 0.1) })}
            className="block w-full text-left hover:bg-slate-700 px-2 py-1 rounded"
          >
            Size +
          </button>
        </div>
      )}
    </div>
  );
}
