import React from "react";
import { Info } from "lucide-react";

export default function GalaxySectorHud({
  currentSector,
  isLoggedIn,
  exploration,
}) {
  return (
    <div className="absolute top-8 right-8 z-50 text-right">
      <p className="text-[10px] tracking-[0.5em] text-white/30 uppercase font-black mb-1">
        Navigation System
      </p>
      <div className="flex items-center justify-end gap-3 mb-2">
        <div className="h-[2px] w-8 bg-[var(--primary)] shadow-[0_0_10px_var(--primary-glow)]" />
        <p
          className="text-xl font-light text-white uppercase"
          style={{ fontFamily: "Arkhip" }}
        >
          {currentSector} SECTOR
        </p>
      </div>
      {isLoggedIn && (
        <>
          <p className="text-[9px] tracking-[0.4em] text-[var(--success)] uppercase font-bold tabular-nums">
            Matrix Explored: {exploration.toFixed(5)}%
          </p>
        </>
      )}
      <p className="text-[7px] tracking-[0.3em] text-white/20 uppercase font-bold mt-1">
        Universal Signals: 6,174,821 Entry Total
      </p>

      {/* NEURAL INTEGRITY DISCLAIMER */}
      <div className="group relative mt-4 flex items-center justify-end gap-2 cursor-help">
        <p className="text-[7px] tracking-[0.2em] text-white/10 uppercase font-bold opacity-0 group-hover:opacity-100 transition-opacity max-w-[150px] leading-relaxed">
          Vector boundaries are fluid. In high-dimensional space, clusters bleed
          together and outliers may appear in unexpected sectors.
        </p>
        <Info
          size={10}
          className="text-white/20 group-hover:text-[var(--primary)] transition-colors"
        />
      </div>
    </div>
  );
}
