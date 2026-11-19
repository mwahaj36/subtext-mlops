import { Keyboard, MousePointer2 } from "lucide-react";

export default function GalaxyLegend({ isLoggedIn }) {
  return (
    <>
      <div className="absolute bottom-12 right-12 z-50 flex flex-col gap-2 p-6 bg-black/60 border border-white/10 backdrop-blur-3xl rounded-sm text-[9px] tracking-widest text-white/60 uppercase font-bold min-w-[200px]">
        <p className="text-[var(--primary)] mb-2 font-black tracking-[0.3em]">
          Flight Controls
        </p>
        <div className="flex justify-between gap-8 items-center border-b border-white/5 pb-2">
          <span className="flex items-center gap-2 text-white">
            <Keyboard size={10} /> WASD
          </span>
          <span>Move</span>
        </div>
        <div className="flex justify-between gap-8 items-center border-b border-white/5 pb-2">
          <span className="flex items-center gap-2 text-white">
            <Keyboard size={10} /> QE
          </span>
          <span>Up / Down</span>
        </div>
        <div className="flex justify-between gap-8 items-center border-b border-white/5 pb-2">
          <span className="flex items-center gap-2 text-white">
            <Keyboard size={10} /> SHIFT
          </span>
          <span>Neural Sprint</span>
        </div>
        <div className="flex justify-between gap-8 items-center border-b border-white/5 pb-2">
          <span className="flex items-center gap-2 text-white">
            <MousePointer2 size={10} /> LEFT CLICK
          </span>
          <span>Signal Lock</span>
        </div>
        <div className="flex justify-between gap-8 items-center border-b border-white/5 pb-2">
          <span className="flex items-center gap-2 text-white">
            <MousePointer2 size={10} /> RIGHT CLICK
          </span>
          <span className="text-[var(--primary)]">Emergency Purge</span>
        </div>
        <div className="flex justify-between gap-8 items-center pt-2">
          <span className="flex items-center gap-2 text-white">
            <Keyboard size={10} /> ESC
          </span>
          <span>Unlock Cursor</span>
        </div>
      </div>

      <div className="absolute bottom-12 left-12 z-50 flex flex-col gap-3 p-4 bg-black/40 border border-white/10 backdrop-blur-3xl rounded-sm font-bold">
        {isLoggedIn && (
          <>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-[var(--primary)] shadow-[0_0_10px_var(--primary-glow)]" />
              <span className="text-[8px] tracking-[0.2em] text-white/60 uppercase">
                ◈ Favourite
              </span>
            </div>
            <div className="flex items-center gap-3 ml-4 border-l border-white/10 pl-3">
              <div className="h-[1px] w-4 bg-[var(--primary)] opacity-50" />
              <span className="text-[7px] tracking-[0.2em] text-white/40 uppercase italic">
                Favorite Constellation
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1">
              <div className="h-2 w-2 rounded-full bg-[#ef4444] shadow-[0_0_10px_rgba(239,68,68,0.5)]" />
              <span className="text-[8px] tracking-[0.2em] text-white/60 uppercase">
                ⚡ Recently Seen
              </span>
            </div>
            <div className="flex items-center gap-3 ml-4 border-l border-white/10 pl-3">
              <div className="h-[1px] w-4 bg-[#ef4444] opacity-50" />
              <span className="text-[7px] tracking-[0.2em] text-white/40 uppercase italic">
                Recent Constellation
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1">
              <div className="h-2 w-2 rounded-full bg-[var(--success)]" />
              <span className="text-[8px] tracking-[0.2em] text-white/60 uppercase font-bold">
                ● Seen
              </span>
            </div>
          </>
        )}
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-[#2563eb] shadow-[0_0_10px_rgba(37,99,235,0.5)]" />
          <span className="text-[8px] tracking-[0.2em] text-white/60 uppercase font-bold">
            ⌬ Similar Signal (Neighbor)
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-[#ffaa00] shadow-[0_0_10px_rgba(255,170,0,0.5)]" />
          <span className="text-[8px] tracking-[0.2em] text-white/60 uppercase font-bold">
            Active Signal (Target)
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-[#ffffff] shadow-[0_0_10px_rgba(255,255,255,0.3)]" />
          <span className="text-[8px] tracking-[0.2em] text-white/60 uppercase font-bold">
            Undiscovered Movie
          </span>
        </div>
      </div>
    </>
  );
}
