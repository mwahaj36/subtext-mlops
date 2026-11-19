import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2 } from "lucide-react";

export default function GalaxySearchOverlay({
  searchQuery,
  handleSearch,
  searchResults,
  warpToMovie,
  selectedNode,
  hoverNode,
  neighborIds,
  hoveredDetails,
  selectedDetails,
  THEME,
}) {
  return (
    <div className="absolute top-8 left-8 z-50 flex flex-col gap-6 items-start text-white">
      <input
        type="text"
        placeholder="WARP TO MOVIE..."
        value={searchQuery}
        onChange={(e) => handleSearch(e.target.value)}
        className="w-48 bg-transparent border-b border-[var(--primary)] py-2 text-[10px] tracking-[0.4em] text-[var(--primary)] outline-none placeholder:text-[var(--primary)]/40 uppercase transition-all focus:w-80 font-bold"
      />

      <AnimatePresence>
        {searchResults.length > 0 && (
          <motion.div className="w-80 bg-black/95 backdrop-blur-3xl border border-white/20 shadow-2xl overflow-hidden font-bold">
            {searchResults.map((m) => (
              <button
                key={m.id}
                onClick={() => warpToMovie(m)}
                className="w-full p-4 text-left text-[9px] tracking-widest text-white/60 hover:text-white hover:bg-[var(--primary)]/20 uppercase transition-all flex justify-between items-center border-b border-white/5 last:border-0"
              >
                <span>{m.name}</span>
                {m.type === "favorite" && (
                  <span className="text-[var(--primary)]">◈</span>
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {(selectedNode || (hoverNode && neighborIds.has(hoverNode.id))) && (
          <motion.div
            key={(hoverNode || selectedNode).id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex gap-6 border border-white/10 bg-black/40 p-6 backdrop-blur-3xl shadow-2xl rounded-sm"
          >
            <div className="relative h-32 w-20 border border-white/10 bg-white/5 overflow-hidden flex items-center justify-center shrink-0">
              {hoveredDetails || selectedDetails ? (
                <img
                  src={`https://image.tmdb.org/t/p/w200${(hoveredDetails || selectedDetails).poster_path}`}
                  className="h-full w-full object-cover shadow-2xl"
                />
              ) : (
                <Loader2 className="h-6 w-6 animate-spin text-white/20" />
              )}
            </div>
            <div className="flex flex-col justify-center text-left">
              <p
                className={`text-[8px] tracking-[0.4em] uppercase font-black mb-2 ${
                  (hoverNode || selectedNode).type === "favorite"
                    ? "text-[var(--primary)]"
                    : (hoverNode || selectedNode).type === "recent"
                      ? "text-[var(--secondary)]"
                      : neighborIds.has((hoverNode || selectedNode).id)
                        ? "text-[var(--accent)]"
                        : (hoverNode || selectedNode).type === "watched"
                          ? "text-[var(--success)]"
                          : "text-white/60"
                }`}
              >
                {selectedNode &&
                  (hoverNode || selectedNode).id === selectedNode.id && (
                    <span className="text-[var(--accent)] mr-2">◈ ACTIVE</span>
                  )}
                {(hoverNode || selectedNode).type === "favorite"
                  ? "Favourite"
                  : (hoverNode || selectedNode).type === "recent"
                    ? "Recently Seen"
                    : neighborIds.has((hoverNode || selectedNode).id)
                      ? "Similar Signal"
                      : (hoverNode || selectedNode).type === "watched"
                        ? "Seen"
                        : "Undiscovered"}
              </p>
              <h2 className="text-xl font-light tracking-tight text-white uppercase italic leading-tight max-w-[200px]">
                {(hoverNode || selectedNode).name}
              </h2>
              <p className="text-[10px] tracking-[0.2em] text-white/40 uppercase font-bold mt-1">
                {(hoveredDetails || selectedDetails)?.release_date?.split(
                  "-",
                )[0] || "Year Unknown"}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
