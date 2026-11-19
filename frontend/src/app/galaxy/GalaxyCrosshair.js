import React from "react";
import { motion } from "framer-motion";

export default function GalaxyCrosshair({
  loading,
  hoverNode,
  neighborIds,
  isLocked,
  THEME,
}) {
  return (
    <div
      className={`pointer-events-none absolute inset-0 z-[100] flex items-center justify-center transition-all duration-700 ${loading ? "opacity-20 scale-150" : "opacity-100 scale-100"}`}
    >
      <div className="relative">
        <motion.div
          animate={{
            borderColor: hoverNode
              ? hoverNode.type === "favorite"
                ? THEME.primary
                : hoverNode.type === "recent"
                  ? THEME.secondary
                  : neighborIds.has(hoverNode.id)
                    ? THEME.accent
                    : hoverNode.type === "watched"
                      ? THEME.success
                      : THEME.active
              : isLocked
                ? "rgba(255, 255, 255, 0.4)"
                : "rgba(255, 255, 255, 0.1)",
            scale: hoverNode ? 1.2 : 1,
            borderWidth: isLocked ? "2px" : "1px",
          }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className="h-6 w-6 border rounded-full shadow-[0_0_20px_rgba(var(--primary-rgb),0.2)]"
        />
        <motion.div
          animate={{
            backgroundColor: hoverNode
              ? hoverNode.type === "favorite"
                ? THEME.primary
                : hoverNode.type === "recent"
                  ? THEME.secondary
                  : neighborIds.has(hoverNode.id)
                    ? THEME.accent
                    : hoverNode.type === "watched"
                      ? THEME.success
                      : THEME.active
              : "rgba(255, 255, 255, 0.2)",
          }}
          className="absolute top-1/2 left-1/2 h-[1px] w-3 -translate-x-1/2 -translate-y-1/2"
        />
        <motion.div
          animate={{
            backgroundColor: hoverNode
              ? hoverNode.type === "favorite"
                ? THEME.primary
                : hoverNode.type === "recent"
                  ? THEME.secondary
                  : neighborIds.has(hoverNode.id)
                    ? THEME.accent
                    : hoverNode.type === "watched"
                      ? THEME.success
                      : THEME.active
              : "rgba(255, 255, 255, 0.2)",
          }}
          className="absolute top-1/2 left-1/2 h-3 w-[1px] -translate-x-1/2 -translate-y-1/2"
        />
      </div>
    </div>
  );
}
