"use client";
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();
  const [isLightMode, setIsLightMode] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    setIsLightMode(false);
    setIsMenuOpen(false);
  }, [pathname]);

  const navItems = [{ name: "Search", href: "/search" }];

  return (
    <motion.nav
      animate={{
        backgroundColor: isLightMode
          ? "rgba(255, 255, 255, 0.4)"
          : "rgba(0, 0, 0, 0.4)",
        borderColor: isLightMode
          ? "rgba(0, 0, 0, 0.1)"
          : "rgba(255, 255, 255, 0.05)",
      }}
      className="sticky top-0 z-[1500] w-full border-b backdrop-blur-2xl group transition-colors duration-500"
    >
      {!isLightMode && (
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[var(--primary)]/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-1000 pointer-events-none" />
      )}

      <div className="w-full px-6 md:px-8 h-16 md:h-20 flex items-center justify-between relative z-10">
        <Link href="/search" className="flex items-center gap-3 group/logo">
          <motion.div
            animate={{ backgroundColor: isLightMode ? "#000000" : "#ffffff" }}
            className="w-20 md:w-28 h-6 md:h-8 group-hover/logo:bg-[var(--primary)] transition-all duration-500 transform group-hover/logo:scale-105 [mask-image:url(/file.svg)] [mask-size:contain] [mask-repeat:no-repeat] [mask-position:center]"
          />
        </Link>

        <div className="hidden md:flex items-center gap-12">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`text-[10px] font-black uppercase tracking-[0.3em] transition-all duration-300 relative group/link ${
                  isActive
                    ? "text-[var(--primary)]"
                    : isLightMode
                      ? "text-black/60 hover:text-black"
                      : "text-gray-400 hover:text-white"
                }`}
              >
                {item.name}
                <span
                  className={`absolute -bottom-1 left-0 h-[1px] bg-[var(--primary)] transition-all duration-500 shadow-[0_0_8px_var(--primary)] ${
                    isActive
                      ? "w-full opacity-100"
                      : "w-0 opacity-0 group-hover/link:w-full group-hover/link:opacity-100"
                  }`}
                />
              </Link>
            );
          })}
        </div>

        <button
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="flex md:hidden p-2 text-white/60 hover:text-[var(--primary)] transition-colors"
          aria-label="Toggle navigation menu"
        >
          {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className={`md:hidden overflow-hidden border-t border-white/5 backdrop-blur-3xl ${
              isLightMode ? "bg-white/95" : "bg-black/95"
            }`}
          >
            <div className="flex flex-col p-6 gap-6">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className="text-xs font-black uppercase tracking-[0.4em] text-white/60 hover:text-[var(--primary)]"
                >
                  {item.name}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
