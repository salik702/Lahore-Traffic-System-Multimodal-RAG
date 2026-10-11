"use client";

import React from "react";
import { motion } from "motion/react";

interface GlowingBorderProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
}

export function GlowingBorder({
  children,
  className = "",
  glowColor = "var(--brand)",
}: GlowingBorderProps) {
  return (
    <div className={`relative p-[1px] rounded-xl overflow-hidden ${className}`}>
      <div
        className="absolute inset-0 animate-border-beam"
        style={{
          background: `conic-gradient(from 0deg, transparent 0deg, ${glowColor} 180deg, transparent 360deg)`
        }}
      />
      <div className="relative h-full w-full rounded-[11px] bg-inherit z-10">
        {children}
      </div>
    </div>
  );
}
