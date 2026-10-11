"use client";

import React from "react";
import { motion } from "motion/react";

interface AuroraBackgroundProps {
  className?: string;
}

export function AuroraBackground({ className = "" }: AuroraBackgroundProps) {
  return (
    <div className={`fixed inset-0 -z-10 overflow-hidden pointer-events-none ${className}`}>
      <div className="absolute inset-0 opacity-30">
        <div className="absolute -inset-[10%] animate-aurora-1 bg-[conic-gradient(from_0deg_at_50%_50%,var(--brand)_0deg,transparent_120deg,var(--brand)_240deg,transparent_360deg)] blur-[100px]" />
        <div className="absolute -inset-[10%] animate-aurora-2 bg-[conic-gradient(from_180deg_at_50%_50%,var(--accent-mark)_0deg,transparent_120deg,var(--brand)_240deg,transparent_360deg)] blur-[100px]" />
      </div>
      <div className="absolute inset-0 bg-[var(--bg)]/60 backdrop-blur-3xl" />
    </div>
  );
}
