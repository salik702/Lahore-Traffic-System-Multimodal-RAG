"use client";

import React from "react";
import { motion } from "motion/react";

interface InfiniteMarqueeProps {
  items: React.ReactNode[];
  direction?: "left" | "right";
  speed?: number;
  className?: string;
}

export function InfiniteMarquee({
  items,
  direction = "left",
  speed = 30,
  className = "",
}: InfiniteMarqueeProps) {
  return (
    <div className={`group flex overflow-hidden p-2 ${className}`}>
      <motion.div
        className="flex gap-8 whitespace-nowrap"
        animate={{
          x: direction === "left" ? [0, -1000] : [-1000, 0],
        }}
        transition={{
          duration: speed,
          repeat: Infinity,
          ease: "linear",
        }}
      >
        {[...items, ...items].map((item, i) => (
          <div key={i} className="flex shrink-0 items-center">
            {item}
          </div>
        ))}
      </motion.div>
    </div>
  );
}
