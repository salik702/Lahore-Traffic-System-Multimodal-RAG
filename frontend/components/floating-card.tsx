"use client";

import React from "react";
import { motion } from "motion/react";

interface FloatingCardProps {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  duration?: number;
}

export function FloatingCard({
  children,
  className = "",
  delay = 0,
  duration = 5
}: FloatingCardProps) {
  return (
    <motion.div
      animate={{
        y: [0, -8, 0],
      }}
      transition={{
        duration: duration,
        repeat: Infinity,
        ease: "easeInOut",
        delay: delay,
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
