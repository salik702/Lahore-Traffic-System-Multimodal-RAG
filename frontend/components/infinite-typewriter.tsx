"use client";

import React, { useState, useEffect } from "react";
import { motion } from "motion/react";

interface InfiniteTypewriterProps {
  texts: string[];
  className?: string;
  typingSpeed?: number;
  deletingSpeed?: number;
  pauseTime?: number;
}

export function InfiniteTypewriter({
  texts,
  className = "",
  typingSpeed = 80,
  deletingSpeed = 50,
  pauseTime = 2000,
}: InfiniteTypewriterProps) {
  const [index, setIndex] = useState(0);
  const [subIndex, setSubIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const currentText = texts[index];

    if (!isDeleting && subIndex === currentText.length) {
      // Pause at the end of the string before starting to delete
      const timeout = setTimeout(() => setIsDeleting(true), pauseTime);
      return () => clearTimeout(timeout);
    }

    if (isDeleting && subIndex === 0) {
      // Switch to next text after deleting completely
      setIsDeleting(false);
      setIndex((prev) => (prev + 1) % texts.length);
      return;
    }

    const timeout = setTimeout(() => {
      setSubIndex((prev) => prev + (isDeleting ? -1 : 1));
    }, isDeleting ? deletingSpeed : typingSpeed);

    return () => clearTimeout(timeout);
  }, [subIndex, isDeleting, index, texts, typingSpeed, deletingSpeed, pauseTime]);

  return (
    <span className={className}>
      {texts[index].substring(0, subIndex)}
      <motion.span
        animate={{ opacity: [1, 0] }}
        transition={{ duration: 0.6, repeat: Infinity, ease: "linear" }}
        className="inline-block w-[2px] h-[1em] bg-current ml-1 align-middle"
      />
    </span>
  );
}
