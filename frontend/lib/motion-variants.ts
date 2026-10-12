import { Variants } from "motion/react";

export const easing: Record<string, [number, number, number, number]> = {
  outExpo: [0.16, 1, 0.3, 1],
  inOutQuart: [0.65, 0, 0.35, 1],
  outBack: [0.175, 0.885, 0.32, 1.275],
};

export const springs = {
  stiff: { type: "spring", damping: 25, stiffness: 260 },
  smooth: { type: "spring", damping: 30, stiffness: 180 },
  bouncy: { type: "spring", damping: 15, stiffness: 200 },
};

export const variants: { [key: string]: Variants } = {
  fadeUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  },
  staggerContainer: {
    animate: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  },
  scaleIn: {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
  },
  textReveal: {
    initial: { clipPath: "inset(100% 0% 0% 0%)" },
    animate: {
      clipPath: "inset(0% 0% 0% 0%)",
      transition: { duration: 0.8, ease: easing.outExpo },
    },
  },
};
