"use client";

import { useEffect, useRef, useState } from "react";

export function usePersistedState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(initial);
  const hydrated = useRef(false);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      try { const stored = localStorage.getItem(key); if (stored) setValue(JSON.parse(stored) as T); } catch { /* keep the safe default */ }
      hydrated.current = true;
    }, 0);
    return () => window.clearTimeout(timer);
  }, [key]);
  useEffect(() => {
    if (!hydrated.current) return;
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be disabled */ }
  }, [key, value]);
  return [value, setValue] as const;
}
