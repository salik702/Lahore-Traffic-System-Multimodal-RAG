"use client";

import Image from "next/image";
import { ImageOff, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { resolveMediaUrl } from "@/lib/api";

type MediaImageProps = {
  src?: string | null;
  alt: string;
  className?: string;
  fill?: boolean;
  width?: number;
  height?: number;
  sizes?: string;
  preview?: boolean;
};

export function MediaImage({ src, alt, className, fill = false, width = 160, height = 160, sizes, preview = false }: MediaImageProps) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const [opened, setOpened] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ src: string; width: number; height: number } | null>(null);
  const previewButton = useRef<HTMLButtonElement>(null);
  const resolved = resolveMediaUrl(src);
  useEffect(() => {
    if (!opened) return;
    const trigger = previewButton.current;
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpened(false);
      // The preview has one interactive control; keep keyboard focus in it.
      if (event.key === "Tab") {
        event.preventDefault();
        document.querySelector<HTMLButtonElement>('[aria-label="Close image preview"]')?.focus();
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKey);
      (trigger ?? previouslyFocused)?.focus();
    };
  }, [opened]);
  if (!resolved || failedSrc === resolved) {
    return <div role="img" aria-label={`${alt} image unavailable`} className="grid h-full w-full place-items-center text-center text-[var(--faint)]"><div><ImageOff className="mx-auto" size={24}/><span className="mt-2 block text-[11px] font-semibold">Image unavailable</span></div></div>;
  }
  const dimensions = naturalSize?.src === resolved ? naturalSize : null;
  return <><Image src={resolved} alt={alt} fill={fill} width={fill ? undefined : width} height={fill ? undefined : height} sizes={sizes} unoptimized onLoad={event => setNaturalSize({ src: resolved, width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} onError={() => setFailedSrc(resolved)} className={className}/>
    {preview && <button ref={previewButton} type="button" aria-label={`Preview ${alt}`} onClick={event => { event.preventDefault(); setOpened(true); }} className="focus-ring absolute inset-0 z-10 cursor-zoom-in"/>}
    {opened && createPortal(<div role="dialog" aria-modal="true" aria-label={alt} className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-5" onClick={() => setOpened(false)} onKeyDown={event => { if (event.key === "Escape") setOpened(false); }}><button type="button" autoFocus aria-label="Close image preview" onClick={() => setOpened(false)} className="absolute right-5 top-5 rounded-lg bg-white/10 p-3 text-white"><X/></button><Image src={resolved} alt={alt} width={dimensions?.width ?? 1200} height={dimensions?.height ?? 900} unoptimized className="max-h-[85vh] w-auto max-w-full object-contain" style={dimensions ? { maxWidth: `min(100%, ${dimensions.width}px)` } : undefined}/><p className="absolute bottom-3 max-w-[90%] text-center text-sm text-white">{alt}</p></div>, document.body)}
  </>;
}
