"use client";

/* eslint-disable @next/next/no-img-element -- temporary local upload preview */
import { useEffect, useRef, useState } from "react";
import { CheckCircle2, ImageUp, LoaderCircle, RefreshCw, ScanLine, UploadCloud } from "lucide-react";
import { api } from "@/lib/api";
import type { VisionResult } from "@/lib/vision";
import { PageIntro } from "@/components/ui";
import { VisionResults } from "@/components/vision-results";

export function VisualSearchView() {
  const [file,setFile] = useState<File | null>(null);
  const [preview,setPreview] = useState("");
  const [result,setResult] = useState<VisionResult | null>(null);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState("");
  const [dragging,setDragging] = useState(false);
  const [jurisdiction,setJurisdiction] = useState("Punjab");
  const [maxMb,setMaxMb] = useState(8);
  const [elapsed,setElapsed] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const request = useRef<AbortController | null>(null);
  const generation = useRef(0);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); },[preview]);
  useEffect(() => {
    const controller = new AbortController();
    const generationRef = generation;
    const requestRef = request;
    void api<{ max_upload_mb: number }>("/vision/capabilities", { signal: controller.signal }).then(r => setMaxMb(r.max_upload_mb)).catch(() => {});
    return () => { controller.abort(); generationRef.current++; requestRef.current?.abort(); };
  },[]);
  useEffect(() => {
    if (!loading) return;
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now()-started)/1000)),1000);
    return () => window.clearInterval(timer);
  },[loading]);

  function reset() {
    generation.current++; request.current?.abort(); setFile(null); setPreview(""); setResult(null); setLoading(false); setError("");
    if (input.current) input.current.value = "";
  }
  function choose(next?: File) {
    if (!next) return;
    if (!["image/jpeg","image/png","image/webp"].includes(next.type) || next.size > maxMb*1024*1024 || !next.size) {
      setError(`Choose a non-empty JPEG, PNG, or WebP image up to ${maxMb} MB.`); return;
    }
    generation.current++; request.current?.abort();
    setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setError(""); setLoading(false); setElapsed(0);
  }
  async function analyze() {
    if (!file || loading) return;
    const id = ++generation.current;
    setResult(null); setError(""); setLoading(true); setElapsed(0);
    const body = new FormData(); body.append("file",file); body.append("jurisdiction",jurisdiction);
    const controller = new AbortController(); request.current = controller;
    const timer = window.setTimeout(() => controller.abort(),180000);
    try {
      const response = await api<VisionResult>("/vision/analyze",{ method:"POST",body,signal:controller.signal });
      if (generation.current === id) setResult(response);
    } catch (e) {
      if (generation.current === id) setError(e instanceof DOMException && e.name === "AbortError" ? "Analysis timed out. Retry shortly; the server may still be finishing the previous image." : e instanceof Error ? e.message : "Analysis failed.");
    } finally { window.clearTimeout(timer); if (generation.current === id) setLoading(false); }
  }
  return <div>
    <PageIntro kicker="Computer vision · Evidence first" title="AI Traffic Violation Detection" description="Inspect each vehicle independently. Local models assess motorcycle riders and visible registration plates; supported findings are matched to verified jurisdiction-specific records."/>
    <div className="grid items-start gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <section className="surface rounded-xl p-5">
        <div className="label">1 · Upload a road photograph</div>
        <input ref={input} type="file" aria-label="Upload road photograph" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={e => choose(e.target.files?.[0])}/>
        <button type="button" onClick={() => input.current?.click()} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }} className={`focus-ring mt-4 grid min-h-60 w-full place-items-center rounded-xl border-2 border-dashed p-4 text-center ${dragging ? "border-[var(--brand)] bg-[var(--brand-soft)]" : "border-[var(--line-strong)] bg-[var(--surface-2)]"}`}>
          {preview ? <div><img src={preview} alt="Selected road photograph preview" className="mx-auto max-h-72 max-w-full rounded-lg object-contain"/><p className="mt-3 break-all text-xs font-bold">{file?.name}</p><p className="mt-1 text-[10px] text-[var(--muted)]">{file && (file.size/1024/1024).toFixed(2)} MB · Click or drop to replace</p></div> : <div><UploadCloud className="mx-auto text-[var(--brand)]" size={32}/><p className="mt-4 text-sm font-bold">Drag & drop an image</p><p className="mt-2 text-xs text-[var(--muted)]">or browse your files</p></div>}
        </button>
        <button type="button" onClick={() => input.current?.click()} className="focus-ring mt-3 w-full rounded-lg border border-[var(--line)] px-3 py-2 text-xs font-bold">Browse image</button>
        <p className="mt-3 text-[11px] leading-5 text-[var(--muted)]">JPEG, PNG, WebP · Up to {maxMb} MB / 20 megapixels · At least 32 × 32 pixels</p>
        <label className="mt-5 block text-xs font-bold" htmlFor="vision-jurisdiction">Jurisdiction</label>
        <select id="vision-jurisdiction" value={jurisdiction} disabled={loading} onChange={e => { setJurisdiction(e.target.value); setResult(null); }} className="focus-ring mt-2 w-full rounded-lg border border-[var(--line)] bg-[var(--surface)] p-3 text-sm">{["Punjab","Sindh","Khyber Pakhtunkhwa","Balochistan","Islamabad","National highways","Unknown"].map(j => <option key={j}>{j}</option>)}</select>
        <p className="mt-2 text-[10px] leading-4 text-[var(--muted)]">Location is supplied by you. Verified fine records currently cover Punjab. Other jurisdictions receive no assumed penalty amounts.</p>
        <button type="button" disabled={!file || loading} onClick={() => void analyze()} className="focus-ring mt-5 flex w-full items-center justify-center gap-2 rounded-lg bg-[var(--brand-fill)] px-4 py-3 text-sm font-bold text-[var(--on-brand)] disabled:cursor-not-allowed disabled:opacity-50">{loading ? <LoaderCircle size={18} className="animate-spin"/> : <ImageUp size={18}/>} {loading ? "Analyzing image…" : "Analyze Image"}</button>
        {error && <p role="alert" className="mt-3 rounded-lg bg-[var(--danger-soft)] p-3 text-xs leading-5 text-[var(--danger)]">{error}</p>}
        {file && <button type="button" onClick={reset} className="focus-ring mt-4 flex items-center gap-2 rounded text-xs font-bold text-[var(--muted)]"><RefreshCw size={14}/>Clear image</button>}
        <div className="mt-5 space-y-3 border-t border-[var(--line)] pt-4" aria-label="Analysis progress">{[
          ["Image selected",Boolean(file)], ["Model inference & evidence checks",Boolean(result)], ["PDF report ready",Boolean(result)],
        ].map(([label,complete],i) => <div key={String(label)} className="flex items-center gap-2 text-xs text-[var(--muted)]">{complete ? <CheckCircle2 size={14} className="text-[var(--brand)]"/> : loading && i === 1 ? <LoaderCircle size={14} className="animate-spin"/> : <span className="h-3.5 w-3.5 rounded-full border border-[var(--line-strong)]"/>}{label}</div>)}</div>
        <p className="mt-5 text-[10px] leading-5 text-[var(--muted)]">Temporary upload files are deleted after analysis. Report evidence stays in a bounded private memory cache for up to 15 minutes and may be evicted earlier. The PDF follows your selected vehicle and includes its visible plate text.</p>
      </section>
      <div aria-live="polite" aria-busy={loading}>
        {result ? <VisionResults key={result.report_id} result={result}/> : <section className="surface grid min-h-[520px] place-items-center rounded-xl p-8 text-center"><div>{loading ? <><LoaderCircle size={36} className="mx-auto animate-spin text-[var(--brand)]"/><h2 className="mt-5 text-lg font-bold">Analyzing the image · {elapsed}s</h2><p className="mx-auto mt-3 max-w-md text-sm leading-6 text-[var(--muted)]">Local vehicle detection, person association, helmet evidence, plate OCR, legal matching and PDF preparation are running.</p><div className="loading-line mx-auto mt-5 h-1 w-40 rounded bg-[var(--brand-fill)]" role="progressbar" aria-label="Analysis in progress"/><p className="mt-4 text-xs text-[var(--muted)]">Progress is indeterminate. CPU inference can take a moment.</p></> : <><ScanLine size={36} className="mx-auto text-[var(--brand)]"/><h2 className="mt-5 text-lg font-bold">Evidence for every finding</h2><p className="mx-auto mt-3 max-w-md text-sm leading-6 text-[var(--muted)]">Choose an image, then analyze it. Inspect individual vehicles, review uncertainty, and download a report generated from the actual results.</p><p className="mx-auto mt-5 max-w-md text-xs leading-5 text-[var(--muted)]">Seatbelts, phone use and road behavior require additional evaluated detectors or context. The system will say when an image cannot establish a condition.</p></>}</div></section>}
      </div>
    </div>
  </div>;
}
