"use client";

/* eslint-disable @next/next/no-img-element -- local in-memory evidence URLs */
import { useState } from "react";
import { CarFront, CheckCircle2, Download, ExternalLink, Eye, LoaderCircle, ScanLine, ShieldAlert } from "lucide-react";
import { API_URL } from "@/lib/api";
import { human, money, percent, type VisionResult, type VehicleResult } from "@/lib/vision";

function VehicleDetails({ vehicle, result }: { vehicle: VehicleResult; result: VisionResult }) {
  const plate = vehicle.plate;
  const observations = vehicle.assessments.filter(a => a.status === "uncertain" || a.status === "not_detected");
  return <div className="mt-5 space-y-4">
    <div className="grid gap-4 md:grid-cols-[1fr_180px]">
      <div className="rounded-xl border border-[var(--line)] p-4">
        <div className="label">Vehicle information</div>
        <h3 className="mt-2 flex items-center gap-2 text-lg font-extrabold capitalize"><CarFront size={20}/>{vehicle.vehicle_type} <span className="text-xs font-normal text-[var(--muted)]">{vehicle.vehicle_id}</span></h3>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
          <div><dt className="text-[var(--muted)]">Detection score</dt><dd className="mt-1 font-bold">{percent(vehicle.detection_confidence)}</dd></div>
          <div><dt className="text-[var(--muted)]">Classification</dt><dd className="mt-1 font-bold capitalize">{vehicle.classification_status}</dd></div>
          <div><dt className="text-[var(--muted)]">Supported potential findings</dt><dd className="mt-1 font-bold">{vehicle.summary.supported_findings}</dd></div>
          <div><dt className="text-[var(--muted)]">Verified potential total</dt><dd className="mt-1 font-bold">{money(vehicle.summary.verified_potential_fine)}</dd></div>
        </dl>
      </div>
      {vehicle.image_url && <img src={vehicle.image_url} alt={`${vehicle.vehicle_type} evidence crop for ${vehicle.vehicle_id}`} className="h-44 w-full rounded-xl border border-[var(--line)] bg-[var(--surface-2)] object-contain"/>}
    </div>
    <div className="rounded-xl border border-[var(--line)] p-4">
      <h3 className="flex items-center gap-2 text-sm font-bold"><ScanLine size={17}/>Number plate</h3>
      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1"><p className="font-mono text-lg font-bold tracking-wide">{plate?.text ?? (plate?.status === "visibility_not_established" ? "Visibility not established" : "Number plate not readable.")}</p>
          <p className="mt-2 text-xs leading-5 text-[var(--muted)]">{plate?.reason ?? "Motor-vehicle plate analysis does not apply to this vehicle."}</p>
          {plate && <p className="mt-2 text-[11px] text-[var(--muted)]">Localization: {percent(plate.confidence)} · OCR: {percent(plate.ocr_confidence)} · {human(plate.status)}</p>}
        </div>
        {plate?.crop_url && <img src={plate.crop_url} alt="Localized number plate crop; compare it with the OCR reading" className="max-h-24 w-44 rounded-lg border border-[var(--line)] object-contain"/>}
      </div>
    </div>
    <div className="rounded-xl border border-[var(--line)] p-4">
      <h3 className="flex items-center gap-2 text-sm font-extrabold"><ShieldAlert size={17}/>Potential violations · {vehicle.violations.length}</h3>
      {!vehicle.violations.length && <p className="mt-3 text-sm leading-6 text-[var(--muted)]">No supported potential violation was established. This does not establish full compliance.</p>}
      {vehicle.violations.map(v => <article key={v.id} className="mt-4 border-t border-[var(--line)] pt-4">
        <div className="flex flex-wrap justify-between gap-2"><h4 className="text-sm font-bold text-[var(--danger)]">{v.name}</h4><span className="rounded bg-[var(--danger-soft)] px-2 py-1 text-xs font-bold text-[var(--danger)]">{percent(v.confidence)} evidence score</span></div>
        <ul className="mt-3 list-inside list-disc space-y-1 text-xs leading-5 text-[var(--muted)]">{v.visual_evidence.map(e => <li key={e}>{e}</li>)}</ul>
        <div className="mt-3 flex flex-wrap gap-3">{result.evidence_crops.filter(c => v.evidence_ids.includes(c.id)).map(c => <figure key={c.id}><img src={c.image_url} alt={`${c.label} supporting ${v.name}`} className="h-24 max-w-36 rounded-lg border border-[var(--line)] object-contain"/><figcaption className="mt-1 text-[10px] text-[var(--muted)]">{c.label} · {percent(c.confidence)}</figcaption></figure>)}</div>
        <p className="mt-3 text-sm font-bold">{v.fine_verified && v.fine_amount != null ? `${money(v.fine_amount)} potential penalty` : "Official fine amount requires verification."}</p>
        <p className="mt-1 text-[11px] leading-5 text-[var(--muted)]">{v.fine_status} · {result.jurisdiction}</p>
        <p className="mt-1 text-[11px] leading-5 text-[var(--muted)]">{v.legal_reference} {v.last_verified_on && `· Verified ${v.last_verified_on}`}</p>
        <div className="mt-2 flex flex-col items-start gap-2">{v.sources.map(s => <a key={s.url} href={s.url} target="_blank" rel="noopener noreferrer" className="focus-ring inline-flex items-center gap-1 rounded text-xs font-bold text-[var(--brand)]">{s.title}<ExternalLink size={12}/></a>)}</div>
      </article>)}
    </div>
    {vehicle.riders.length > 0 && <details className="rounded-xl border border-[var(--line)] p-4"><summary className="cursor-pointer text-sm font-bold">Rider relationships · {vehicle.riders.length}</summary><div className="mt-3 space-y-3">{vehicle.riders.map(r => <div key={r.id} className="text-xs leading-5"><b>{r.id} · {human(r.helmet_status)}</b><p className="text-[var(--muted)]">Association: {r.association} · {percent(r.association_confidence)}. {r.association_evidence}</p><p className="text-[var(--muted)]">{r.helmet_reason}</p></div>)}</div></details>}
    {observations.length > 0 && <div className="rounded-xl bg-[var(--accent-soft)] p-4"><h3 className="text-sm font-bold">Other observations</h3>{observations.map((a,i) => <p key={`${a.type}-${i}`} className="mt-2 text-xs leading-5"><b className="capitalize">{human(a.type)} · {human(a.status)}</b><br/>{a.reason}</p>)}</div>}
    <details className="rounded-xl border border-[var(--line)] p-4"><summary className="cursor-pointer text-sm font-bold">Cannot determine from this image · {vehicle.assessments.filter(a => a.status === "not_assessable").length} assessments</summary><div className="mt-3 space-y-3">{vehicle.assessments.filter(a => a.status === "not_assessable").map((a,i) => <p key={`${a.type}-${i}`} className="text-xs leading-5 text-[var(--muted)]"><b className="capitalize text-[var(--ink)]">{human(a.type)}</b><br/>{a.reason}</p>)}</div></details>
  </div>;
}

export function VisionResults({ result }: { result: VisionResult }) {
  const [selectedId, setSelectedId] = useState(result.primary_vehicle_id ?? result.vehicle_results[0]?.vehicle_id ?? "");
  const [annotated, setAnnotated] = useState(true);
  const [focused, setFocused] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState("");
  const selected = result.vehicle_results.find(v => v.vehicle_id === selectedId) ?? result.vehicle_results[0];
  const summary = selected?.summary ?? result.summary;
  const bounds = focused ? selected?.focus_bbox : undefined;
  const aspect = bounds ? (bounds[2]-bounds[0])/(bounds[3]-bounds[1]) : result.image_size.width/result.image_size.height;
  const imageSource = focused && selected ? (annotated ? selected.focus_annotated_image_url : selected.image_url) : (annotated ? result.annotated_image_url : result.original_image_url);
  async function download() {
    setDownloading(true); setDownloadError("");
    try {
      const query = selected ? `?vehicle_id=${encodeURIComponent(selected.vehicle_id)}` : "";
      const response = await fetch(`${API_URL}${result.report_url}${query}`, { cache: "no-store" });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail ?? "Report download failed.");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = url; link.download = `LTI-violation-report-${result.report_id.slice(0,12)}.pdf`;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setDownloadError(e instanceof Error ? e.message : "Report download failed."); }
    finally { setDownloading(false); }
  }
  return <section className="surface min-w-0 rounded-xl p-5 md:p-6">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="label flex items-center gap-2 text-[var(--brand)]"><CheckCircle2 size={14}/>Analysis complete · human review required</div><h2 className="mt-2 text-xl font-extrabold">Inspect the evidence</h2><p className="mt-1 text-xs text-[var(--muted)]">{result.jurisdiction} · {result.engine} · {(result.timings.analysis_ms / 1000).toFixed(2)}s</p></div>
      <button type="button" onClick={() => void download()} disabled={downloading} className="focus-ring flex items-center gap-2 rounded-lg bg-[var(--brand-fill)] px-4 py-3 text-xs font-bold text-[var(--on-brand)] disabled:opacity-60">{downloading ? <LoaderCircle size={16} className="animate-spin"/> : <Download size={16}/>}Download Violation Report (PDF)</button></div>
    {downloadError && <p role="alert" className="mt-3 text-xs text-[var(--danger)]">{downloadError}</p>}
    {selected && <p className="mt-4 rounded-lg bg-[var(--brand-soft)] p-3 text-xs leading-5" role="status"><b className="capitalize">Report subject: {selected.vehicle_type} · {selected.vehicle_id}</b><br/>The PDF includes this vehicle and its associated riders only. {selected.vehicle_id === result.primary_vehicle_id ? result.foreground_selection.reason : "You selected this vehicle for the report."}</p>}
    <div className="mt-5 grid grid-cols-2 gap-2 md:grid-cols-4">{[
      [summary.supported_findings, "Supported potential findings"],
      [summary.uncertain_findings, "Uncertain observations"],
      [money(summary.verified_potential_fine), "Verified potential fines"],
      [summary.unverified_penalties, "Unverified penalties"],
    ].map(([value,label]) => <div key={label} className="rounded-lg bg-[var(--surface-2)] p-3"><div className="text-lg font-extrabold">{value}</div><p className="mt-1 text-[10px] leading-4 text-[var(--muted)]">{label}</p></div>)}</div>
    <p className="mt-2 text-[11px] text-[var(--muted)]">Estimated potential penalties for the selected vehicle only. Surrounding vehicles and uncertain observations are excluded.</p>
    {!result.image_quality.sufficient && <p role="status" className="mt-4 rounded-lg bg-[var(--accent-soft)] p-3 text-xs leading-5">Insufficient image detail. Violation findings are withheld. {result.image_quality.reasons.join(". ")}</p>}
    <div className="mt-5 flex flex-wrap gap-2">{[true,false].map(value => <button key={String(value)} type="button" aria-pressed={annotated === value} onClick={() => setAnnotated(value)} className={`focus-ring flex items-center gap-2 rounded-md border border-[var(--line)] px-3 py-2 text-xs font-bold ${annotated === value ? "bg-[var(--brand-soft)] text-[var(--brand)]" : "text-[var(--muted)]"}`}><Eye size={14}/>{value ? "Annotated" : "Original"}</button>)}{[true,false].map(value => <button key={`scope-${value}`} type="button" aria-pressed={focused === value} onClick={() => setFocused(value)} className={`focus-ring rounded-md border border-[var(--line)] px-3 py-2 text-xs font-bold ${focused === value ? "bg-[var(--brand-soft)] text-[var(--brand)]" : "text-[var(--muted)]"}`}>{value ? "Report subject" : "Scene"}</button>)}</div>
    <div className="relative mx-auto mt-3 w-full" style={{ maxWidth: `${Math.min(1200,540*aspect)}px` }}>
      <img src={imageSource ?? result.original_image_url} alt={focused && selected ? (annotated ? "Annotated evidence for the selected report vehicle" : "Original selected vehicle and rider crop") : annotated ? "Vehicle and associated object annotations from actual model inference" : "Original image at analysis resolution"} className="h-auto w-full rounded-lg border border-[var(--line)]"/>
      {!focused && <svg viewBox={`0 0 ${result.image_size.width} ${result.image_size.height}`} className="absolute inset-0 h-full w-full" aria-label="Select a detected vehicle">
        {result.vehicle_results.map(v => <rect key={v.vehicle_id} x={v.bbox[0]} y={v.bbox[1]} width={v.bbox[2]-v.bbox[0]} height={v.bbox[3]-v.bbox[1]} fill="transparent" stroke={selected?.vehicle_id === v.vehicle_id ? "var(--accent-mark)" : "transparent"} strokeWidth={4} vectorEffect="non-scaling-stroke" role="button" tabIndex={0} aria-label={`Inspect ${v.vehicle_type} ${v.vehicle_id}`} className="cursor-pointer focus:outline-none focus:stroke-[var(--accent-mark)]" onClick={() => setSelectedId(v.vehicle_id)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelectedId(v.vehicle_id); } }}><title>{v.vehicle_type} · {percent(v.detection_confidence)}</title></rect>)}
      </svg>}
    </div>
    <p className="mt-2 text-[10px] leading-4 text-[var(--muted)]">The foreground estimate is selected initially. Use the selectors below or open Scene to choose another vehicle. The PDF always follows your selection. Blue: vehicles · Amber: people · Green: helmet · Red: bare-head observation · Purple: plate.</p>
    <div className="scrollbar mt-4 flex gap-2 overflow-x-auto pb-2" aria-label="Detected vehicles">{result.vehicle_results.map((v,i) => <button key={v.vehicle_id} type="button" aria-pressed={selected?.vehicle_id === v.vehicle_id} onClick={() => setSelectedId(v.vehicle_id)} className={`focus-ring shrink-0 rounded-lg border px-3 py-2 text-left text-xs ${selected?.vehicle_id === v.vehicle_id ? "border-[var(--brand)] bg-[var(--brand-soft)]" : "border-[var(--line)]"}`}><span className="block font-bold capitalize">{i+1} · {v.vehicle_type}</span><span className="mt-1 block text-[var(--muted)]">{v.violations.length} supported findings</span></button>)}</div>
    {selected ? <VehicleDetails vehicle={selected} result={result}/> : <p className="mt-4 rounded-lg bg-[var(--surface-2)] p-4 text-sm text-[var(--muted)]">No supported vehicles detected. No penalties are inferred.</p>}
    <p className="mt-5 text-xs leading-6 text-[var(--muted)]">{result.ai_explanation}</p>
    <details className="mt-5 rounded-xl border border-[var(--line)] p-4"><summary className="cursor-pointer text-sm font-bold">Capabilities, quality and limitations</summary><p className="mt-3 text-xs leading-5 text-[var(--muted)]">{result.confidence_calibration}</p><div className="mt-3 space-y-3">{result.capabilities.map(c => <p key={c.id} className="text-xs leading-5"><b>{c.name}</b> <span className="text-[var(--muted)]">· {c.status === "unsupported" ? "Unsupported" : "Implemented; Pakistan evaluation needed"}</span><br/><span className="text-[var(--muted)]">{c.reason}</span></p>)}</div><ul className="mt-4 list-inside list-disc space-y-2 text-xs leading-5 text-[var(--muted)]">{[...result.warnings,...result.limitations].map(t => <li key={t}>{t}</li>)}</ul><p className="mt-3 text-[10px] text-[var(--muted)]">Pipeline {result.pipeline_version} · Plate engine: {result.plate_engine} · Image {result.image_size.width} × {result.image_size.height}</p></details>
    <p className="mt-5 rounded-lg bg-[var(--brand-soft)] p-3 text-xs leading-5 text-[var(--brand)]">{result.disclaimer}</p>
  </section>;
}
