"use client";

import { AlertTriangle, LoaderCircle, Search } from "lucide-react";

export function PageIntro({ kicker, title, description, action }: { kicker: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="mb-7 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
    <div className="max-w-3xl"><div className="label text-[var(--brand)]">{kicker}</div><h1 className="mt-2 text-3xl font-extrabold tracking-[-.045em] md:text-[2.55rem] md:leading-[1.05]">{title}</h1><p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted)] md:text-[15px]">{description}</p></div>
    {action && <div className="shrink-0">{action}</div>}
  </div>;
}

export function SearchField({ value, onChange, placeholder, label = "Search" }: { value: string; onChange: (value: string) => void; placeholder: string; label?: string }) {
  return <label className="focus-within:border-[var(--brand)] flex min-w-0 flex-1 items-center rounded-lg border border-[var(--line)] bg-[var(--surface)] px-3 transition">
    <span className="sr-only">{label}</span><Search size={17} className="shrink-0 text-[var(--faint)]"/><input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} className="w-full bg-transparent px-3 py-3 text-sm outline-none placeholder:text-[var(--faint)]"/>
  </label>;
}

export function LoadingState({ label = "Loading verified records" }: { label?: string }) {
  return <div className="surface grid min-h-52 place-items-center rounded-xl p-8 text-center"><div><LoaderCircle className="mx-auto animate-spin text-[var(--brand)]"/><p className="mt-3 text-sm font-semibold">{label}</p></div></div>;
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return <div role="alert" className="rounded-xl border border-[var(--danger)] bg-[var(--danger-soft)] p-5"><AlertTriangle size={19} className="text-[var(--danger)]"/><p className="mt-3 text-sm font-bold">We couldn&apos;t load this view.</p><p className="mt-1 text-sm text-[var(--muted)]">{message}</p>{retry && <button onClick={retry} className="focus-ring mt-4 rounded-md bg-[var(--ink)] px-4 py-2 text-xs font-bold text-[var(--bg)]">Try again</button>}</div>;
}

export function EmptyState({ title = "No matching records", copy = "Try a broader search or clear the selected filter." }: { title?: string; copy?: string }) {
  return <div className="surface col-span-full rounded-xl p-10 text-center"><div className="text-sm font-bold">{title}</div><p className="mt-2 text-sm text-[var(--muted)]">{copy}</p></div>;
}

export function Chip({ children, active, onClick }: { children: React.ReactNode; active?: boolean; onClick?: () => void }) {
  return <button type="button" onClick={onClick} className={`focus-ring whitespace-nowrap rounded-full border px-3 py-2 text-xs font-bold transition ${active ? "border-[var(--brand)] bg-[var(--brand-fill)] text-[var(--on-brand)]" : "border-[var(--line)] bg-[var(--surface)] text-[var(--muted)] hover:border-[var(--line-strong)] hover:text-[var(--ink)]"}`}>{children}</button>;
}
