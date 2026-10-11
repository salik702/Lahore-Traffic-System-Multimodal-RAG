"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Bookmark, CircleDollarSign, Info, SlidersHorizontal, X } from "lucide-react";
import { api } from "@/lib/api";
import type { Fine, Sign } from "@/lib/models";
import { usePersistedState } from "@/lib/use-persisted-state";
import { Chip, EmptyState, ErrorState, LoadingState, PageIntro, SearchField } from "@/components/ui";
import { MediaImage } from "@/components/media-image";
import { InfiniteTypewriter } from "./infinite-typewriter";

const vehicleColumns: [keyof Fine, string][] = [["motorcycle","Motorcycle"],["three_wheeler","3-wheeler"],["motorcar_under_2000cc","Car under 2000cc"],["motorcar_over_2000cc","Car over 2000cc"],["transport_vehicle","Transport"]];

export function FinesView() {
  const [items,setItems]=useState<Fine[]>([]); const [query,setQuery]=useState(""); const [vehicle,setVehicle]=useState<keyof Fine>("motorcycle"); const [loading,setLoading]=useState(true); const [error,setError]=useState("");
  const load=useCallback(async()=>{try{const r=await api<{items:Fine[]}>("/fines");setItems(r.items);setError("");}catch(e){setError(e instanceof Error?e.message:"Unable to load fines");}finally{setLoading(false);}},[]); useEffect(()=>{const timer=window.setTimeout(()=>void load(),0);return()=>window.clearTimeout(timer);},[load]);
  const filtered=useMemo(()=>{const q=query.trim().toLowerCase();return q?items.filter(item=>`${item.violation} ${item.keywords.join(" ")}`.toLowerCase().includes(q)):items;},[items,query]);
  return <div><PageIntro
    kicker="Traffic fines"
    title={<InfiniteTypewriter texts={["Know the penalty before the journey.", "Verified fine schedules for Punjab.", "Compare fines by vehicle class."]} />}
    description="Search Punjab's official fine schedule and compare the amount for each vehicle class. A dash means the class is not listed for that violation."
    action={<div className="flex items-center gap-2 rounded-full bg-[var(--brand-soft)] px-3 py-2 text-xs font-bold text-[var(--brand)]"><CircleDollarSign size={15}/>Amounts in PKR</div>}
  /><div className="mb-5 flex flex-col gap-3 lg:flex-row"><SearchField value={query} onChange={setQuery} placeholder="Search helmet, speeding, signal, phone…"/><label className="surface flex items-center gap-2 rounded-lg px-3"><SlidersHorizontal size={16} className="text-[var(--faint)]"/><span className="sr-only">Vehicle type</span><select value={String(vehicle)} onChange={e=>setVehicle(e.target.value as keyof Fine)} className="min-h-11 bg-transparent pr-3 text-sm font-bold outline-none">{vehicleColumns.map(([key,label])=><option key={String(key)} value={String(key)}>{label}</option>)}</select></label></div>{loading?<LoadingState/>:error?<ErrorState message={error} retry={load}/>:<>
    <div className="grid gap-3 md:hidden">{filtered.map(item=><article key={item.code} className="surface rounded-xl p-5"><div className="flex items-start justify-between gap-4"><div><div className="label">Code {String(item.code).padStart(2,"0")}</div><h2 className="mt-2 text-sm font-extrabold leading-5">{item.violation}</h2></div><div className="text-right"><div className="label">{vehicleColumns.find(([key])=>key===vehicle)?.[1]}</div><div className="mt-1 font-mono text-lg font-semibold text-[var(--brand)]">{typeof item[vehicle]==="number"?Number(item[vehicle]).toLocaleString("en-PK"):"—"}</div></div></div></article>)}{!filtered.length&&<EmptyState/>}</div>
    <div className="surface hidden overflow-x-auto rounded-xl md:block"><table className="w-full min-w-[920px] border-collapse text-left"><thead className="bg-[var(--surface-2)]"><tr><th className="p-4 text-xs">Code / violation</th>{vehicleColumns.map(([,label])=><th key={label} className="p-4 text-right text-xs">{label}<span className="label mt-1 block">PKR</span></th>)}</tr></thead><tbody>{filtered.map(item=><tr key={item.code} className="border-t border-[var(--line)] transition hover:bg-[var(--surface-2)]"><td className="max-w-md p-4"><span className="mr-3 font-mono text-xs text-[var(--faint)]">{String(item.code).padStart(2,"0")}</span><span className="text-sm font-bold">{item.violation}</span></td>{vehicleColumns.map(([key])=><td key={String(key)} className={`p-4 text-right font-mono text-sm font-semibold ${key===vehicle?"bg-[var(--brand-soft)] text-[var(--brand)]":"text-[var(--muted)]"}`}>{typeof item[key]==="number"?Number(item[key]).toLocaleString("en-PK"):"—"}</td>)}</tr>)}</tbody></table>{!filtered.length&&<div className="p-4"><EmptyState/></div>}</div>
    <p className="mt-3 flex items-center gap-2 text-xs text-[var(--muted)]"><Info size={14}/>Source: Traffic Police Punjab · official schedule in the indexed corpus</p></>}</div>;
}

export function SignsView() {
  const [items,setItems]=useState<Sign[]>([]);
  const [query,setQuery]=useState("");
  const [category,setCategory]=useState("All");
  const [selected,setSelected]=useState<Sign|null>(null);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState("");
  const [favorites,setFavorites]=usePersistedState<string[]>("lti-favorite-signs",[]);
  const load=useCallback(async()=>{try{const r=await api<{items:Sign[]}>("/signs");setItems(r.items);setError("");}catch(e){setError(e instanceof Error?e.message:"Unable to load signs");}finally{setLoading(false);}},[]);
  useEffect(()=>{const timer=window.setTimeout(()=>void load(),0);return()=>window.clearTimeout(timer);},[load]);
  const categories=useMemo(()=>["All",...Array.from(new Set(items.map(i=>i.category)))],[items]);
  const filtered=useMemo(()=>{const q=query.trim().toLowerCase();return items.filter(i=>(category==="All"||i.category===category)&&(!q||`${i.name} ${i.meaning}`.toLowerCase().includes(q)));},[items,query,category]);
  const toggle=(id:string)=>setFavorites(old=>old.includes(id)?old.filter(x=>x!==id):[...old,id]);

  return <div>
    <PageIntro
      kicker="Traffic signs"
      title={<InfiniteTypewriter texts={["Read the road at a glance.", "Explore official NHMP signs.", "Learn traffic signs effortlessly."]} />}
      description="Explore signs from official NHMP educational publications. Save the ones you want to revisit."
      action={favorites.length?<div className="flex items-center gap-2 rounded-full bg-[var(--accent-soft)] px-3 py-2 text-xs font-bold"><Bookmark size={14} fill="currentColor"/>{favorites.length} saved</div>:undefined}
    />
    <SearchField value={query} onChange={setQuery} placeholder="Search a sign by name or meaning…"/>
    <div className="scrollbar mt-3 flex gap-2 overflow-x-auto pb-2">{categories.map(value=><Chip key={value} active={category===value} onClick={()=>setCategory(value)}>{value}</Chip>)}</div>
    {loading?<div className="mt-5"><LoadingState/></div>:error?<div className="mt-5"><ErrorState message={error} retry={load}/></div>:<div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">{filtered.map(sign=>{const saved=favorites.includes(sign.id);return <article key={sign.id} className="surface group overflow-hidden rounded-xl"><div className="relative aspect-[4/3] bg-white dark:bg-[var(--surface-2)]"><button onClick={()=>setSelected(sign)} aria-label={`View details for ${sign.name}`} className="focus-ring grid h-full w-full place-items-center p-7"><MediaImage src={sign.image_url} alt={sign.name} width={150} height={150} className="h-28 w-28 rounded-lg bg-white object-scale-down transition duration-300 group-hover:scale-105"/></button><button onClick={()=>toggle(sign.id)} aria-label={saved?`Remove ${sign.name} from saved signs`:`Save ${sign.name}`} className={`focus-ring absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-full border shadow-sm ${saved?"border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]":"border-[var(--line)] bg-[var(--surface)] text-[var(--muted)]"}`}><Bookmark size={16} fill={saved?"currentColor":"none"}/></button></div><div className="border-t border-[var(--line)] p-4"><div className="label text-[var(--brand)]">{sign.category}</div><h2 className="mt-2 font-extrabold">{sign.name}</h2><p className="mt-2 text-xs leading-5 text-[var(--muted)]">{sign.meaning}</p><div className="label mt-4 text-[9px]">{sign.image_source_id === "project-illustration" ? "Project illustration" : "NHMP"}{sign.image_source_page ? ` · Page ${sign.image_source_page}` : ""}</div></div></article>;})}{!filtered.length&&<EmptyState/>}</div>}
    {selected&&<div className="fixed inset-0 z-50 grid place-items-center bg-black/55 p-4" role="presentation" onMouseDown={event=>{if(event.target===event.currentTarget)setSelected(null);}}><section role="dialog" aria-modal="true" aria-labelledby="sign-detail-title" className="surface w-full max-w-lg overflow-hidden rounded-2xl shadow-2xl"><div className="flex items-center justify-between border-b border-[var(--line)] p-4"><div className="label text-[var(--brand)]">{selected.category}</div><button onClick={()=>setSelected(null)} aria-label="Close sign details" className="focus-ring grid h-9 w-9 place-items-center rounded-md border border-[var(--line)]"><X size={17}/></button></div><div className="grid aspect-[16/10] place-items-center bg-white p-8 dark:bg-[var(--surface-2)]"><MediaImage src={selected.image_url} alt={selected.name} width={240} height={240} className="h-48 w-48 rounded-lg bg-white object-scale-down"/></div><div className="p-5"><h2 id="sign-detail-title" className="text-xl font-extrabold">{selected.name}</h2><p className="mt-2 text-sm leading-6 text-[var(--muted)]">{selected.meaning}</p><div className="label mt-4">{selected.image_source_id === "project-illustration" ? "Project illustration" : "Source: NHMP"}{selected.image_source_page ? ` · Page ${selected.image_source_page}` : ""}</div>{selected.image_source_url && <a href={selected.image_source_url} target="_blank" rel="noreferrer" className="mt-3 block text-xs font-bold text-[var(--brand)]">View original source ↗</a>}</div></section></div>}
  </div>;
}
