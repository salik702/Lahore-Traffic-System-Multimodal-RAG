"use client";

import { useState, useSyncExternalStore } from "react";
import { useTheme } from "next-themes";
import { AnimatePresence, motion } from "motion/react";
import { BookOpenCheck, Bot, CarFront, CircleDollarSign, CloudSun, Home, ImageUp, Menu, Moon, Route, Signpost, Sun, X } from "lucide-react";
import type { Section } from "@/lib/models";

const nav: { id: Section; label: string; icon: typeof Home }[] = [
  { id: "home", label: "Home", icon: Home }, { id: "assistant", label: "AI Assistant", icon: Bot },
  { id: "fines", label: "Traffic Fines", icon: CircleDollarSign }, { id: "signs", label: "Traffic Signs", icon: Signpost },
  { id: "safety", label: "Road Safety", icon: BookOpenCheck }, { id: "visual", label: "AI Violation Detection", icon: ImageUp },
  { id: "accidents", label: "Accident Insights", icon: CarFront },
];

function ThemeSwitch() {
  const { theme, setTheme } = useTheme();
  const mounted = useSyncExternalStore(() => () => undefined, () => true, () => false);
  const current = mounted ? (theme ?? "system") : "system";
  const next = current === "system" ? "light" : current === "light" ? "dark" : "system";
  const Icon = current === "system" ? CloudSun : current === "dark" ? Moon : Sun;
  return <button type="button" aria-label={`Theme: ${current}. Switch to ${next}.`} title={`Theme: ${current}`} onClick={() => setTheme(next)} className="focus-ring grid h-9 w-9 place-items-center rounded-md border border-[var(--line)] bg-[var(--surface)] text-[var(--muted)] transition hover:text-[var(--ink)]"><Icon size={17}/></button>;
}

export function AppShell({ section, onNavigate, children }: { section: Section; onNavigate: (section: Section) => void; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const navigate = (id: Section) => { onNavigate(id); setOpen(false); };
  return <div className="min-h-screen">
    <header className="sticky top-0 z-40 border-b border-[var(--line)] bg-[color-mix(in_srgb,var(--bg)_88%,transparent)] backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center gap-4 px-4 md:px-6">
        <button className="focus-ring flex shrink-0 items-center gap-2 rounded-md" onClick={() => navigate("home")} aria-label="Lahore Traffic Intelligence home">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-[var(--brand-fill)] text-[var(--on-brand)]"><Route size={19}/></span>
          <span className="hidden text-left sm:block"><span className="block text-sm font-extrabold leading-none tracking-[-.02em]">Lahore Traffic</span><span className="label mt-1 block text-[9px]">Intelligence</span></span>
        </button>
        <nav aria-label="Primary navigation" className="ml-3 hidden min-w-0 flex-1 items-center gap-1 xl:flex">
          {nav.map(({id,label}) => <button key={id} onClick={() => navigate(id)} className={`focus-ring whitespace-nowrap rounded-md px-3 py-2 text-xs font-bold transition ${section === id ? "bg-[var(--brand-soft)] text-[var(--brand)]" : "text-[var(--muted)] hover:bg-[var(--surface-2)] hover:text-[var(--ink)]"}`}>{label}</button>)}
        </nav>
        <div className="ml-auto flex items-center gap-2"><div className="hidden items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--surface)] px-3 py-1.5 text-[10px] font-bold text-[var(--muted)] md:flex"><span className="h-1.5 w-1.5 rounded-full bg-[var(--brand-fill)]"/>Lahore · Punjab</div><ThemeSwitch/><button onClick={() => setOpen(true)} aria-label="Open navigation" className="focus-ring grid h-9 w-9 place-items-center rounded-md border border-[var(--line)] bg-[var(--surface)] xl:hidden"><Menu size={19}/></button></div>
      </div>
    </header>
    <AnimatePresence>{open && <><motion.button aria-label="Close navigation" className="fixed inset-0 z-40 bg-black/45 xl:hidden" onClick={() => setOpen(false)} initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}/><motion.aside className="fixed inset-y-0 right-0 z-50 w-[min(88vw,360px)] bg-[var(--surface)] p-5 shadow-2xl xl:hidden" initial={{x:"100%"}} animate={{x:0}} exit={{x:"100%"}} transition={{type:"spring",damping:28,stiffness:260}}><div className="flex items-center justify-between"><div><div className="font-extrabold">Navigate</div><div className="label mt-1">Lahore Traffic Intelligence</div></div><button aria-label="Close navigation" onClick={() => setOpen(false)} className="focus-ring grid h-9 w-9 place-items-center rounded-md border border-[var(--line)]"><X size={18}/></button></div><nav className="mt-8 space-y-1">{nav.map(({id,label,icon:Icon}) => <button key={id} onClick={() => navigate(id)} className={`focus-ring flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm font-bold ${section === id ? "bg-[var(--brand-soft)] text-[var(--brand)]" : "text-[var(--muted)] hover:bg-[var(--surface-2)]"}`}><Icon size={18}/>{label}</button>)}</nav></motion.aside></>}</AnimatePresence>
    <main id="main-content" className="mx-auto max-w-[1440px] px-4 py-7 md:px-6 md:py-10">{children}</main>
    <footer className="mt-10 border-t border-[var(--line)]"><div className="mx-auto flex max-w-[1440px] flex-col gap-2 px-4 py-6 text-xs text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between md:px-6"><span>Source-grounded road intelligence for Lahore and Punjab.</span><span>Always follow current directions from traffic authorities.</span></div></footer>
  </div>;
}
