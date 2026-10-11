"use client";

import { FormEvent, useState } from "react";
import { motion } from "motion/react";
import { ArrowRight, BookOpenCheck, Bot, CircleDollarSign, Clock3, ImageUp, RotateCcw, Signpost } from "lucide-react";
import { variants } from "@/lib/motion-variants";
import { InfiniteTypewriter } from "./infinite-typewriter";
import { FloatingCard } from "./floating-card";
import { GlowingBorder } from "./glowing-border";
import type { RecentSearch, Section, Stats } from "@/lib/models";

const actions: { id: Section; title: string; copy: string; icon: typeof Bot }[] = [
  { id: "assistant", title: "Ask Traffic AI", copy: "Get an evidence-backed answer", icon: Bot },
  { id: "fines", title: "Check a fine", copy: "Compare official vehicle rates", icon: CircleDollarSign },
  { id: "signs", title: "Learn road signs", copy: "Browse the official sign library", icon: Signpost },
  { id: "visual", title: "Detect violations", copy: "Upload a road image for AI analysis", icon: ImageUp },
];

export function HomeView({ stats, recent, onNavigate, onAsk, onClearRecent }: { stats: Stats | null; recent: RecentSearch[]; onNavigate: (s: Section) => void; onAsk: (q: string) => void; onClearRecent: () => void }) {
  const [query, setQuery] = useState("");
  const submit = (event: FormEvent) => { event.preventDefault(); if (query.trim()) onAsk(query.trim()); };
  return (
    <motion.div initial="initial" animate="animate" variants={variants.staggerContainer}>
      <section className="map-grid relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)] px-5 py-8 md:px-9 md:py-11">
        <div className="pointer-events-none absolute -right-24 -top-24 h-80 w-80 rounded-full bg-[color-mix(in_srgb,var(--brand)_12%,transparent)] blur-3xl"/>
        <div className="relative max-w-3xl">
          <motion.div variants={variants.fadeUp} className="label inline-flex items-center gap-2 text-[var(--brand)]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent-mark)]"/>Road knowledge, made useful
          </motion.div>
          <motion.h1 variants={variants.textReveal} className="mt-4 max-w-2xl text-4xl font-extrabold leading-[1.02] tracking-[-.055em] md:text-6xl">
            <InfiniteTypewriter
              texts={[
                "Clear answers for Lahore's roads.",
                "Verified traffic rules.",
                "Real-time safety guidance.",
                "Intelligent road intelligence."
              ]}
              className="text-current"
            />
          </motion.h1>
          <motion.p variants={variants.fadeUp} className="mt-4 max-w-xl text-base leading-7 text-[var(--muted)]">Search verified traffic rules, fines, signs, and safety guidance—grounded in official sources.</motion.p>
          <motion.form variants={variants.scaleIn} onSubmit={submit} className="mt-7 flex max-w-2xl flex-col gap-2 rounded-xl border border-[var(--line-strong)] bg-[var(--surface)] p-2 shadow-[var(--shadow)] sm:flex-row">
            <GlowingBorder className="flex min-w-0 flex-1 items-center rounded-lg bg-[var(--surface)]">
              <label className="flex w-full items-center"><span className="sr-only">Ask a traffic question</span><Bot size={19} className="ml-3 shrink-0 text-[var(--brand)]"/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Ask about helmets, licences, signals or fines…" className="min-w-0 flex-1 bg-transparent px-3 py-3 text-sm outline-none"/></label>
            </GlowingBorder>
            <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} className="focus-ring flex items-center justify-center gap-2 rounded-lg bg-[var(--brand-fill)] px-5 py-3 text-sm font-bold text-[var(--on-brand)] transition hover:bg-[var(--brand-fill-hover)]">Ask AI <ArrowRight size={16}/></motion.button>
          </motion.form>
        </div>
      </section>
      <section aria-labelledby="quick-actions" className="mt-7">
        <div className="mb-3 flex items-center justify-between">
          <motion.h2 variants={variants.fadeUp} id="quick-actions" className="text-sm font-extrabold">Start with a task</motion.h2>
          <span className="label">Verified collections</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {actions.map(({id,title,copy,icon:Icon}, i) => (
            <FloatingCard key={id} delay={i * 0.2} duration={4 + i * 0.5}>
              <motion.button
                variants={variants.fadeUp}
                whileHover={{ y: -4, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onNavigate(id)}
                className="focus-ring surface group w-full rounded-xl p-5 text-left transition-shadow hover:border-[var(--line-strong)] hover:shadow-[var(--shadow)]"
              >
                <div className="flex items-start justify-between">
                  <span className="grid h-10 w-10 place-items-center rounded-lg bg-[var(--brand-soft)] text-[var(--brand)]"><Icon size={19}/></span>
                  <ArrowRight size={16} className="text-[var(--faint)] transition group-hover:translate-x-1 group-hover:text-[var(--brand)]"/>
                </div>
                <div className="mt-5 font-extrabold">{title}</div>
                <p className="mt-1 text-xs leading-5 text-[var(--muted)]">{copy}</p>
              </motion.button>
            </FloatingCard>
          ))}
        </div>
      </section>
      <div className="mt-7 grid gap-5 lg:grid-cols-[1.35fr_.65fr]">
        <section className="surface rounded-xl p-5 md:p-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="label">Your workspace</div>
              <h2 className="mt-1 text-lg font-extrabold">Recent questions</h2>
            </div>
            {recent.length > 0 && <button onClick={onClearRecent} className="focus-ring flex items-center gap-2 rounded-md px-2 py-2 text-xs font-bold text-[var(--muted)] hover:bg-[var(--surface-2)]"><RotateCcw size={14}/>Clear</button>}
          </div>
          {recent.length ? (
            <div className="mt-4 divide-y divide-[var(--line)]">
              {recent.slice(0,5).map((item, i) => (
                <motion.button
                  key={`${item.at}-${item.query}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => onAsk(item.query)}
                  className="focus-ring flex w-full items-center gap-3 py-3 text-left"
                >
                  <Clock3 size={15} className="shrink-0 text-[var(--faint)]"/><span className="min-w-0 flex-1 truncate text-sm font-semibold">{item.query}</span><ArrowRight size={14} className="text-[var(--faint)]"/>
                </motion.button>
              ))}
            </div>
          ) : (
            <div className="mt-5 rounded-lg bg-[var(--surface-2)] p-5 text-sm text-[var(--muted)]">Your recent AI questions will appear here for quick access.</div>
          )}
        </section>
        <aside className="rounded-xl bg-[var(--feature-bg)] p-6 text-[var(--feature-ink)]">
          <GlowingBorder glowColor="var(--feature-accent)" className="h-full w-full rounded-xl bg-transparent p-6">
            <BookOpenCheck size={22} className="text-[var(--feature-accent)]"/>
            <div className="label mt-5 text-[var(--feature-muted)]">Knowledge coverage</div>
            <p className="mt-2 text-lg font-bold leading-7">
              {stats ? `${stats.fine_records} fines, ${stats.traffic_signs} signs, and ${stats.rules} safety rules` : "Official road guidance"}
            </p>
            <button onClick={() => onNavigate("sources")} className="focus-ring mt-5 flex items-center gap-2 text-xs font-bold text-[var(--feature-accent)]">Review the sources <ArrowRight size={14}/></button>
          </GlowingBorder>
        </aside>
      </div>
    </motion.div>
  );
}
