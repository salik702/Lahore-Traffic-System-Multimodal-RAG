"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { api } from "@/lib/api";
import type { RecentSearch, Section, Stats } from "@/lib/models";
import { usePersistedState } from "@/lib/use-persisted-state";
import { AppShell } from "@/components/app-shell";
import { HomeView } from "@/components/home-view";
import { AssistantView } from "@/components/assistant-view";
import { FinesView, SignsView } from "@/components/catalog-views";
import { AccidentsView, SafetyView, SourcesView } from "@/components/knowledge-views";
import { VisualSearchView } from "@/components/visual-search-view";
import { DigitalTwinView } from "@/components/digital-twin-view";

const validSections: Section[] = ["home","digital-twin","assistant","fines","signs","safety","visual","accidents","sources"];

export default function HomePage() {
  const [section,setSection]=useState<Section>("home"); const [stats,setStats]=useState<Stats|null>(null); const [assistantQuery,setAssistantQuery]=useState(""); const [recent,setRecent]=usePersistedState<RecentSearch[]>("lti-recent-searches",[]);
  useEffect(()=>{const fromHash=window.location.hash.slice(1) as Section;if(validSections.includes(fromHash))queueMicrotask(()=>setSection(fromHash));api<Stats>("/statistics").then(setStats).catch(()=>undefined);},[]);
  function navigate(next:Section){setSection(next);history.replaceState(null,"",next==="home"?window.location.pathname:`#${next}`);window.scrollTo({top:0,behavior:"smooth"});}
  function ask(query:string){setAssistantQuery(query);navigate("assistant");}
  function remember(query:string){setRecent(items=>[{query,at:Date.now()},...items.filter(item=>item.query.toLowerCase()!==query.toLowerCase())].slice(0,8));}
  return <AppShell section={section} onNavigate={navigate}><AnimatePresence mode="wait"><motion.div key={section} initial={{opacity:0, scale:0.98, filter:"blur(4px)"}} animate={{opacity:1, scale:1, filter:"blur(0px)"}} exit={{opacity:0, scale:1.02, filter:"blur(4px)"}} transition={{duration:0.4, ease:[0.16, 1, 0.3, 1]}}>
    {section==="home"&&<HomeView stats={stats} recent={recent} onNavigate={navigate} onAsk={ask} onClearRecent={()=>setRecent([])}/>}
    {section==="digital-twin"&&<DigitalTwinView onAsk={ask}/>}
    {section==="assistant"&&<AssistantView initialQuery={assistantQuery} onSearched={remember}/>}
    {section==="fines"&&<FinesView/>}{section==="signs"&&<SignsView/>}
    {section==="safety"&&<SafetyView onAsk={ask}/>} {section==="visual"&&<VisualSearchView/>}
    {section==="accidents"&&<AccidentsView onAsk={ask}/>} {section==="sources"&&<SourcesView/>}
  </motion.div></AnimatePresence></AppShell>;
}
