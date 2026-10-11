"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { AnimatePresence, motion } from "motion/react";
import { ArrowUp, Bot, ExternalLink, FileText, LoaderCircle, PanelRight, ShieldCheck, X } from "lucide-react";
import { api, type ChatResponse } from "@/lib/api";
import { PageIntro } from "@/components/ui";
import { MediaImage } from "@/components/media-image";
import { variants } from "@/lib/motion-variants";
import { InfiniteTypewriter } from "./infinite-typewriter";

const prompts = ["What is the helmet rule and fine for a motorcycle?", "What documents must a driver produce on demand?", "What happens after a road accident?", "Explain how neural networks learn."];

export function AssistantView({ initialQuery, onSearched }: { initialQuery: string; onSearched: (q: string) => void }) {
  const [query, setQuery] = useState(initialQuery); const [submittedQuery, setSubmittedQuery] = useState(""); const [result, setResult] = useState<ChatResponse | null>(null); const [loading, setLoading] = useState(false); const [error, setError] = useState(""); const [drawer, setDrawer] = useState(false); const bottom = useRef<HTMLDivElement>(null); const sessionId = useRef<string | null>(null);
  const generalAnswer = result ? ["GENERAL_KNOWLEDGE", "GENERAL_CONVERSATION", "OUT_OF_DOMAIN"].includes(result.intent) : false;
  useEffect(() => { if (result) bottom.current?.scrollIntoView({behavior:"smooth",block:"nearest"}); }, [result]);
  async function submit(event?: FormEvent) { event?.preventDefault(); const message = query.trim(); if (!message || loading) return; setSubmittedQuery(message); setQuery(""); setResult(null); setLoading(true); setError(""); onSearched(message); try { const response = await api<ChatResponse>("/chat", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message,session_id:sessionId.current})}); sessionId.current = response.session_id; setResult(response); } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to retrieve a response. Try again."); } finally { setLoading(false); } }
  return <div><PageIntro
    kicker="AI assistant"
    title={<InfiniteTypewriter texts={["Traffic guidance and general questions.", "Ask anything about Lahore's roads.", "Your personal traffic co-pilot."]} />}
    description="Traffic answers use verified project sources. General questions are answered by Groq, with conversation context."/>
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_330px]">
      <section className="surface overflow-hidden rounded-xl"><div className="flex items-center border-b border-[var(--line)] px-4 py-3 md:px-5"><span className="grid h-9 w-9 place-items-center rounded-lg bg-[var(--brand-fill)] text-[var(--on-brand)]"><Bot size={18}/></span><div className="ml-3"><div className="text-sm font-extrabold">Traffic AI</div><div className="text-[11px] text-[var(--muted)]">Traffic sources and general answers</div></div>{result && <button onClick={() => setDrawer(true)} className="focus-ring ml-auto flex items-center gap-2 rounded-md border border-[var(--line)] px-3 py-2 text-xs font-bold"><PanelRight size={14}/>{generalAnswer ? "Response details" : "Evidence trace"}</button>}</div>
        <div className="scrollbar min-h-[430px] max-h-[60vh] overflow-y-auto p-4 md:p-6">
          <AnimatePresence mode="popLayout">
            {submittedQuery && (
              <motion.div
                initial={{ opacity: 0, x: 20, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                className="mb-5 ml-auto max-w-[85%] rounded-xl rounded-br-sm bg-[var(--brand-fill)] px-4 py-3 text-sm font-semibold text-[var(--on-brand)]"
                aria-label="Your submitted message"
              >
                {submittedQuery}
              </motion.div>
            )}
          </AnimatePresence>

          {!result && !loading && !error && !submittedQuery && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-auto max-w-2xl py-8"
            >
              <div className="rounded-xl bg-[var(--surface-2)] p-5">
                <ShieldCheck className="text-[var(--brand)]" size={20}/>
                <h2 className="mt-3 font-extrabold">What would you like to understand?</h2>
                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">Ask about traffic rules and safety, or explore general topics such as science, programming and everyday questions.</p>
              </div>
              <div className="mt-5 grid gap-2 sm:grid-cols-2">
                {prompts.map(prompt => (
                  <motion.button
                    key={prompt}
                    whileHover={{ scale: 1.02, borderColor: "var(--brand)" }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setQuery(prompt)}
                    className="focus-ring rounded-lg border border-[var(--line)] bg-[var(--surface)] p-4 text-left text-sm font-semibold transition"
                  >
                    {prompt}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}

          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid min-h-72 place-items-center text-center"
            >
              <div className="relative">
                <LoaderCircle className="mx-auto animate-spin text-[var(--brand)]"/>
                <div className="absolute inset-0 grid place-items-center"><div className="h-1 w-1 rounded-full bg-[var(--brand)] animate-ping"/></div>
              </div>
              <div className="mt-4 text-sm font-bold">Preparing your answer</div>
              <div className="loading-line mx-auto mt-4 h-1 w-28 rounded bg-[var(--accent)]"/>
            </motion.div>
          )}

          {error && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              role="alert"
              className="rounded-lg border border-[var(--danger)] bg-[var(--danger-soft)] p-4 text-sm text-[var(--danger)]"
            >
              {error}
            </motion.div>
          )}

          {result && !loading && (
            <motion.article
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            >
              <div className="mb-5 flex flex-wrap items-center gap-2">
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="label rounded-full bg-[var(--brand-soft)] px-3 py-1.5 text-[var(--brand)]"
                >
                  {answerLabel(result.intent)}
                </motion.span>
              </div>
              {generalAnswer && <p className="mb-4 text-xs text-[var(--faint)]">{result.model ? "General AI answer from Groq. No project sources or live data were consulted." : "General AI answer unavailable. Please try again shortly."}</p>}
              <div className="markdown text-[15px] leading-7 text-[var(--muted)]">
                <ReactMarkdown>{citationMarkdown(result)}</ReactMarkdown>
              </div>
              {result.images?.length > 0 && (
                <div className="mt-7">
                  <h3 className="text-sm font-extrabold">Retrieved source images</h3>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    {result.images.slice(0,6).map((item,index) => (
                      <motion.article
                        key={`${item.url}-${index}`}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--surface)]"
                      >
                        <div className="relative aspect-[16/10] bg-[var(--surface-2)]">
                          {item.url.startsWith("/media/signs/") ? (
                            <div className="grid h-full place-items-center p-5">
                              <div className="relative h-36 w-36">
                                <MediaImage src={item.url} alt={item.title} width={144} height={144} preview className="h-full w-full rounded-lg bg-white object-scale-down"/>
                              </div>
                            </div>
                          ) : (
                            <MediaImage src={item.url} alt={item.title} fill preview sizes="(max-width:640px) 100vw, 360px" className="object-cover"/>
                          )}
                          {item.verified_record_match && <span className="absolute left-2 top-2 rounded-full bg-[color-mix(in_srgb,var(--feature-bg)_90%,transparent)] px-2 py-1 text-[9px] font-bold uppercase tracking-wider text-[var(--feature-ink)]">Verified match</span>}
                        </div>
                        <div className="p-3">
                          <div className="text-sm font-bold">{item.title}</div>
                          <p className="mt-1 text-xs leading-5 text-[var(--muted)]">{item.description}</p>
                          <div className="mt-2 text-[10px] font-semibold text-[var(--faint)]">{[item.location||item.region,item.date ? `${item.date_kind === "photograph" ? "Photo date: " : ""}${item.date}` : null,item.source].filter(Boolean).join(" · ")}</div>
                          <div className="mt-1 text-[10px] text-[var(--faint)]">{item.author}{item.author && item.license ? " · " : ""}{item.license}</div>
                          {item.license_url && <a href={item.license_url} target="_blank" rel="noreferrer" className="mt-2 block text-xs text-[var(--brand)]">License terms ↗</a>}
                          <a href={item.source_url} target="_blank" rel="noreferrer" className="mt-3 block text-xs font-bold text-[var(--brand)]">View original source ↗</a>
                        </div>
                      </motion.article>
                    ))}
                  </div>
                </div>
              )}
              {result.citations.length > 0 && (
                <div className="mt-7 border-t border-[var(--line)] pt-5">
                  <h3 className="flex items-center gap-2 text-sm font-extrabold">
                    <FileText size={16}/>Sources used
                  </h3>
                  <div className="mt-3 space-y-2">
                    {result.citations.map((citation,index) => (
                      <motion.a
                        key={`${citation.id}-${index}`}
                        href={citation.url}
                        target="_blank"
                        rel="noreferrer"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="focus-ring flex items-start gap-3 rounded-lg border border-[var(--line)] p-3 transition hover:border-[var(--line-strong)]"
                      >
                        <span className="font-mono text-xs font-bold text-[var(--brand)]">{index+1}</span>
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-bold">{citation.title}</span>
                          <span className="mt-1 block text-[11px] text-[var(--muted)]">{citation.organization} · {citation.region}{citation.page ? ` · Page ${citation.page}` : "" }</span>
                        </span>
                        <ExternalLink size={14} className="shrink-0 text-[var(--faint)]"/>
                      </motion.a>
                    ))}
                  </div>
                </div>
              )}
              <div ref={bottom}/>
            </motion.article>
          )}
        </div>
        <form onSubmit={submit} className="flex items-end gap-2 border-t border-t-[var(--line)] bg-[var(--surface-2)] p-3">
          <label className="min-w-0 flex-1">
            <span className="sr-only">Your question</span>
            <textarea
              rows={1}
              value={query}
              onChange={e=>setQuery(e.target.value)}
              onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();void submit();}}}
              placeholder="Ask about traffic or any general topic…"
              className="max-h-32 min-h-12 w-full resize-none rounded-lg border border-[var(--line)] bg-[var(--surface)] px-4 py-3 text-sm outline-none focus:border-[var(--brand)] transition-all"
            />
          </label>
          <motion.button
            disabled={loading || !query.trim()}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            aria-label="Send question"
            className="focus-ring grid h-12 w-12 place-items-center rounded-lg bg-[var(--brand-fill)] text-[var(--on-brand)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <ArrowUp size={19}/>
          </motion.button>
        </form>
      </section>
      <aside className="space-y-4">
        <div className="surface rounded-xl p-5">
          <div className="label">How answers work</div>
          <ol className="mt-4 space-y-4">
            {["Understand your question","Use traffic sources when relevant","Answer with sources or general knowledge","Keep the conversation in context"].map((item,index)=><li key={item} className="flex gap-3 text-sm"><span className="grid h-6 w-6 shrink-to-fit shrink-0 place-items-center rounded-full bg-[var(--brand-soft)] font-mono text-[10px] font-bold text-[var(--brand)]">{index+1}</span><span className="pt-0.5 font-semibold">{item}</span></li>)}
          </ol>
        </div>
        <div className="rounded-xl border border-[var(--line)] bg-[var(--accent-soft)] p-5">
          <div className="text-sm font-extrabold">A careful co-pilot</div>
          <p className="mt-2 text-xs leading-5 text-[var(--muted)]">Use this for guidance and source discovery. On the road, follow posted signs and directions from traffic authorities.</p>
        </div>
      </aside>
    </div>
    <AnimatePresence>{drawer && result && <EvidenceDrawer result={result} close={() => setDrawer(false)}/>}</AnimatePresence>
  </div>;
}

function EvidenceDrawer({result,close}:{result:ChatResponse;close:()=>void}) { return <><motion.button aria-label="Close evidence trace" onClick={close} className="fixed inset-0 z-40 bg-black/45" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}/><motion.aside role="dialog" aria-modal="true" aria-label="Evidence trace" className="fixed inset-y-0 right-0 z-50 w-full max-w-xl overflow-y-auto bg-[var(--surface)] p-5 shadow-2xl md:p-7" initial={{x:"100%"}} animate={{x:0}} exit={{x:"100%"}} transition={{type:"spring",damping:28,stiffness:260}}><div className="flex items-start justify-between"><div><div className="label text-[var(--brand)]">Evidence trace</div><h2 className="mt-2 text-2xl font-extrabold">How this answer was formed</h2><p className="mt-2 text-xs text-[var(--muted)]">{result.retrieval_engine}</p></div><button onClick={close} aria-label="Close" className="focus-ring grid h-9 w-9 place-items-center rounded-md border border-[var(--line)]"><X size={18}/></button></div><div className="mt-7 space-y-5">{result.trace.map((step,index)=><div key={`${step.label}-${index}`} className="grid grid-cols-[28px_1fr] gap-3"><span className="grid h-7 w-7 place-items-center rounded-full bg-[var(--brand-fill)] font-mono text-[10px] font-bold text-[var(--on-brand)]">{index+1}</span><div><div className="text-sm font-bold">{step.label}</div><p className="mt-1 text-xs leading-5 text-[var(--muted)]">{step.detail}</p></div></div>)}</div><div className="mt-8 border-t border-[var(--line)] pt-6"><h3 className="text-sm font-extrabold">{result.evidence.length ? "Retrieved evidence" : "No project sources were consulted"}</h3><div className="mt-3 space-y-2">{result.evidence.map((e,index)=><details key={`${e.id}-${index}`} className="rounded-lg border border-[var(--line)] p-4"><summary className="cursor-pointer text-sm font-bold">{e.title}<span className="label ml-2 text-[var(--brand)]">{e.modality} · {e.score.toFixed(2)}</span></summary><p className="mt-3 text-xs leading-6 text-[var(--muted)]">{e.text}</p></details>)}</div></div></motion.aside></>; }

function answerLabel(intent: string) {
  const labels: Record<string, string> = { ACCIDENT_GUIDANCE: "Accident guidance", EMERGENCY_GUIDANCE: "Emergency guidance", ACCIDENT_STATISTICS: "Accident statistics", ACCIDENT_INFORMATION: "Accident reports", ACCIDENT_IMAGE_REQUEST: "Accident photographs", TRAFFIC_FINE: "Traffic fine", TRAFFIC_SIGN: "Traffic signs", TRAFFIC_RULE: "Traffic rules", ROAD_SAFETY: "Road safety", DOCUMENT_SEARCH: "Sources", GENERAL_KNOWLEDGE: "General answer", GENERAL_CONVERSATION: "Conversation", OUT_OF_DOMAIN: "General answer" };
  return labels[intent] ?? "Traffic assistant";
}

function citationMarkdown(result: ChatResponse) {
  return result.answer.replace(/\[(\d+)\](?!\()/g, (match, number: string) =>
    result.citations[Number(number) - 1] ? "["+number+"](#source-"+number+")" : match);
}
