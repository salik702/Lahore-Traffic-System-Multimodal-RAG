"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "motion/react";
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Bot,
  BrainCircuit,
  CarFront,
  CheckCircle2,
  Clock,
  Compass,
  FastForward,
  Flame,
  Gauge,
  GitCompare,
  HelpCircle,
  History,
  Layers,
  Pause,
  Play,
  RefreshCw,
  Ruler,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Volume2,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  ComparisonReport,
  ExperimentRecord,
  SimulationMetrics,
  SimulationNetwork,
  SimulationRiskEvent,
  SimulationScenario,
  SimulationSignal,
  SimulationSnapshot,
  SimulationVehicle,
  TrainingStatus,
} from "@/lib/models";

// Dynamic import for Leaflet map to prevent SSR window reference error
const SimulationMap = dynamic(
  () => import("./simulation-map").then((mod) => mod.SimulationMap),
  { ssr: false }
);

interface DigitalTwinViewProps {
  onAsk: (query: string) => void;
}

type TabMode = "twin" | "compare" | "scenarios" | "risk" | "training";

export function DigitalTwinView({ onAsk }: DigitalTwinViewProps) {
  const [activeTab, setActiveTab] = useState<TabMode>("twin");

  // Live state
  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [network, setNetwork] = useState<SimulationNetwork | null>(null);
  const [scenarios, setScenarios] = useState<SimulationScenario[]>([]);
  const [selectedSignalId, setSelectedSignalId] = useState<string | null>(null);
  const [selectedRoadId, setSelectedRoadId] = useState<string | null>(null);

  // Controls
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [demandScale, setDemandScale] = useState(1.0);
  const [controllerMode, setControllerMode] = useState<string>("fixed_time");
  const [scenarioName, setScenarioName] = useState<string>("normal");
  const [loadingAction, setLoadingAction] = useState(false);

  // Experiments & comparison
  const [comparisonReport, setComparisonReport] = useState<ComparisonReport | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [isComparing, setIsComparing] = useState(false);
  const [compareDuration, setCompareDuration] = useState(120);

  // RL Training
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Initial loads: Network, Scenarios, Initial Snapshot, Experiments
  useEffect(() => {
    api<SimulationNetwork>("/simulation/network")
      .then((roadNetwork) => {
        setNetwork(roadNetwork);
        setSelectedRoadId((current) => current ?? roadNetwork.edges.find((edge) => edge.highway === "primary" || edge.highway === "trunk")?.id ?? roadNetwork.edges[0]?.id ?? null);
      })
      .catch(() => undefined);

    api<{ scenarios: SimulationScenario[] }>("/simulation/scenarios")
      .then((res) => setScenarios(res.scenarios))
      .catch(() => undefined);

    api<SimulationSnapshot>("/simulation/snapshot")
      .then((snap) => {
        setSnapshot(snap);
        setIsRunning(snap.running);
        setSpeed(snap.speed);
        setDemandScale(snap.demand_scale);
        setControllerMode(snap.controller_mode);
        setScenarioName(snap.scenario);
      })
      .catch(() => undefined);

    api<{ experiments: ExperimentRecord[] }>("/simulation/experiments")
      .then((res) => setExperiments(res.experiments))
      .catch(() => undefined);

    api<TrainingStatus>("/simulation/training/status")
      .then(setTrainingStatus)
      .catch(() => undefined);
  }, []);

  // Live streaming / polling loop
  useEffect(() => {
    let unmounted = false;

    // Polling function for robust updates
    async function poll() {
      try {
        const snap = await api<SimulationSnapshot>("/simulation/snapshot");
        if (!unmounted) {
          setSnapshot(snap);
          setIsRunning(snap.running);
        }
      } catch {
        // ignore network error
      }
    }

    // Try WebSocket connection
    const wsUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1")
      .replace(/^http/, "ws") + "/simulation/ws";

    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data: SimulationSnapshot = JSON.parse(event.data);
          if (!unmounted) {
            setSnapshot(data);
            setIsRunning(data.running);
          }
        } catch {
          // parse error
        }
      };

      ws.onerror = () => {
        // Fallback to polling on WS error
        if (!pollTimerRef.current) {
          pollTimerRef.current = setInterval(poll, isRunning ? 350 : 1200);
        }
      };

      ws.onclose = () => {
        // Fallback to polling when WS closes
        if (!pollTimerRef.current) {
          pollTimerRef.current = setInterval(poll, isRunning ? 350 : 1200);
        }
      };
    } catch {
      pollTimerRef.current = setInterval(poll, isRunning ? 350 : 1200);
    }

    // Polling fallback interval
    pollTimerRef.current = setInterval(poll, isRunning ? 350 : 1200);

    return () => {
      unmounted = true;
      if (ws) ws.close();
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [isRunning]);

  // Simulation lifecycle handlers
  async function handleControl(action: "start" | "pause" | "resume" | "reset" | "step") {
    setLoadingAction(true);
    try {
      await api("/simulation/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, controller_mode: controllerMode, scenario: scenarioName, demand_scale: demandScale }),
      });
      const snap = await api<SimulationSnapshot>("/simulation/snapshot");
      setSnapshot(snap);
      setIsRunning(snap.running);
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoadingAction(false);
    }
  }

  async function updateConfig(newConfig: {
    speed?: number;
    demand_scale?: number;
    controller_mode?: string;
    scenario?: string;
  }) {
    try {
      if (newConfig.speed !== undefined) setSpeed(newConfig.speed);
      if (newConfig.demand_scale !== undefined) setDemandScale(newConfig.demand_scale);
      if (newConfig.controller_mode !== undefined) setControllerMode(newConfig.controller_mode);
      if (newConfig.scenario !== undefined) setScenarioName(newConfig.scenario);

      await api("/simulation/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newConfig),
      });

      const snap = await api<SimulationSnapshot>("/simulation/snapshot");
      setSnapshot(snap);
    } catch (err) {
      console.error(err);
    }
  }

  // Run Comparison Experiment
  async function runComparison() {
    setIsComparing(true);
    try {
      const res = await api<ComparisonReport>("/simulation/experiment/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario: scenarioName,
          demand_scale: demandScale,
          duration: compareDuration,
          seed: 42,
        }),
      });
      setComparisonReport(res);
      // Refresh experiment history list
      const exps = await api<{ experiments: ExperimentRecord[] }>("/simulation/experiments");
      setExperiments(exps.experiments);
    } catch (err) {
      console.error(err);
    } finally {
      setIsComparing(false);
    }
  }

  // Run RL Training
  async function startRLTraining() {
    setIsTraining(true);
    try {
      await api("/simulation/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ episodes: 5, episode_duration: 60.0 }),
      });

      // Poll training status
      const poll = setInterval(async () => {
        try {
          const status = await api<TrainingStatus>("/simulation/training/status");
          setTrainingStatus(status);
          if (status.status === "completed" || status.status === "failed") {
            clearInterval(poll);
            setIsTraining(false);
          }
        } catch {
          clearInterval(poll);
          setIsTraining(false);
        }
      }, 1500);
    } catch (err) {
      console.error(err);
      setIsTraining(false);
    }
  }

  const metrics = snapshot?.metrics;
  const signals = snapshot?.signals || [];
  const vehicles = snapshot?.vehicles || [];
  const riskEvents = snapshot?.risk_events || [];

  const selectedSignal = signals.find((s) => s.id === selectedSignalId) || signals[0];
  const selectedRoad = network?.edges.find((edge) => edge.id === selectedRoadId) ?? network?.edges[0];
  const selectedRoadTraffic = selectedRoad ? snapshot?.edge_traffic?.[selectedRoad.id] : undefined;
  const selectedRoadStorage = selectedRoad ? Math.max(1, Math.floor((selectedRoad.length / 7) * selectedRoad.lanes)) : 0;
  const selectedRoadLoad = selectedRoadStorage ? Math.min(100, Math.round(((selectedRoadTraffic?.vehicles ?? 0) / selectedRoadStorage) * 100)) : 0;

  return (
    <div className="space-y-6 pb-12">
      {/* Top Header Card */}
      <div className="relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-xl backdrop-blur-xl">
        <div className="relative z-10 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--brand-fill)] text-[var(--on-brand)] shadow-md">
                <BrainCircuit size={19} />
              </span>
              <h1 className="text-xl font-extrabold tracking-tight sm:text-2xl text-[var(--ink)]">
                Lahore Traffic Digital Twin
              </h1>
              <span className="rounded-full border border-blue-500/30 bg-blue-500/10 px-2.5 py-0.5 text-[11px] font-bold text-blue-600 dark:text-blue-400">
                Kalma Chowk Network
              </span>
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <CheckCircle2 size={12} />
                {snapshot?.engine_type === "sumo_traci" ? "Eclipse SUMO / TraCI" : "Native OSM microsimulation"}
              </span>
            </div>
            <p className="text-xs text-[var(--muted)] max-w-3xl">
              Microscopic multi-agent simulation using real OpenStreetMap geometry for Lahore, Pakistan. Signal control, vehicle motion, risk indicators, and what-if outcomes are derived from simulation state.
            </p>
          </div>

          {/* Quick RAG Assistant Bridge */}
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => onAsk("Why is Kalma Chowk congested right now and which signal causes the longest queue?")}
              className="focus-ring flex items-center gap-1.5 rounded-xl border border-[var(--brand)] bg-[var(--brand-soft)] px-3.5 py-2 text-xs font-extrabold text-[var(--brand)] transition hover:scale-102 hover:shadow-md"
            >
              <Bot size={15} />
              <span>Ask AI About Traffic</span>
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="mt-6 flex flex-wrap items-center gap-1.5 border-t border-[var(--line)] pt-4">
          {[
            { id: "twin", label: "Live Digital Twin", icon: Activity },
            { id: "compare", label: "Before & After AI Comparison", icon: GitCompare },
            { id: "scenarios", label: "What-If Scenarios", icon: Layers },
            { id: "risk", label: "Collision & Conflict Analysis", icon: ShieldAlert },
            { id: "training", label: "RL Training Hub", icon: BrainCircuit },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setActiveTab(id as TabMode)}
              className={`focus-ring relative flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-bold transition ${
                activeTab === id
                  ? "bg-[var(--brand-fill)] text-[var(--on-brand)] shadow-sm font-extrabold"
                  : "text-[var(--muted)] hover:bg-[var(--surface-2)] hover:text-[var(--ink)]"
              }`}
            >
              <Icon size={14} />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* TAB 1: LIVE DIGITAL TWIN */}
      {activeTab === "twin" && (
        <div className="space-y-6">
          {/* Key Metrics HUD Bar */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 shadow-sm">
              <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--muted)]">
                <span>Congestion Level</span>
                <Flame size={14} className={metrics?.congestion_level === "high" || metrics?.congestion_level === "severe" ? "text-rose-500" : "text-emerald-500"} />
              </div>
              <div className="mt-1 flex items-baseline gap-1.5">
                <span className={`text-base font-black uppercase ${
                  metrics?.congestion_level === "low" ? "text-emerald-500" :
                  metrics?.congestion_level === "moderate" ? "text-amber-500" : "text-rose-500"
                }`}>
                  {metrics?.congestion_level || "low"}
                </span>
                <span className="text-[10px] text-[var(--muted)]">({((metrics?.occupancy || 0) * 100).toFixed(0)}% occ)</span>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 shadow-sm">
              <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--muted)]">
                <span>Avg Waiting Time</span>
                <Clock size={14} className="text-blue-500" />
              </div>
              <div className="mt-1 flex items-baseline gap-1">
                <span className="text-lg font-black text-[var(--ink)]">
                  {metrics?.average_waiting_time?.toFixed(1) || "0.0"}
                </span>
                <span className="text-[10px] text-[var(--muted)]">sec/veh</span>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 shadow-sm">
              <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--muted)]">
                <span>Peak Queue</span>
                <Sliders size={14} className="text-purple-500" />
              </div>
              <div className="mt-1 flex items-baseline gap-1">
                <span className="text-lg font-black text-[var(--ink)]">
                  {metrics?.max_queue_length || 0}
                </span>
                <span className="text-[10px] text-[var(--muted)]">veh (avg: {metrics?.average_queue_length?.toFixed(1) || "0"})</span>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 shadow-sm">
              <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--muted)]">
                <span>Throughput</span>
                <Gauge size={14} className="text-emerald-500" />
              </div>
              <div className="mt-1 flex items-baseline gap-1">
                <span className="text-lg font-black text-[var(--ink)]">
                  {metrics?.throughput_per_hour?.toFixed(0) || "0"}
                </span>
                <span className="text-[10px] text-[var(--muted)]">veh/hr</span>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 shadow-sm">
              <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--muted)]">
                <span>Active / Completed</span>
                <Activity size={14} className="text-amber-500" />
              </div>
              <div className="mt-1 flex items-baseline gap-1">
                <span className="text-lg font-black text-[var(--ink)]">
                  {metrics?.total_vehicles || 0}
                </span>
                <span className="text-[10px] text-[var(--muted)]">/ {metrics?.vehicles_completed || 0} done</span>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3.5 shadow-sm">
              <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--muted)]">
                <span>Simulated Risks</span>
                <ShieldAlert size={14} className="text-rose-500" />
              </div>
              <div className="mt-1 flex items-baseline gap-1">
                <span className="text-lg font-black text-rose-500">
                  {metrics?.risk_events || 0}
                </span>
                <span className="text-[10px] text-[var(--muted)]">({metrics?.simulated_collisions || 0} collisions)</span>
              </div>
            </div>
          </div>

          {/* Interactive Simulation Dashboard: Map + Control Sidebar */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Map Canvas (8 cols) */}
            <div className="lg:col-span-8 flex flex-col space-y-4">
              <div className="h-[540px] w-full">
                <SimulationMap
                  network={network}
                  vehicles={vehicles}
                  signals={signals}
                  riskEvents={riskEvents}
                  edgeTraffic={snapshot?.edge_traffic ?? {}}
                  closedEdges={snapshot?.closed_edges}
                  selectedSignalId={selectedSignalId}
                  selectedRoadId={selectedRoad?.id}
                  onSelectSignal={setSelectedSignalId}
                  onSelectRoad={setSelectedRoadId}
                  congestionLevel={metrics?.congestion_level}
                />
              </div>

              {/* Quick AI Prompts below Map */}
              <div className="flex flex-wrap items-center gap-2 rounded-xl border border-[var(--line)] bg-[var(--surface)] p-3 text-xs">
                <span className="font-bold text-[var(--muted)] flex items-center gap-1">
                  <Sparkles size={13} className="text-amber-500" /> Grounded Inquiries:
                </span>
                {[
                  "Why is this intersection congested?",
                  "Which signal is causing the longest queue?",
                  "How did the AI controller perform?",
                  "What happened after closing this road?",
                  "Recommend a traffic-control strategy",
                ].map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => onAsk(prompt)}
                    className="focus-ring rounded-lg border border-[var(--line)] bg-[var(--surface-2)] px-2.5 py-1 text-[11px] font-medium text-[var(--ink)] transition hover:border-[var(--brand)] hover:text-[var(--brand)]"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            {/* Sidebar Controls (4 cols) */}
            <div className="lg:col-span-4 space-y-4">
              {/* Network-wide counts come directly from the simulation engine. */}
              <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-sm space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-xs font-extrabold uppercase tracking-wider text-[var(--ink)]">Live Traffic Summary</h2>
                    <p className="mt-0.5 text-[10px] text-[var(--muted)]">Simulation totals · resets with the run</p>
                  </div>
                  <Activity size={15} className={isRunning ? "text-emerald-500" : "text-[var(--muted)]"} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                    <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><CheckCircle2 size={12} />Trips completed</div>
                    <div className="mt-1 text-xl font-black text-[var(--ink)]">{metrics?.vehicles_completed ?? 0}</div>
                    <div className="text-[9px] text-[var(--muted)]">Completed route and exited</div>
                  </div>
                  <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                    <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><CarFront size={12} />Vehicles entered</div>
                    <div className="mt-1 text-xl font-black text-[var(--ink)]">{metrics?.spawned ?? 0}</div>
                    <div className="text-[9px] text-[var(--muted)]">Since last reset</div>
                  </div>
                  <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                    <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><Activity size={12} />On network</div>
                    <div className="mt-1 text-xl font-black text-[var(--ink)]">{metrics?.total_vehicles ?? 0}</div>
                    <div className="text-[9px] text-[var(--muted)]">Active vehicles now</div>
                  </div>
                  <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                    <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><Clock size={12} />Waiting now</div>
                    <div className="mt-1 text-xl font-black text-[var(--ink)]">{metrics?.vehicles_waiting ?? 0}</div>
                    <div className="text-[9px] text-[var(--muted)]">Stopped or queued</div>
                  </div>
                  <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                    <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><Gauge size={12} />Average speed</div>
                    <div className="mt-1 text-xl font-black text-[var(--ink)]">{((metrics?.mean_speed ?? 0) * 3.6).toFixed(1)}<span className="ml-1 text-[10px] font-semibold text-[var(--muted)]">km/h</span></div>
                    <div className="text-[9px] text-[var(--muted)]">Active vehicles</div>
                  </div>
                  <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                    <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><ArrowUpRight size={12} />Flow rate</div>
                    <div className="mt-1 text-xl font-black text-[var(--ink)]">{metrics?.throughput_per_hour?.toFixed(0) ?? 0}<span className="ml-1 text-[10px] font-semibold text-[var(--muted)]">/hr</span></div>
                    <div className="text-[9px] text-[var(--muted)]">Completed trips, since start</div>
                  </div>
                </div>
                <div className="flex items-center justify-between rounded-lg border border-amber-500/20 bg-amber-500/5 px-2.5 py-2 text-[10px]">
                  <span className="font-semibold text-[var(--muted)]">Removed by blocked routes</span>
                  <span className="font-black text-[var(--ink)]">{metrics?.vehicles_stranded ?? 0}</span>
                </div>
                <p className="text-[9px] leading-relaxed text-[var(--muted)]">These are modelled counts on the imported road network, not live roadside sensor measurements.</p>
              </div>

              {selectedRoad && (
                <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-sm space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <span className="text-[10px] font-extrabold uppercase tracking-wider text-[var(--muted)]">Selected road segment</span>
                      <h2 className="mt-0.5 truncate text-sm font-extrabold text-[var(--ink)]" title={selectedRoad.name}>{selectedRoad.name}</h2>
                    </div>
                    <CarFront size={18} className="mt-1 shrink-0 text-[var(--brand)]" />
                  </div>
                  <select
                    aria-label="Choose a road segment"
                    value={selectedRoad.id}
                    onChange={(event) => setSelectedRoadId(event.target.value)}
                    className="focus-ring w-full rounded-lg border border-[var(--line)] bg-[var(--surface-2)] px-2.5 py-2 text-xs font-semibold text-[var(--ink)]"
                  >
                    {(network?.edges ?? []).map((edge) => (
                      <option key={edge.id} value={edge.id}>{edge.name} · {edge.lanes} lane{edge.lanes === 1 ? "" : "s"} · {Math.round(edge.length)} m</option>
                    ))}
                  </select>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                      <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><CarFront size={12} />On segment now</div>
                      <div className="mt-0.5 text-lg font-black text-[var(--ink)]">{selectedRoadTraffic?.vehicles ?? 0}<span className="ml-1 text-[10px] font-semibold text-[var(--muted)]">vehicles</span></div>
                      <div className="text-[10px] text-[var(--muted)]">{selectedRoadTraffic?.waiting ?? 0} waiting · {((selectedRoadTraffic?.mean_speed ?? 0) * 3.6).toFixed(0)} km/h avg</div>
                    </div>
                    <div className="rounded-lg bg-[var(--surface-2)] p-2.5">
                      <div className="flex items-center gap-1 text-[10px] font-semibold text-[var(--muted)]"><Activity size={12} />Segment entries</div>
                      <div className="mt-0.5 text-lg font-black text-[var(--ink)]">{selectedRoadTraffic?.entries_last_60s ?? 0}<span className="ml-1 text-[10px] font-semibold text-[var(--muted)]">vehicles</span></div>
                      <div className="text-[10px] text-[var(--muted)]">last {Math.min(60, snapshot?.time ?? 0).toFixed(1)} simulated seconds</div>
                      <div className="text-[9px] text-[var(--muted)]">{selectedRoadTraffic?.entries_total ?? 0} entries since reset</div>
                    </div>
                  </div>
                  <div className="space-y-1.5 border-t border-[var(--line)] pt-2.5">
                    <div className="flex justify-between text-xs"><span className="flex items-center gap-1 text-[var(--muted)]"><Ruler size={12} />Length · direction</span><span className="font-bold text-[var(--ink)]">{Math.round(selectedRoad.length)} m · {selectedRoad.from} → {selectedRoad.to}</span></div>
                    <div className="flex justify-between text-xs"><span className="text-[var(--muted)]">Road class</span><span className="font-bold capitalize text-[var(--ink)]">{selectedRoad.highway.replaceAll("_", " ")}</span></div>
                    <div className="flex justify-between text-xs"><span className="text-[var(--muted)]">Lanes · estimated width</span><span className="font-bold text-[var(--ink)]">{selectedRoad.lanes} · ~{(selectedRoad.lanes * 3.2).toFixed(1)} m</span></div>
                    <div className="text-right text-[9px] text-[var(--muted)]">{selectedRoad.lane_source} · width assumes 3.2 m per lane</div>
                    <div className="flex justify-between text-xs"><span className="text-[var(--muted)]">Speed · source</span><span className="font-bold text-[var(--ink)]">{Math.round(selectedRoad.speed * 3.6)} km/h · {selectedRoad.speed_source}</span></div>
                    <div className="flex justify-between text-xs"><span className="text-[var(--muted)]">Estimated stopped storage</span><span className="font-bold text-[var(--ink)]">~{selectedRoadStorage} car spaces</span></div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-[var(--line)]"><div className={`h-full rounded-full ${selectedRoadLoad >= 80 ? "bg-rose-500" : selectedRoadLoad >= 45 ? "bg-amber-500" : "bg-emerald-500"}`} style={{ width: `${selectedRoadLoad}%` }} /></div>
                    <div className="flex justify-between text-[10px] text-[var(--muted)]"><span>Estimated segment fill</span><span>{selectedRoadLoad}% · ~{Math.max(0, selectedRoadStorage - (selectedRoadTraffic?.vehicles ?? 0))} spaces free</span></div>
                  </div>
                  <p className="text-[9px] leading-relaxed text-[var(--muted)]">Traffic counts are from this simulation. Lane width and stopped storage are estimates; OSM does not provide complete pavement-width measurements.</p>
                  <p className="text-[10px] text-[var(--brand)]">Select a different segment here or click a road on the map.</p>
                </div>
              )}

              {/* Playback Controls Card */}
              <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-extrabold uppercase tracking-wider text-[var(--muted)]">
                    Playback Controls
                  </span>
                  <div className="flex items-center gap-2">
                    <span className={`inline-block h-2 w-2 rounded-full ${isRunning ? "bg-emerald-500 animate-ping" : "bg-zinc-400"}`}></span>
                    <span className="text-xs font-bold text-[var(--ink)]">
                      {isRunning ? "Simulating" : "Paused"} ({snapshot?.time?.toFixed(1) || 0}s)
                    </span>
                  </div>
                </div>

                {/* Primary Action Buttons */}
                <div className="grid grid-cols-4 gap-2">
                  {isRunning ? (
                    <button
                      type="button"
                      disabled={loadingAction}
                      onClick={() => handleControl("pause")}
                      className="focus-ring col-span-2 flex items-center justify-center gap-1.5 rounded-lg bg-amber-500 py-2.5 text-xs font-bold text-white shadow-md transition hover:bg-amber-600 disabled:opacity-50"
                    >
                      <Pause size={15} /> Pause
                    </button>
                  ) : (
                    <button
                      type="button"
                      disabled={loadingAction}
                      onClick={() => handleControl("start")}
                      className="focus-ring col-span-2 flex items-center justify-center gap-1.5 rounded-lg bg-emerald-600 py-2.5 text-xs font-bold text-white shadow-md transition hover:bg-emerald-700 disabled:opacity-50"
                    >
                      <Play size={15} /> Run Simulation
                    </button>
                  )}

                  <button
                    type="button"
                    disabled={loadingAction}
                    onClick={() => handleControl("step")}
                    className="focus-ring flex items-center justify-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] py-2.5 text-xs font-bold text-[var(--ink)] transition hover:bg-[var(--surface-3)]"
                    title="Single Step (+0.5s)"
                  >
                    <FastForward size={14} /> Step
                  </button>

                  <button
                    type="button"
                    disabled={loadingAction}
                    onClick={() => handleControl("reset")}
                    className="focus-ring flex items-center justify-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--surface-2)] py-2.5 text-xs font-bold text-[var(--ink)] transition hover:bg-[var(--surface-3)]"
                    title="Reset Simulation State"
                  >
                    <RotateCcw size={14} /> Reset
                  </button>
                </div>

                {/* Speed Multiplier */}
                <div className="space-y-1.5 pt-2 border-t border-[var(--line)]">
                  <div className="flex items-center justify-between text-xs font-semibold text-[var(--muted)]">
                    <span>Simulation Speed</span>
                    <span className="font-bold text-[var(--ink)]">{speed}×</span>
                  </div>
                  <div className="grid grid-cols-4 gap-1.5">
                    {[0.5, 1.0, 2.0, 5.0].map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => updateConfig({ speed: s })}
                        className={`focus-ring rounded-md py-1 text-xs font-bold transition ${
                          speed === s
                            ? "bg-[var(--brand-fill)] text-[var(--on-brand)]"
                            : "border border-[var(--line)] bg-[var(--surface-2)] text-[var(--muted)] hover:text-[var(--ink)]"
                        }`}
                      >
                        {s}×
                      </button>
                    ))}
                  </div>
                </div>

                {/* Traffic Demand Slider */}
                <div className="space-y-1.5 pt-2 border-t border-[var(--line)]">
                  <div className="flex items-center justify-between text-xs font-semibold text-[var(--muted)]">
                    <span>Traffic Arrival Demand</span>
                    <span className="font-bold text-[var(--ink)]">{demandScale.toFixed(1)}×</span>
                  </div>
                  <input
                    type="range"
                    min="0.4"
                    max="3.0"
                    step="0.2"
                    value={demandScale}
                    onChange={(e) => updateConfig({ demand_scale: parseFloat(e.target.value) })}
                    className="w-full accent-[var(--brand)] cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-[var(--muted)]">
                    <span>Off-peak (0.4×)</span>
                    <span>Normal (1.0×)</span>
                    <span>Heavy (3.0×)</span>
                  </div>
                </div>

                {/* Controller Strategy Mode */}
                <div className="space-y-1.5 pt-2 border-t border-[var(--line)]">
                  <label className="block text-xs font-semibold text-[var(--muted)]">
                    Signal Control Strategy
                  </label>
                  <div className="grid grid-cols-3 gap-1">
                    {[
                      { id: "fixed_time", label: "Fixed-Time" },
                      { id: "max_pressure", label: "Pressure" },
                      { id: "rl", label: "AI RL Agent" },
                    ].map((mode) => (
                      <button
                        key={mode.id}
                        type="button"
                        onClick={() => updateConfig({ controller_mode: mode.id })}
                        className={`focus-ring rounded-lg py-1.5 text-xs font-bold transition ${
                          controllerMode === mode.id
                            ? "bg-blue-600 text-white shadow-sm"
                            : "border border-[var(--line)] bg-[var(--surface-2)] text-[var(--muted)] hover:text-[var(--ink)]"
                        }`}
                      >
                        {mode.label}
                      </button>
                    ))}
                  </div>
                  <p className="text-[10px] text-[var(--muted)]">
                    {controllerMode === "fixed_time" && "Deterministic fixed cyclic phase durations with safe yellow transition clearance intervals."}
                    {controllerMode === "max_pressure" && "Coordinated max-pressure heuristic balancing queue differentials across intersection approaches."}
                    {controllerMode === "rl" && "Deep Q-Network RL agent coordinating multi-intersection phase switching based on queue states."}
                  </p>
                </div>

                {/* Scenario Dropdown */}
                <div className="space-y-1.5 pt-2 border-t border-[var(--line)]">
                  <label className="block text-xs font-semibold text-[var(--muted)]">
                    Active What-If Scenario
                  </label>
                  <select
                    value={scenarioName}
                    onChange={(e) => updateConfig({ scenario: e.target.value })}
                    className="focus-ring w-full rounded-lg border border-[var(--line)] bg-[var(--surface-2)] px-3 py-2 text-xs font-bold text-[var(--ink)]"
                  >
                    {scenarios.map((sc) => (
                      <option key={sc.id} value={sc.id}>
                        {sc.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-[10px] leading-4 text-[var(--muted)]">
                    High traffic loads 3× synthetic arrivals (about 6 vehicles/second across this network), including a larger motorcycle share. Counts are for simulation only, not measured Lahore traffic.
                  </p>
                </div>
              </div>

              {/* Selected Intersection Details Card */}
              {selectedSignal && (
                <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-5 shadow-sm space-y-3">
                  <div className="flex items-center justify-between border-b border-[var(--line)] pb-2.5">
                    <div>
                      <span className="text-[10px] font-extrabold uppercase text-[var(--muted)]">Selected Junction</span>
                      <h3 className="text-sm font-extrabold text-[var(--ink)]">{selectedSignal.name}</h3>
                    </div>
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                      selectedSignal.display === "green" ? "bg-emerald-500/10 text-emerald-500" :
                      selectedSignal.display === "yellow" ? "bg-amber-500/10 text-amber-500" : "bg-rose-500/10 text-rose-500"
                    }`}>
                      Phase {selectedSignal.phase + 1} ({selectedSignal.display.toUpperCase()})
                    </span>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-[var(--muted)]">Phase Elapsed:</span>
                      <span className="font-bold text-[var(--ink)]">{selectedSignal.elapsed.toFixed(1)} s</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-[var(--muted)]">Phase Switches:</span>
                      <span className="font-bold text-[var(--ink)]">{selectedSignal.switches} cycles</span>
                    </div>
                  </div>

                  <div className="space-y-1.5 pt-2 border-t border-[var(--line)]">
                    <span className="text-[11px] font-semibold text-[var(--muted)]">Incoming Approaches:</span>
                    <div className="space-y-1 max-h-36 overflow-y-auto">
                      {selectedSignal.incoming.map((inc) => (
                        <div key={inc.edge_id} className="flex items-center justify-between rounded-md bg-[var(--surface-2)] px-2.5 py-1 text-xs">
                          <span className="truncate max-w-[170px] text-[var(--ink)]" title={inc.name}>
                            {inc.name}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-[var(--muted)]">Q: {inc.queue}</span>
                            <span className={`h-2.5 w-2.5 rounded-full ${
                              inc.state === "green" ? "bg-emerald-500 shadow-sm shadow-emerald-500" :
                              inc.state === "yellow" ? "bg-amber-400 shadow-sm shadow-amber-400" : "bg-rose-500"
                            }`}></span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: BEFORE-AND-AFTER AI COMPARISON */}
      {activeTab === "compare" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-sm space-y-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-base font-extrabold text-[var(--ink)]">
                  Before-and-After Strategy Evaluation
                </h2>
                <p className="text-xs text-[var(--muted)]">
                  Executes twin runs comparing traditional Fixed-Time signal cycles vs AI-Controlled (RL/Max-Pressure) on identical Lahore road network, traffic arrivals, and seeds.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <select
                  value={compareDuration}
                  onChange={(e) => setCompareDuration(parseInt(e.target.value))}
                  className="focus-ring rounded-lg border border-[var(--line)] bg-[var(--surface-2)] px-3 py-2 text-xs font-bold text-[var(--ink)]"
                >
                  <option value={60}>60s Duration</option>
                  <option value={120}>120s Duration</option>
                  <option value={300}>300s Duration</option>
                </select>

                <button
                  type="button"
                  disabled={isComparing}
                  onClick={runComparison}
                  className="focus-ring flex items-center gap-1.5 rounded-lg bg-[var(--brand-fill)] px-4 py-2 text-xs font-extrabold text-[var(--on-brand)] shadow-md transition hover:scale-102 disabled:opacity-50"
                >
                  {isComparing ? (
                    <>
                      <RefreshCw size={14} className="animate-spin" /> Simulating Both Runs...
                    </>
                  ) : (
                    <>
                      <GitCompare size={14} /> Run Comparative Experiment
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Comparison Results Cards */}
            {comparisonReport ? (
              <div className="space-y-6 pt-4 border-t border-[var(--line)]">
                {/* Result Claim Banner */}
                <div className={`rounded-xl p-4 border flex items-center gap-3 ${
                  comparisonReport.comparison.claim === "ai_lower_waiting_time"
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300"
                    : comparisonReport.comparison.claim === "fixed_time_lower_waiting_time"
                    ? "bg-amber-500/10 border-amber-500/30 text-amber-700 dark:text-amber-300"
                    : "bg-blue-500/10 border-blue-500/30 text-blue-700 dark:text-blue-300"
                }`}>
                  <Zap size={20} className="shrink-0" />
                  <div className="text-xs space-y-0.5">
                    <span className="font-extrabold uppercase">
                      Experiment Result: {comparisonReport.comparison.claim.replaceAll("_", " ")}
                    </span>
                    <p className="opacity-90">{comparisonReport.comparison.note}</p>
                  </div>
                </div>

                {/* Comparative Metric Cards */}
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {comparisonReport.comparison.rows
                    .filter((r) => ["average_waiting_time", "average_travel_time", "max_queue_length", "throughput_per_hour"].includes(r.metric))
                    .map((row) => {
                      const isWaitingOrQueue = row.metric.includes("wait") || row.metric.includes("queue");
                      const isImprovement = isWaitingOrQueue
                        ? (row.percent_change || 0) < 0
                        : (row.percent_change || 0) > 0;

                      return (
                        <div key={row.metric} className="rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-4 space-y-2">
                          <span className="text-[11px] font-extrabold uppercase tracking-wide text-[var(--muted)]">
                            {row.metric.replaceAll("_", " ")}
                          </span>

                          <div className="flex items-baseline justify-between">
                            <div>
                              <span className="text-[10px] text-[var(--muted)]">Fixed-Time:</span>
                              <div className="text-base font-extrabold text-[var(--ink)]">{row.fixed_time.toFixed(1)}</div>
                            </div>
                            <div className="text-right">
                              <span className="text-[10px] text-[var(--muted)]">AI Control:</span>
                              <div className="text-base font-extrabold text-blue-600 dark:text-blue-400">{row.ai_controlled.toFixed(1)}</div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between pt-2 border-t border-[var(--line)] text-xs font-bold">
                            <span className="text-[var(--muted)]">Delta:</span>
                            <span className={`flex items-center gap-0.5 ${isImprovement ? "text-emerald-500" : "text-rose-500"}`}>
                              {isImprovement ? <TrendingDown size={14} /> : <TrendingUp size={14} />}
                              {row.percent_change !== null ? `${row.percent_change > 0 ? "+" : ""}${row.percent_change}%` : "N/A"}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                </div>

                {/* Full Comparative Table */}
                <div className="rounded-xl border border-[var(--line)] overflow-hidden bg-[var(--surface)]">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[var(--surface-2)] text-[var(--muted)] font-extrabold uppercase text-[10px] border-b border-[var(--line)]">
                      <tr>
                        <th className="px-4 py-3">Metric Parameter</th>
                        <th className="px-4 py-3">Fixed-Time Baseline</th>
                        <th className="px-4 py-3">AI / RL Treatment</th>
                        <th className="px-4 py-3">Absolute Delta</th>
                        <th className="px-4 py-3">Percentage Change</th>
                        <th className="px-4 py-3">Evidence Kind</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--line)] text-[var(--ink)]">
                      {comparisonReport.comparison.rows.map((row) => (
                        <tr key={row.metric} className="hover:bg-[var(--surface-2)] transition">
                          <td className="px-4 py-2.5 font-bold capitalize">{row.metric.replaceAll("_", " ")}</td>
                          <td className="px-4 py-2.5 font-mono">{row.fixed_time.toFixed(2)}</td>
                          <td className="px-4 py-2.5 font-mono font-bold text-blue-600 dark:text-blue-400">{row.ai_controlled.toFixed(2)}</td>
                          <td className="px-4 py-2.5 font-mono">{row.absolute_change > 0 ? `+${row.absolute_change}` : row.absolute_change}</td>
                          <td className="px-4 py-2.5 font-bold font-mono">
                            {row.percent_change !== null ? (
                              <span className={row.percent_change < 0 ? "text-emerald-500" : "text-rose-500"}>
                                {row.percent_change > 0 ? `+${row.percent_change}%` : `${row.percent_change}%`}
                              </span>
                            ) : "0.0%"}
                          </td>
                          <td className="px-4 py-2.5 text-[10px] text-[var(--muted)]">Simulated micro-steps</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-[var(--line)] p-8 text-center text-xs text-[var(--muted)] space-y-2">
                <GitCompare size={28} className="mx-auto text-[var(--muted)] opacity-60" />
                <p>No comparison run executed yet. Click &quot;Run Comparative Experiment&quot; to test Fixed-Time vs AI control under identical traffic conditions.</p>
              </div>
            )}
          </div>

          {/* Past Experiment History List */}
          <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-sm space-y-4">
            <h3 className="text-sm font-extrabold text-[var(--ink)] flex items-center gap-2">
              <History size={16} /> Experiment Run Log (Database History)
            </h3>
            {experiments.length > 0 ? (
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {experiments.map((exp) => (
                  <div key={exp.id} className="flex flex-col sm:flex-row sm:items-center justify-between rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-3 text-xs gap-2">
                    <div>
                      <div className="font-extrabold text-[var(--ink)] flex items-center gap-2">
                        <span>{exp.notes || exp.kind}</span>
                        <span className="rounded-full bg-blue-500/10 text-blue-600 px-2 py-0.5 text-[10px] font-bold">
                          {exp.controller}
                        </span>
                      </div>
                      <div className="text-[11px] text-[var(--muted)]">
                        Scenario: {exp.scenario} · Duration: {exp.duration}s · Demand: {exp.demand_scale}×
                      </div>
                    </div>
                    <div className="text-right text-[11px] text-[var(--muted)]">
                      <div>Avg Wait: <span className="font-bold text-[var(--ink)]">{exp.metrics.average_waiting_time?.toFixed(1)}s</span></div>
                      <div>Throughput: <span className="font-bold text-[var(--ink)]">{exp.metrics.throughput_per_hour?.toFixed(0)}/hr</span></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-[var(--muted)]">No persisted experiments yet.</p>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: WHAT-IF SCENARIOS */}
      {activeTab === "scenarios" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-sm space-y-4">
            <div>
              <h2 className="text-base font-extrabold text-[var(--ink)]">
                Configurable What-If Traffic Scenarios
              </h2>
              <p className="text-xs text-[var(--muted)]">
                Evaluate urban mobility resilience under simulated disruptions, surges, and lane closures on the Kalma Chowk road network.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {scenarios.map((sc) => {
                const isSelected = scenarioName === sc.id;
                return (
                  <div
                    key={sc.id}
                    className={`rounded-xl border p-4 space-y-3 transition cursor-pointer ${
                      isSelected
                        ? "border-[var(--brand)] bg-[var(--brand-soft)]/20 shadow-md ring-1 ring-[var(--brand)]"
                        : "border-[var(--line)] bg-[var(--surface)] hover:bg-[var(--surface-2)]"
                    }`}
                    onClick={() => updateConfig({ scenario: sc.id })}
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="font-extrabold text-sm text-[var(--ink)]">{sc.label}</h3>
                      <span className="rounded-full bg-[var(--surface-3)] px-2 py-0.5 text-[10px] font-bold text-[var(--muted)]">
                        {sc.demand_scale}× Demand
                      </span>
                    </div>

                    <p className="text-xs text-[var(--muted)] leading-relaxed">
                      {sc.description}
                    </p>

                    <div className="flex flex-wrap gap-1.5 pt-2 border-t border-[var(--line)] text-[10px]">
                      {sc.closed_edges.length > 0 && (
                        <span className="rounded-md bg-rose-500/10 text-rose-600 px-2 py-0.5 font-bold">
                          Closed Road
                        </span>
                      )}
                      {sc.blocked_lane_edges.length > 0 && (
                        <span className="rounded-md bg-amber-500/10 text-amber-600 px-2 py-0.5 font-bold">
                          Blocked Lane
                        </span>
                      )}
                      {sc.incident_edges.length > 0 && (
                        <span className="rounded-md bg-purple-500/10 text-purple-600 px-2 py-0.5 font-bold">
                          Capacity Incident
                        </span>
                      )}
                      {sc.signal_failure.length > 0 && (
                        <span className="rounded-md bg-red-600/10 text-red-700 px-2 py-0.5 font-bold">
                          Signal Caution Failure
                        </span>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        updateConfig({ scenario: sc.id });
                        handleControl("start");
                        setActiveTab("twin");
                      }}
                      className="w-full rounded-lg bg-[var(--surface-2)] hover:bg-[var(--brand-fill)] hover:text-[var(--on-brand)] py-2 text-xs font-bold text-[var(--ink)] transition flex items-center justify-center gap-1.5"
                    >
                      <Play size={13} /> Activate & Simulate
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: COLLISION & CONFLICT ANALYSIS */}
      {activeTab === "risk" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-sm space-y-4">
            <div>
              <h2 className="text-base font-extrabold text-[var(--ink)] flex items-center gap-2">
                <ShieldAlert size={18} className="text-rose-500" />
                Kinematic Accident-Risk & Conflict Analysis
              </h2>
              <p className="text-xs text-[var(--muted)]">
                Continuous trajectory monitoring measuring Time-to-Collision (TTC &lt; 1.5s), following distances, and sudden deceleration events across the simulated vehicle flow.
              </p>
            </div>

            {/* Conflict Stats Bar */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-4">
                <span className="text-[11px] font-bold text-[var(--muted)]">Total Risk Events</span>
                <div className="text-xl font-black text-amber-500 mt-1">{riskEvents.length}</div>
                <span className="text-[10px] text-[var(--muted)]">TTC &lt; 1.5s or decel &le; -4.0 m/s²</span>
              </div>
              <div className="rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-4">
                <span className="text-[11px] font-bold text-[var(--muted)]">Simulated Collisions</span>
                <div className="text-xl font-black text-rose-500 mt-1">{metrics?.simulated_collisions || 0}</div>
                <span className="text-[10px] text-[var(--muted)]">Overlap gap &le; 0.4m in car-following</span>
              </div>
              <div className="rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-4">
                <span className="text-[11px] font-bold text-[var(--muted)]">Threshold Limitations</span>
                <p className="text-[10px] text-[var(--muted)] mt-1">
                  Model-based kinematic indicators, not field crash reports. Short headways in stop-and-go queues are filtered to avoid false collisions.
                </p>
              </div>
            </div>

            {/* Risk Events Table */}
            <div className="rounded-xl border border-[var(--line)] overflow-hidden bg-[var(--surface)]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[var(--surface-2)] text-[var(--muted)] font-extrabold uppercase text-[10px] border-b border-[var(--line)]">
                  <tr>
                    <th className="px-4 py-3">Event Kind</th>
                    <th className="px-4 py-3">Vehicle ID</th>
                    <th className="px-4 py-3">TTC (s)</th>
                    <th className="px-4 py-3">Following Gap</th>
                    <th className="px-4 py-3">Relative Speed</th>
                    <th className="px-4 py-3">Sim Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)] text-[var(--ink)]">
                  {riskEvents.slice(-15).reverse().map((ev, i) => (
                    <tr key={i} className="hover:bg-[var(--surface-2)] transition">
                      <td className="px-4 py-2.5 font-bold">
                        <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-extrabold ${
                          ev.kind === "simulated_collision"
                            ? "bg-rose-500/10 text-rose-600"
                            : "bg-amber-500/10 text-amber-600"
                        }`}>
                          <AlertTriangle size={11} />
                          {ev.kind.replaceAll("_", " ").toUpperCase()}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 font-mono text-[var(--muted)]">{ev.vehicle_id || "v-sim"}</td>
                      <td className="px-4 py-2.5 font-mono font-bold">{ev.ttc !== null ? `${ev.ttc}s` : "N/A"}</td>
                      <td className="px-4 py-2.5 font-mono">{ev.gap} m</td>
                      <td className="px-4 py-2.5 font-mono">{ev.relative_speed} m/s</td>
                      <td className="px-4 py-2.5 font-mono text-[var(--muted)]">{ev.time ?? "0"}s</td>
                    </tr>
                  ))}
                  {riskEvents.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-6 text-center text-xs text-[var(--muted)]">
                        No conflict risk events recorded yet in current simulation run.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: RL TRAINING HUB */}
      {activeTab === "training" && (
        <div className="space-y-6">
          <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-6 shadow-sm space-y-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-base font-extrabold text-[var(--ink)] flex items-center gap-2">
                  <BrainCircuit size={18} className="text-blue-500" />
                  Multi-Agent Deep Reinforcement Learning
                </h2>
                <p className="text-xs text-[var(--muted)]">
                  Trains intersection signal agents using Deep Q-Networks (DQN). State vector encodes approach queue lengths, waiting times, current phase elapsed, and neighboring intersection pressure.
                </p>
              </div>

              <button
                type="button"
                disabled={isTraining || trainingStatus?.status === "running"}
                onClick={startRLTraining}
                className="focus-ring flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-xs font-extrabold text-white shadow-md transition hover:bg-blue-700 disabled:opacity-50"
              >
                {isTraining || trainingStatus?.status === "running" ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" /> Training In Progress...
                  </>
                ) : (
                  <>
                    <Zap size={14} /> Train RL Controller (5 Episodes)
                  </>
                )}
              </button>
            </div>

            {/* Training Status Card */}
            <div className="rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[var(--ink)]">
                  Training Status:{" "}
                  <span className="uppercase text-blue-600 font-extrabold">
                    {trainingStatus?.status || "Idle"}
                  </span>
                </span>
                <span className="text-xs font-bold text-[var(--muted)]">
                  {trainingStatus?.progress || 0}% Complete
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-[var(--line)] h-2.5 rounded-full overflow-hidden">
                <div
                  className="bg-blue-600 h-full transition-all duration-300"
                  style={{ width: `${trainingStatus?.progress || 0}%` }}
                ></div>
              </div>

              <div className="flex justify-between text-[11px] text-[var(--muted)]">
                <span>Episodes: {trainingStatus?.current_episode || 0} / {trainingStatus?.episodes || 5}</span>
                <span>Reward Function: -0.35ΔWait - 0.25Queue + 1.8Throughput - 0.6Switch</span>
              </div>
            </div>

            {/* Episode Rewards Log */}
            {trainingStatus?.rewards && trainingStatus.rewards.length > 0 && (
              <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 space-y-2">
                <span className="text-xs font-extrabold text-[var(--ink)]">Episode Cumulative Rewards</span>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                  {trainingStatus.rewards.map((r, i) => (
                    <div key={i} className="rounded-lg border border-[var(--line)] bg-[var(--surface-2)] p-2.5 text-center text-xs">
                      <div className="text-[10px] text-[var(--muted)]">Episode {i + 1}</div>
                      <div className="font-extrabold text-blue-600 mt-0.5">{r}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
