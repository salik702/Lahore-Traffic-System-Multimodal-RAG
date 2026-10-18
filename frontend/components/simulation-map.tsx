"use client";

import { useEffect, useRef, useState } from "react";
import type { SimulationNetwork, SimulationRiskEvent, SimulationSignal, SimulationVehicle } from "@/lib/models";
import { AlertTriangle, Car, Eye, Layers, LocateFixed, Maximize2, ShieldAlert } from "lucide-react";

interface SimulationMapProps {
  network: SimulationNetwork | null;
  vehicles: SimulationVehicle[];
  signals: SimulationSignal[];
  riskEvents: SimulationRiskEvent[];
  edgeTraffic: Record<string, { vehicles: number; waiting: number; mean_speed: number; entries_last_60s: number; entries_total: number }>;
  closedEdges?: string[];
  selectedSignalId?: string | null;
  selectedRoadId?: string | null;
  onSelectSignal?: (signalId: string) => void;
  onSelectRoad?: (roadId: string) => void;
  congestionLevel?: string;
}

export function SimulationMap({
  network,
  vehicles,
  signals,
  riskEvents,
  edgeTraffic,
  closedEdges = [],
  selectedSignalId,
  selectedRoadId,
  onSelectSignal,
  onSelectRoad,
  congestionLevel = "low",
}: SimulationMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const lastFocusedRoadRef = useRef<string | null>(null);
  const roadLayersRef = useRef<Array<{ id: string; casing: any; line: any; edge: SimulationNetwork["edges"][number]; isClosed: boolean }>>([]);
  const roadStyleKeyRef = useRef<Map<string, string>>(new Map());
  const vehicleMarkersRef = useRef<Map<string, any>>(new Map());
  const vehicleAnimationRef = useRef<Map<string, number>>(new Map());
  const vehicleTrailsRef = useRef<Map<string, { layer: any; points: [number, number][] }>>(new Map());
  const lastFollowAtRef = useRef(0);
  const signalMarkersRef = useRef<Map<string, any>>(new Map());
  const riskMarkersRef = useRef<any[]>([]);

  const [showVehicles, setShowVehicles] = useState(true);
  const [showSignals, setShowSignals] = useState(true);
  const [showRisks, setShowRisks] = useState(true);
  const [showCongestion, setShowCongestion] = useState(true);
  const [showTrails, setShowTrails] = useState(false);
  const [followFleet, setFollowFleet] = useState(false);
  const [motionSpeed, setMotionSpeed] = useState(1);
  const [vehicleType, setVehicleType] = useState<"all" | SimulationVehicle["type"]>("all");
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);
  const closedEdgeKey = closedEdges.join("|");

  // Initialize Leaflet map
  useEffect(() => {
    let isMounted = true;
    if (!containerRef.current) return;

    // Dynamically load leaflet on client
    import("leaflet").then((L) => {
      if (!isMounted || !containerRef.current) return;
      if (mapInstanceRef.current) return;

      const defaultCenter: [number, number] = network?.bounds
        ? [(network.bounds.min_lat + network.bounds.max_lat) / 2, (network.bounds.min_lon + network.bounds.max_lon) / 2]
        : [31.5035, 74.3315];

      const map = L.map(containerRef.current, {
        center: defaultCenter,
        zoom: 16,
        minZoom: 14,
        maxZoom: 19,
        zoomControl: true,
        attributionControl: true,
      });

      const cartoKey = process.env.NEXT_PUBLIC_CARTO_API_KEY?.trim();
      const cartoTileUrl = `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png${cartoKey ? `?key=${encodeURIComponent(cartoKey)}` : ""}`;
      L.tileLayer(cartoTileUrl, {
        maxZoom: 19,
        subdomains: "abcd",
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      }).addTo(map);

      mapInstanceRef.current = map;
    });

    return () => {
      isMounted = false;
      if (mapInstanceRef.current) {
        vehicleAnimationRef.current.forEach((frame) => cancelAnimationFrame(frame));
        vehicleAnimationRef.current.clear();
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Draw road network polylines
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !network) return;

    import("leaflet").then((L) => {
      if (mapInstanceRef.current !== map || !map.getPane("markerPane")) return;
      // Clear previous road layers
      roadLayersRef.current.forEach(({ casing, line }) => {
        casing.remove();
        line.remove();
      });
      roadLayersRef.current = [];
      roadStyleKeyRef.current.clear();

      network.edges.forEach((edge) => {
        const isClosed = closedEdges.includes(edge.id) || edge.closed;
        const latlngs: [number, number][] = edge.polyline.map((p) => [p.lat, p.lon]);
        const surface = edge.highway === "motorway" || edge.highway === "trunk" ? "#7988a8" : edge.highway === "primary" ? "#8795a6" : "#9aa5b3";

        const casing = L.polyline(latlngs, {
          color: "#263344",
          weight: Math.max(6, edge.lanes * 4.2),
          opacity: isClosed ? 0.6 : 0.92,
          dashArray: isClosed ? "8, 8" : undefined,
          lineCap: "round",
          lineJoin: "round",
        }).addTo(map);
        const line = L.polyline(latlngs, {
          color: isClosed ? "#dc2626" : surface,
          weight: Math.max(3, edge.lanes * 2.5),
          opacity: isClosed ? 0.7 : 0.95,
          dashArray: isClosed ? "8, 8" : undefined,
          lineCap: "round",
          lineJoin: "round",
        }).addTo(map);
        const onRoadClick = () => onSelectRoad?.(edge.id);
        casing.on("click", onRoadClick);
        line.on("click", onRoadClick);
        const tooltip = `<b>${edge.name}</b><br/>${edge.lanes} lane${edge.lanes === 1 ? "" : "s"} · ${Math.round(edge.length)} m · ${Math.round(edge.speed * 3.6)} km/h${isClosed ? " · CLOSED" : ""}`;
        casing.bindTooltip(tooltip, {
          sticky: true,
          className: "custom-leaflet-tooltip",
        });
        line.bindTooltip(tooltip, { sticky: true, className: "custom-leaflet-tooltip" });
        roadLayersRef.current.push({ id: edge.id, casing, line, edge, isClosed });
      });
    });
  }, [network, closedEdgeKey, onSelectRoad]);

  // Recolor only the road strokes as live edge occupancy changes; keep geometry and click targets stable.
  useEffect(() => {
    for (const { id, casing, line, edge, isClosed } of roadLayersRef.current) {
      const traffic = edgeTraffic[id];
      const storageEstimate = Math.max(1, Math.floor((edge.length / 7) * edge.lanes));
      const load = traffic ? traffic.vehicles / storageEstimate : 0;
      let surface = edge.highway === "motorway" || edge.highway === "trunk" ? "#7988a8" : edge.highway === "primary" ? "#8795a6" : "#9aa5b3";
      if (showCongestion && load >= 0.8) surface = "#ef4444";
      else if (showCongestion && (load >= 0.45 || (traffic?.waiting ?? 0) > 0)) surface = "#f59e0b";
      if (isClosed) surface = "#dc2626";
      const selected = selectedRoadId === id;
      const styleKey = `${surface}|${selected}|${isClosed}`;
      if (roadStyleKeyRef.current.get(id) === styleKey) continue;
      roadStyleKeyRef.current.set(id, styleKey);
      casing.setStyle({ color: selected ? "#22d3ee" : "#263344", weight: Math.max(6, edge.lanes * 4.2) + (selected ? 3 : 0) });
      line.setStyle({ color: selected ? "#b6f5ff" : surface, weight: Math.max(3, edge.lanes * 2.5) });
    }
  }, [edgeTraffic, selectedRoadId, showCongestion]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    const road = network?.edges.find((edge) => edge.id === selectedRoadId);
    if (!map || !road || lastFocusedRoadRef.current === road.id) return;
    lastFocusedRoadRef.current = road.id;
    import("leaflet").then((L) => {
      if (mapInstanceRef.current !== map || !map.getPane("markerPane")) return;
      const bounds = L.latLngBounds(road.polyline.map((point) => [point.lat, point.lon] as [number, number]));
      map.fitBounds(bounds, { padding: [72, 72], maxZoom: 18, animate: true });
    });
  }, [network, selectedRoadId]);

  // Update Signals
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    import("leaflet").then((L) => {
      if (mapInstanceRef.current !== map || !map.getPane("markerPane")) return;
      // If signals turned off, remove all
      if (!showSignals) {
        signalMarkersRef.current.forEach((m) => m.remove());
        signalMarkersRef.current.clear();
        return;
      }

      signals.forEach((signal) => {
        const isSelected = selectedSignalId === signal.id;
        let colorClass = "bg-emerald-500 shadow-emerald-500/50";
        let ringColor = "border-emerald-400";
        if (signal.display === "yellow") {
          colorClass = "bg-amber-400 shadow-amber-400/50";
          ringColor = "border-amber-300";
        } else if (signal.display === "failed") {
          colorClass = "bg-rose-600 shadow-rose-600/50 animate-pulse";
          ringColor = "border-rose-500";
        }

        const iconHtml = `
          <div class="relative flex items-center justify-center cursor-pointer transition-transform hover:scale-125 ${isSelected ? "scale-125" : ""}">
            <div class="absolute -inset-1 rounded-full border-2 ${ringColor} opacity-75 animate-ping"></div>
            <div class="relative h-6 w-6 rounded-full border-2 border-white dark:border-zinc-900 ${colorClass} shadow-lg flex items-center justify-center text-[9px] font-black text-white">
              ${signal.phase + 1}
            </div>
          </div>
        `;

        const icon = L.divIcon({
          html: iconHtml,
          className: "custom-signal-div-icon",
          iconSize: [24, 24],
          iconAnchor: [12, 12],
        });

        const existing = signalMarkersRef.current.get(signal.id);
        if (existing) {
          existing.setIcon(icon);
          existing.setLatLng([signal.lat, signal.lon]);
        } else {
          const marker = L.marker([signal.lat, signal.lon], { icon }).addTo(map);
          marker.on("click", () => {
            if (onSelectSignal) onSelectSignal(signal.id);
          });
          marker.bindTooltip(`<b>${signal.name}</b><br/>Phase: ${signal.phase + 1} (${signal.display})<br/>Elapsed: ${signal.elapsed}s`, {
            sticky: true,
          });
          signalMarkersRef.current.set(signal.id, marker);
        }
      });
    });
  }, [signals, showSignals, selectedSignalId, onSelectSignal]);

  // Update Moving Vehicles
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    import("leaflet").then((L) => {
      if (mapInstanceRef.current !== map || !map.getPane("markerPane")) return;
      if (!showVehicles) {
        vehicleMarkersRef.current.forEach((m) => m.remove());
        vehicleMarkersRef.current.clear();
        vehicleTrailsRef.current.forEach(({ layer }) => layer.remove());
        vehicleTrailsRef.current.clear();
        vehicleAnimationRef.current.forEach((frame) => cancelAnimationFrame(frame));
        vehicleAnimationRef.current.clear();
        return;
      }

      const visibleVehicles = vehicleType === "all" ? vehicles : vehicles.filter((vehicle) => vehicle.type === vehicleType);
      const activeIds = new Set(visibleVehicles.map((v) => v.id));

      // Remove stale vehicle markers
      vehicleMarkersRef.current.forEach((marker, id) => {
        if (!activeIds.has(id)) {
          marker.remove();
          vehicleMarkersRef.current.delete(id);
          const frame = vehicleAnimationRef.current.get(id);
          if (frame) cancelAnimationFrame(frame);
          vehicleAnimationRef.current.delete(id);
          const trail = vehicleTrailsRef.current.get(id);
          trail?.layer.remove();
          vehicleTrailsRef.current.delete(id);
        }
      });

      // Update or add vehicle markers
      visibleVehicles.forEach((veh) => {
        const fill = veh.type === "motorcycle" ? "#34d399" : veh.type === "bus" ? "#f59e0b" : veh.type === "other" ? "#a78bfa" : "#38bdf8";
        const body = veh.type === "motorcycle"
          ? `<svg viewBox="0 0 20 36" width="15" height="27" aria-hidden="true"><circle cx="10" cy="5" r="3" fill="#17202b"/><circle cx="10" cy="31" r="3" fill="#17202b"/><path d="M10 4 6 12l2 5-2 7 4 7 4-7-2-7 2-5z" fill="${fill}" stroke="#e2e8f0" stroke-width="1.4"/><path d="m8 12 4 0m-4 11 4 0" stroke="#ecfeff" stroke-width="2"/></svg>`
          : veh.type === "bus"
          ? `<svg viewBox="0 0 26 46" width="19" height="34" aria-hidden="true"><path d="M5 2h16a3 3 0 0 1 3 3v36a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V5a3 3 0 0 1 3-3" fill="${fill}" stroke="#f8fafc" stroke-width="1.5"/><rect x="5" y="7" width="16" height="11" rx="2" fill="#dff6ff"/><rect x="5" y="21" width="16" height="12" rx="2" fill="#dff6ff"/><path d="M5 38h3m10 0h3" stroke="#17202b" stroke-width="3"/></svg>`
          : `<svg viewBox="0 0 26 42" width="17" height="29" aria-hidden="true"><path d="M9 2h8l4 5 2 8v12l-3 9H6l-3-9V15l2-8z" fill="${fill}" stroke="#f8fafc" stroke-width="1.5"/><path d="M8 8h10l2 7H6zm-1 11h12v7H7z" fill="#dff6ff"/><path d="M5 12H3m20 0h-2M5 31h3m10 0h3" stroke="#17202b" stroke-width="2.4"/><path d="M8 34h10" stroke="#fef3c7" stroke-width="2"/></svg>`;
        const waitPing = veh.waiting ? '<span data-waiting class="absolute -right-1 -top-1 h-2 w-2 rounded-full border border-white bg-rose-500"></span>' : "";
        const iconHtml = `
          <div class="relative flex items-center justify-center transition-transform duration-300 ease-linear" style="transform:rotate(${veh.heading}deg)">
            ${body}
            ${waitPing}
            ${veh.waiting ? `
              <div class="absolute -inset-2 rounded-full bg-rose-500/20 animate-ping"></div>
            ` : ""}
          </div>
        `;

        const icon = L.divIcon({
          html: iconHtml,
          className: "custom-vehicle-div-icon",
          iconSize: [24, 38],
          iconAnchor: [12, 19],
        });

        const existing = vehicleMarkersRef.current.get(veh.id);
        if (existing) {
          const markerNode = existing.getElement?.();
          const bodyNode = markerNode?.querySelector(".relative") as HTMLElement | null;
          if (bodyNode) bodyNode.style.transform = `rotate(${veh.heading}deg)`;
          const waitingNode = markerNode?.querySelector("[data-waiting]");
          if (veh.waiting && !waitingNode) existing.setIcon(icon);
          if (!veh.waiting && waitingNode) existing.setIcon(icon);
          const start = existing.getLatLng();
          const target: [number, number] = [veh.lat, veh.lon];
          const priorFrame = vehicleAnimationRef.current.get(veh.id);
          if (priorFrame) cancelAnimationFrame(priorFrame);
          const distance = map.distance(start, target);
          if (distance < 0.8) {
            existing.setLatLng(target);
          } else {
            const startedAt = performance.now();
            const duration = Math.min(900, Math.max(100, distance / Math.max(veh.speed, 1) * 1000 / motionSpeed));
            const animatePosition = (now: number) => {
              const progress = Math.min(1, (now - startedAt) / duration);
              const eased = progress * progress * (3 - 2 * progress);
              existing.setLatLng([
                start.lat + (target[0] - start.lat) * eased,
                start.lng + (target[1] - start.lng) * eased,
              ]);
              if (progress < 1) vehicleAnimationRef.current.set(veh.id, requestAnimationFrame(animatePosition));
              else vehicleAnimationRef.current.delete(veh.id);
            };
            vehicleAnimationRef.current.set(veh.id, requestAnimationFrame(animatePosition));
          }
        } else {
          const marker = L.marker([veh.lat, veh.lon], { icon }).addTo(map);
          marker.bindTooltip(`<b>${veh.id}</b> (${veh.type.toUpperCase()})<br/>Speed: ${(veh.speed * 3.6).toFixed(1)} km/h<br/>Status: ${veh.waiting ? "Queued / Waiting" : "Moving"}`, {
            sticky: true,
          });
          vehicleMarkersRef.current.set(veh.id, marker);
        }

        if (showTrails) {
          const trailState = vehicleTrailsRef.current.get(veh.id);
          const point: [number, number] = [veh.lat, veh.lon];
          if (trailState) {
            const last = trailState.points[trailState.points.length - 1];
            if (!last || map.distance(last, point) > 1) trailState.points.push(point);
            if (trailState.points.length > 8) trailState.points.shift();
            trailState.layer.setLatLngs(trailState.points);
          } else {
            const trail = L.polyline([point], { color: fill, weight: 2, opacity: 0.62, interactive: false }).addTo(map);
            vehicleTrailsRef.current.set(veh.id, { layer: trail, points: [point] });
          }
        }
      });

      if (!showTrails) {
        vehicleTrailsRef.current.forEach(({ layer }) => layer.remove());
        vehicleTrailsRef.current.clear();
      }
    });
  }, [vehicles, showVehicles, showTrails, motionSpeed, vehicleType]);

  // Pan toward the fleet centroid at most once per 1.2 seconds to keep camera motion smooth.
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!followFleet || !map || !showVehicles || vehicles.length === 0) return;
    const now = performance.now();
    if (now - lastFollowAtRef.current < 1200) return;
    lastFollowAtRef.current = now;
    const center: [number, number] = [
      vehicles.reduce((sum, vehicle) => sum + vehicle.lat, 0) / vehicles.length,
      vehicles.reduce((sum, vehicle) => sum + vehicle.lon, 0) / vehicles.length,
    ];
    if (map.distance(map.getCenter(), center) > 45) map.panTo(center, { animate: true, duration: 0.8 });
  }, [vehicles, followFleet, showVehicles]);

  // Update Risk Event Markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    import("leaflet").then((L) => {
      if (mapInstanceRef.current !== map || !map.getPane("markerPane")) return;
      riskMarkersRef.current.forEach((m) => m.remove());
      riskMarkersRef.current = [];

      if (!showRisks) return;

      const recentRisks = riskEvents.slice(-20);
      recentRisks.forEach((event, idx) => {
        if (!event.lat || !event.lon) return;

        const isCollision = event.kind === "simulated_collision";
        const iconHtml = `
          <div class="relative flex items-center justify-center animate-pulse">
            <div class="absolute h-8 w-8 rounded-full ${isCollision ? "bg-red-600/40" : "bg-amber-500/30"} animate-ping"></div>
            <div class="h-5 w-5 rounded-full ${isCollision ? "bg-red-600 border-red-300" : "bg-amber-500 border-amber-200"} border-2 flex items-center justify-center shadow-lg text-white font-bold text-[9px]">
              !
            </div>
          </div>
        `;

        const icon = L.divIcon({
          html: iconHtml,
          className: "custom-risk-icon",
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        });

        const marker = L.marker([event.lat, event.lon], { icon }).addTo(map);
        marker.bindTooltip(`<b>${isCollision ? "SIMULATED COLLISION" : "NEAR-MISS RISK EVENT"}</b><br/>TTC: ${event.ttc ?? "N/A"}s · Gap: ${event.gap}m<br/>Rel Speed: ${event.relative_speed} m/s`, {
          sticky: true,
        });
        riskMarkersRef.current.push(marker);
      });
    });
  }, [riskEvents, showRisks]);

  function resetView() {
    if (!mapInstanceRef.current) return;
    const center: [number, number] = network?.bounds
      ? [(network.bounds.min_lat + network.bounds.max_lat) / 2, (network.bounds.min_lon + network.bounds.max_lon) / 2]
      : [31.5035, 74.3315];
    mapInstanceRef.current.setView(center, 16, { animate: true });
  }

  return (
    <div className="relative w-full h-full min-h-[520px] rounded-xl overflow-hidden border border-[var(--line)] bg-[var(--surface-2)] shadow-xl">
      {/* The Leaflet DOM container */}
      <div ref={containerRef} className="w-full h-full min-h-[520px] z-0" />

      {/* Map Header Overlay */}
      <div className="absolute top-3 left-3 z-10 flex flex-wrap items-center gap-2 rounded-lg bg-[var(--surface)]/90 backdrop-blur-md px-3 py-2 border border-[var(--line)] shadow-lg text-xs font-semibold">
        <span className="flex items-center gap-1.5 text-blue-600 dark:text-blue-400 font-bold">
          <Car size={15} />
          {network?.area || "Kalma Chowk Road Network"}
        </span>
        <span className="text-[var(--line)]">|</span>
        <span className="text-[var(--muted)]">
          {vehicles.length} Active Vehicles
        </span>
        <span className="text-[var(--line)]">|</span>
        <span className="text-[var(--muted)]">
          {signals.length} Signals
        </span>
        {riskEvents.length > 0 && (
          <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full font-bold">
            <AlertTriangle size={12} />
            {riskEvents.length} Conflicts
          </span>
        )}
      </div>

      {/* Layer Toggles & View Controls Overlay */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5 rounded-lg bg-[var(--surface)]/90 backdrop-blur-md p-1.5 border border-[var(--line)] shadow-lg">
        <button
          type="button"
          onClick={() => setShowVehicles((v) => !v)}
          className={`focus-ring rounded-md px-2 py-1 text-[11px] font-bold transition flex items-center gap-1 ${
            showVehicles
              ? "bg-blue-600 text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--ink)]"
          }`}
          aria-pressed={showVehicles}
          title={showVehicles ? "Hide simulated vehicles" : "Show simulated vehicles"}
        >
          <Car size={12} />
          {showVehicles ? "Hide cars & bikes" : "Show cars & bikes"}
        </button>

        <button
          type="button"
          onClick={() => setShowSignals((s) => !s)}
          className={`focus-ring rounded-md px-2 py-1 text-[11px] font-bold transition flex items-center gap-1 ${
            showSignals
              ? "bg-emerald-600 text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--ink)]"
          }`}
          title="Toggle Traffic Signals"
        >
          <span className="h-2 w-2 rounded-full bg-emerald-300"></span>
          Signals
        </button>

        <button
          type="button"
          onClick={() => setShowRisks((r) => !r)}
          className={`focus-ring rounded-md px-2 py-1 text-[11px] font-bold transition flex items-center gap-1 ${
            showRisks
              ? "bg-amber-600 text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--ink)]"
          }`}
          title="Toggle Conflict & Near-Miss Indicators"
        >
          <ShieldAlert size={12} />
          Risks
        </button>

        <button
          type="button"
          onClick={() => setShowCongestion((c) => !c)}
          className={`focus-ring rounded-md px-2 py-1 text-[11px] font-bold transition flex items-center gap-1 ${
            showCongestion
              ? "bg-purple-600 text-white shadow-sm"
              : "text-[var(--muted)] hover:text-[var(--ink)]"
          }`}
          title="Toggle Congestion Density"
        >
          <Layers size={12} />
          Flow
        </button>

        <button
          type="button"
          onClick={() => setShowTrails((trails) => !trails)}
          className={`focus-ring rounded-md px-2 py-1 text-[11px] font-bold transition flex items-center gap-1 ${
            showTrails ? "bg-cyan-600 text-white shadow-sm" : "text-[var(--muted)] hover:text-[var(--ink)]"
          }`}
          aria-pressed={showTrails}
          title="Show recent vehicle movement trails"
        >
          <Eye size={12} /> Trails
        </button>

        <button
          type="button"
          onClick={resetView}
          className="focus-ring p-1 rounded-md text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--surface-2)] transition"
          title="Recenter Map View"
        >
          <Maximize2 size={14} />
        </button>
      </div>

      {/* Map Legend Overlay */}
      <div className="absolute bottom-3 left-3 z-10 flex flex-wrap items-center gap-3 rounded-lg bg-[var(--surface)]/90 backdrop-blur-md px-3 py-1.5 border border-[var(--line)] shadow-md text-[10px] text-[var(--muted)]">
        <span className="font-bold text-[var(--ink)]">Legend:</span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-blue-500"></span> Car
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-emerald-400"></span> Motorcycle
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-amber-500"></span> Bus
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-emerald-500"></span> Signal Green
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-amber-400"></span> Yellow
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse"></span> Conflict
        </span>
        <span className="flex items-center gap-1 border-l border-[var(--line)] pl-2">
          <span className="h-2 w-2 rounded-full bg-amber-500"></span> Busy road
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-cyan-300"></span> Selected road
        </span>
      </div>

      <div className="absolute bottom-14 right-3 z-10 flex flex-wrap items-center justify-end gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface)]/90 p-2 shadow-lg backdrop-blur-md sm:bottom-3">
        <label className="flex items-center gap-1.5 text-[10px] font-bold text-[var(--muted)]">
          Vehicles
          <select value={vehicleType} onChange={(event) => setVehicleType(event.target.value as typeof vehicleType)} className="rounded-md border border-[var(--line)] bg-[var(--surface)] px-1.5 py-1 text-[10px] text-[var(--ink)]">
            <option value="all">All</option>
            <option value="car">Cars</option>
            <option value="motorcycle">Motorcycles</option>
            <option value="bus">Buses</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[10px] font-bold text-[var(--muted)]">
          Motion
          <select value={motionSpeed} onChange={(event) => setMotionSpeed(Number(event.target.value))} className="rounded-md border border-[var(--line)] bg-[var(--surface)] px-1.5 py-1 text-[10px] text-[var(--ink)]">
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={1.5}>1.5×</option>
            <option value={2}>2×</option>
          </select>
        </label>
        <button type="button" onClick={() => setFollowFleet((following) => !following)} aria-pressed={followFleet} className={`focus-ring flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-bold transition ${followFleet ? "bg-cyan-600 text-white" : "text-[var(--muted)] hover:bg-[var(--surface-2)] hover:text-[var(--ink)]"}`} title="Keep the camera centered on the vehicle fleet">
          <LocateFixed size={12} /> {followFleet ? "Following" : "Follow fleet"}
        </button>
      </div>
    </div>
  );
}
