export type Section = "home" | "digital-twin" | "assistant" | "fines" | "signs" | "safety" | "visual" | "accidents" | "sources";
export type Fine = { code: number; violation: string; keywords: string[]; motorcycle: number | null; three_wheeler: number | null; motorcar_under_2000cc: number | null; motorcar_over_2000cc: number | null; transport_vehicle: number | null };
export type Rule = { id: string; title: string; category: string; summary: string; section: string; page: number; source_id: string; region: string };
export type Sign = { id: string; name: string; category: string; meaning: string; image_url: string | null; source_id: string; source_page: number; image_source_id?: string; image_source_page?: number | null; image_source_url?: string | null };
export type Source = { id: string; title: string; organization: string; original_url: string; retrieval_date: string; data_type: string; region: string; geographic_scope: string; official: boolean; license: string; description: string };
export type AccidentImage = { id: string; image_url: string | null; title: string; description: string; date: string | null; city: string | null; province: string | null; source: string; source_url: string; license: string; license_url?: string; author?: string; verified_record_match: boolean };
export type AccidentRecord = { id: string; title: string; description: string; date: string | null; time: string | null; location: string | null; city: string | null; district: string | null; province: string | null; country: string; accident_type: string | null; vehicle_types: string[]; fatalities: number | null; injuries: number | null; severity: string | null; source_name: string; source_url: string; image_url: string | null; image_verified_for_record: boolean; images: AccidentImage[] };
export type Stats = { documents: number; images: number; traffic_signs: number; fine_records: number; rules: number; accident_records: number; accident_images: number; verified_accident_image_pairs: number; latest_indexed_accident: string | null; sources: number; official_sources: number; regions: string[] };
export type RecentSearch = { query: string; at: number };

// Digital Twin simulation models
export type SimulationMetrics = {
  sim_time: number;
  average_waiting_time: number;
  average_travel_time: number;
  average_queue_length: number;
  max_queue_length: number;
  total_vehicles: number;
  vehicles_completed: number;
  vehicles_stranded: number;
  vehicles_waiting: number;
  throughput_per_hour: number;
  congestion_level: "low" | "moderate" | "high" | "severe";
  active_signals: number;
  risk_events: number;
  simulated_collisions: number;
  mean_speed: number;
  occupancy: number;
  reward?: number;
  spawned?: number;
};

export type SimulationVehicle = {
  id: string;
  type: "car" | "motorcycle" | "bus" | "other";
  lat: number;
  lon: number;
  heading: number;
  speed: number;
  edge_id: string;
  waiting: boolean;
};

export type SimulationSignal = {
  id: string;
  name: string;
  lat: number;
  lon: number;
  phase: number;
  elapsed: number;
  display: "green" | "yellow" | "failed";
  failed: boolean;
  incoming: { edge_id: string; queue: number; state: "green" | "yellow" | "red"; name: string }[];
  switches: number;
};

export type SimulationRiskEvent = {
  kind: "near_miss" | "harsh_braking" | "simulated_collision";
  ttc?: number | null;
  gap: number;
  relative_speed: number;
  time?: number;
  vehicle_id?: string;
  edge_id?: string;
  lat?: number;
  lon?: number;
  junction_id?: string;
  limitation?: string;
};

export type SimulationSnapshot = {
  time: number;
  running: boolean;
  speed: number;
  controller_mode: "fixed_time" | "max_pressure" | "rl";
  scenario: string;
  demand_scale: number;
  engine_type: string;
  sumo_detected: boolean;
  metrics: SimulationMetrics;
  edge_traffic: Record<string, { vehicles: number; waiting: number; mean_speed: number; entries_last_60s: number; entries_total: number }>;
  vehicles: SimulationVehicle[];
  signals: SimulationSignal[];
  risk_events: SimulationRiskEvent[];
  closed_edges: string[];
};

export type SimulationNetwork = {
  area: string;
  synthetic: boolean;
  source: string;
  license: string;
  bounds: { min_lat: number; max_lat: number; min_lon: number; max_lon: number };
  stats: { nodes: number; edges: number; junctions: number; signals: number };
  edges: { id: string; name: string; highway: string; lanes: number; lane_source: string; length: number; speed: number; speed_source: string; from: string; to: string; closed: boolean; polyline: { lat: number; lon: number }[] }[];
  junctions: { id: string; name: string; lat: number; lon: number; controllable: boolean }[];
};

export type SimulationScenario = {
  id: string;
  label: string;
  description: string;
  demand_scale: number;
  mix: Record<string, number>;
  closed_edges: string[];
  blocked_lane_edges: string[];
  incident_edges: string[];
  signal_failure: string[];
  phase_scale: number;
};

export type ComparisonMetricRow = {
  metric: string;
  fixed_time: number;
  ai_controlled: number;
  absolute_change: number;
  percent_change: number | null;
  kind: string;
};

export type ComparisonReport = {
  experiment_id: string;
  baseline_metrics: SimulationMetrics;
  treatment_metrics: SimulationMetrics;
  comparison: {
    rows: ComparisonMetricRow[];
    claim: string;
    note: string;
  };
  config: {
    scenario: string;
    baseline_controller: string;
    treatment_controller: string;
    demand_scale: number;
    duration: number;
    seed: number;
  };
};

export type ExperimentRecord = {
  id: string;
  kind: string;
  controller: string;
  scenario: string;
  seed: number;
  demand_scale: number;
  duration: number;
  engine: string;
  metrics: SimulationMetrics;
  comparison?: { rows: ComparisonMetricRow[]; claim: string; note: string } | null;
  notes: string;
  created_at: string;
  provenance: string;
};

export type TrainingStatus = {
  status: "idle" | "running" | "completed" | "failed";
  episodes: number;
  current_episode: number;
  progress: number;
  rewards: number[];
  error?: string | null;
};
