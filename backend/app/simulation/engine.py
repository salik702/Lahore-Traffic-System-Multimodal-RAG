"""Time-step traffic engine on the imported OSM graph (SUMO-compatible metrics)."""
from __future__ import annotations

import heapq
import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from app.simulation.controllers import ControllerBank, SignalAgent
from app.simulation.metrics import congestion_level, empty_metrics
from app.simulation.osm_network import point_along
from app.simulation.risk import following_risk
from app.simulation.rl import DQNPolicy, OBS_SIZE, observation_vector
from app.simulation.scenarios import BASE_SPAWN_PER_HOUR, VEHICLE_TYPES, resolve_scenario


@dataclass
class Vehicle:
    id: str
    kind: str
    route: list[str]
    index: int = 0
    pos: float = 0.0
    speed: float = 0.0
    wait: float = 0.0
    travel: float = 0.0
    accel: float = 0.0


@dataclass
class Engine:
    network: dict
    seed: int = 7
    controller_mode: str = "fixed_time"
    scenario_name: str = "normal"
    demand_scale: float = 1.0
    dt: float = 0.5
    policy_path: str | None = None
    rng: random.Random = field(init=False)
    edges: dict = field(init=False)
    junction_of: dict = field(init=False)
    adjacency: dict = field(init=False)
    vehicles: dict[str, Vehicle] = field(default_factory=dict)
    on_edge: dict[str, list[str]] = field(default_factory=dict)
    edge_entries: dict[str, int] = field(default_factory=dict)
    edge_entry_times: dict[str, deque] = field(default_factory=dict)
    bank: ControllerBank = field(init=False)
    time: float = 0.0
    completed: int = 0
    stranded: int = 0
    spawned: int = 0
    wait_sum: float = 0.0
    travel_sum: float = 0.0
    reward_sum: float = 0.0
    last_completed: int = 0
    last_wait: float = 0.0
    risk_events: list[dict] = field(default_factory=list)
    collisions: int = 0
    spawn_accum: float = 0.0
    scenario: dict = field(init=False)
    closed: set[str] = field(default_factory=set)
    lane_factor: dict = field(default_factory=dict)
    speed_factor: dict = field(default_factory=dict)
    routes: list[list[str]] = field(default_factory=list)
    _vid: int = 0

    def __post_init__(self) -> None:
        self.reset(self.seed, self.controller_mode, self.scenario_name, self.demand_scale)

    def reset(self, seed: int | None = None, mode: str | None = None, scenario: str | None = None, demand_scale: float | None = None) -> None:
        self.seed = self.seed if seed is None else seed
        self.controller_mode = mode or self.controller_mode
        self.scenario_name = scenario or self.scenario_name
        if demand_scale is not None:
            self.demand_scale = demand_scale
        self.rng = random.Random(self.seed)
        self.edges = deepcopy(self.network["edges"])
        self.time = 0.0
        self.completed = 0
        self.stranded = 0
        self.spawned = 0
        self.wait_sum = 0.0
        self.travel_sum = 0.0
        self.reward_sum = 0.0
        self.last_completed = 0
        self.last_wait = 0.0
        self.risk_events = []
        self.collisions = 0
        self.spawn_accum = 0.0
        self.vehicles = {}
        self.on_edge = {eid: [] for eid in self.edges}
        self.edge_entries = {eid: 0 for eid in self.edges}
        self.edge_entry_times = {eid: deque() for eid in self.edges}
        self._vid = 0
        self.scenario = resolve_scenario(self.scenario_name, self.network, {"demand_scale": self.demand_scale})
        self.demand_scale = float(self.scenario["demand_scale"])
        self.closed = set(self.scenario["closed_edges"])
        self.lane_factor = {eid: 0.5 for eid in self.scenario["blocked_lane_edges"]}
        self.speed_factor = {eid: 0.35 for eid in self.scenario["incident_edges"]}
        self.junction_of = {}
        agents = {}
        for jid, junction in self.network["junctions"].items():
            for edge_id in junction["incoming"]:
                self.junction_of[edge_id] = jid
            if junction.get("controllable") and junction.get("phases"):
                agent = SignalAgent(jid, deepcopy(junction["phases"]))
                agent.failed = jid in self.scenario["signal_failure"]
                agents[jid] = agent
        policy = None
        if self.controller_mode == "rl":
            policy = DQNPolicy.load(self.policy_path)
        effective_mode = self.controller_mode
        if self.controller_mode == "rl" and policy is None:
            effective_mode = "max_pressure"
        self.bank = ControllerBank(effective_mode if self.controller_mode != "rl" else ("rl" if policy else "max_pressure"), agents, policy)
        self.adjacency = {eid: [] for eid in self.edges}
        incoming_index = {}
        for edge in self.edges.values():
            incoming_index.setdefault(edge["from"], [])
            incoming_index.setdefault(edge["to"], [])
        for edge in self.edges.values():
            incoming_index.setdefault(edge["from"], [])
        by_from = {}
        for edge in self.edges.values():
            by_from.setdefault(edge["from"], []).append(edge["id"])
        for eid, edge in self.edges.items():
            self.adjacency[eid] = [other for other in by_from.get(edge["to"], []) if other != eid]
        self.routes = self._build_routes()

    def _build_routes(self, limit: int = 80) -> list[list[str]]:
        sources = [eid for eid, edge in self.edges.items() if eid not in self.closed and not any(other["to"] == edge["from"] for other in self.edges.values() if other["id"] not in self.closed) or sum(1 for other in self.edges.values() if other["to"] == edge["from"]) <= 1]
        if len(sources) < 8:
            sources = [eid for eid in self.edges if eid not in self.closed]
        sinks = [eid for eid, edge in self.edges.items() if eid not in self.closed and sum(1 for other in self.edges.values() if other["from"] == edge["to"]) <= 1]
        if len(sinks) < 8:
            sinks = [eid for eid in self.edges if eid not in self.closed]
        routes = []
        attempts = 0
        while len(routes) < limit and attempts < limit * 12:
            attempts += 1
            start = self.rng.choice(sources)
            goal_edge = self.rng.choice(sinks)
            if start == goal_edge:
                continue
            path = self._shortest(start, self.edges[goal_edge]["to"])
            if path and len(path) >= 2 and not any(item in self.closed for item in path):
                routes.append(path)
        if not routes:
            open_edges = [eid for eid in self.edges if eid not in self.closed]
            for eid in open_edges[:limit]:
                nxt = self.adjacency.get(eid) or []
                routes.append([eid] + nxt[:1])
        return routes or [[eid] for eid in self.edges if eid not in self.closed]

    def _shortest(self, start_edge: str, goal_node: str) -> list[str] | None:
        heap = [(0.0, start_edge, [start_edge])]
        best = {start_edge: 0.0}
        while heap:
            cost, eid, path = heapq.heappop(heap)
            edge = self.edges[eid]
            if edge["to"] == goal_node:
                return path
            if cost > best.get(eid, 1e18):
                continue
            for nxt in self.adjacency[eid]:
                if nxt in self.closed:
                    continue
                ncost = cost + self.edges[nxt]["length"]
                if ncost < best.get(nxt, 1e18):
                    best[nxt] = ncost
                    heapq.heappush(heap, (ncost, nxt, path + [nxt]))
        return None

    def _spawn_rate(self) -> float:
        return BASE_SPAWN_PER_HOUR * self.demand_scale / 3600.0

    def _choose_type(self) -> str:
        mix = self.scenario["mix"]
        draw = self.rng.random()
        total = 0.0
        for kind, share in mix.items():
            total += share
            if draw <= total:
                return kind
        return "car"

    def _spawn(self) -> None:
        if not self.routes:
            return
        route = self.rng.choice(self.routes)
        kind = self._choose_type()
        edge_id = route[0]
        if edge_id in self.closed:
            return
        occupants = self.on_edge[edge_id]
        if occupants:
            last = self.vehicles[occupants[-1]]
            if last.pos < VEHICLE_TYPES[kind]["length"] + 4:
                return
        self._vid += 1
        vid = f"v{self._vid}"
        vehicle = Vehicle(id=vid, kind=kind, route=route)
        self.vehicles[vid] = vehicle
        self.on_edge[edge_id].append(vid)
        self._record_edge_entry(edge_id)
        self.spawned += 1

    def _record_edge_entry(self, edge_id: str) -> None:
        self.edge_entries[edge_id] = self.edge_entries.get(edge_id, 0) + 1
        self.edge_entry_times.setdefault(edge_id, deque()).append(self.time)

    def edge_traffic(self) -> dict[str, dict[str, int]]:
        result = {}
        cutoff = self.time - 60.0
        for edge_id, vehicle_ids in self.on_edge.items():
            entries = self.edge_entry_times[edge_id]
            while entries and entries[0] < cutoff:
                entries.popleft()
            result[edge_id] = {
                "vehicles": len(vehicle_ids),
                "waiting": sum(1 for vid in vehicle_ids if self.vehicles[vid].speed < 0.6),
                "mean_speed": round(sum(self.vehicles[vid].speed for vid in vehicle_ids) / max(1, len(vehicle_ids)), 2),
                "entries_last_60s": len(entries),
                "entries_total": self.edge_entries[edge_id],
            }
        return result

    def queues(self) -> dict[str, int]:
        result = {}
        for eid, ids in self.on_edge.items():
            length = self.edges[eid]["length"]
            result[eid] = sum(1 for vid in ids if self.vehicles[vid].speed < 1.0 and self.vehicles[vid].pos > length - 40)
        return result

    def _neighbor_pressure(self) -> dict[str, float]:
        queues = self.queues()
        pressure = {}
        for jid, agent in self.bank.agents.items():
            others = []
            for other_id, other in self.bank.agents.items():
                if other_id == jid:
                    continue
                others.append(sum(queues.get(eid, 0) for eid in other.phases[other.phase]["green"]))
            pressure[jid] = sum(others) / max(1, len(others))
        return pressure

    def observations(self) -> dict[str, list[float]]:
        queues = self.queues()
        neighbor = self._neighbor_pressure()
        obs = {}
        for jid, agent in self.bank.agents.items():
            incoming = self.network["junctions"][jid]["incoming"]
            q = [queues.get(eid, 0) for eid in incoming[:4]]
            q += [0] * (4 - len(q))
            waiting = []
            for eid in incoming[:4]:
                waiting.append(sum(self.vehicles[vid].wait for vid in self.on_edge[eid] if self.vehicles[vid].speed < 1.0) / max(1, len(self.on_edge[eid])))
            waiting += [0.0] * (4 - len(waiting))
            occupancy = min(1.0, sum(len(self.on_edge[eid]) for eid in incoming) / 40.0)
            obs[jid] = observation_vector(q, waiting, occupancy, agent.phase, agent.elapsed, neighbor.get(jid, 0.0), len(agent.phases))
        return obs

    def step(self) -> float:
        dt = self.dt
        self.spawn_accum += self._spawn_rate() * dt
        while self.spawn_accum >= 1:
            self._spawn()
            self.spawn_accum -= 1
        queues = self.queues()
        obs = self.observations()
        neighbor = self._neighbor_pressure()
        prev_wait = sum(v.wait for v in self.vehicles.values())
        prev_completed = self.completed
        prev_switches = sum(a.switches for a in self.bank.agents.values())
        prev_collisions = self.collisions
        self.bank.step(dt, queues, obs, neighbor, self.scenario["phase_scale"])
        green = set()
        for agent in self.bank.agents.values():
            green |= agent.current_green()
        failed = {jid for jid, agent in self.bank.agents.items() if agent.failed}

        for eid, ids in list(self.on_edge.items()):
            ordered = sorted(ids, key=lambda vid: self.vehicles[vid].pos, reverse=True)
            self.on_edge[eid] = ordered
            edge = self.edges[eid]
            length = edge["length"]
            lanes = max(1.0, edge["lanes"] * self.lane_factor.get(eid, 1.0))
            speed_cap = edge["speed"] * self.speed_factor.get(eid, 1.0)
            leader_pos = length + 80
            leader_speed = speed_cap
            at_stopline = eid in self.junction_of and eid not in green
            if self.junction_of.get(eid) in failed:
                at_stopline = True
            for rank, vid in enumerate(ordered):
                veh = self.vehicles[vid]
                spec = VEHICLE_TYPES[veh.kind]
                gap = leader_pos - veh.pos - spec["length"]
                desired = min(spec["max_speed"], speed_cap)
                if at_stopline:
                    stop_gap = length - 2.0 - veh.pos
                    if stop_gap < 18:
                        desired = min(desired, max(0.0, stop_gap / 2.5))
                if gap < spec["gap"] + 8:
                    desired = min(desired, max(0.0, leader_speed * 0.9, gap * 0.6))
                accel = spec["accel"] if desired > veh.speed else -spec["decel"]
                if gap < spec["gap"] + 1.5:
                    accel = -spec["decel"]
                veh.accel = max(-spec["decel"], min(spec["accel"], accel))
                veh.speed = max(0.0, min(desired, veh.speed + veh.accel * dt))
                if lanes <= 1 and rank > 8:
                    veh.speed *= 0.85
                veh.pos += veh.speed * dt
                veh.travel += dt
                if veh.speed < 0.6:
                    veh.wait += dt
                event = following_risk(leader_speed, veh.speed, max(0.05, gap), veh.accel)
                if event:
                    lat, lon, _ = point_along(edge, veh.pos)
                    event.update({"time": round(self.time, 1), "vehicle_id": vid, "edge_id": eid, "lat": lat, "lon": lon, "junction_id": self.junction_of.get(eid)})
                    self.risk_events.append(event)
                    if event["kind"] == "simulated_collision":
                        self.collisions += 1
                        veh.speed = 0.0
                        veh.pos = min(veh.pos, leader_pos - spec["length"] - 0.5)
                    if len(self.risk_events) > 400:
                        self.risk_events = self.risk_events[-400:]
                leader_pos = veh.pos
                leader_speed = veh.speed

        transferred = []
        for eid, ids in list(self.on_edge.items()):
            keep = []
            for vid in ids:
                veh = self.vehicles[vid]
                if veh.pos < self.edges[eid]["length"]:
                    keep.append(vid)
                    continue
                overflow = veh.pos - self.edges[eid]["length"]
                if veh.index + 1 >= len(veh.route):
                    self.completed += 1
                    self.wait_sum += veh.wait
                    self.travel_sum += veh.travel
                    del self.vehicles[vid]
                    continue
                nxt = veh.route[veh.index + 1]
                if nxt in self.closed:
                    alternatives = [item for item in self.adjacency[eid] if item not in self.closed]
                    if not alternatives:
                        self.stranded += 1
                        del self.vehicles[vid]
                        continue
                    nxt = alternatives[0]
                    veh.route = veh.route[: veh.index + 1] + [nxt]
                wait_for_green = eid in self.junction_of and eid not in green
                if wait_for_green and overflow < 4:
                    veh.pos = self.edges[eid]["length"] - 0.4
                    veh.speed = 0.0
                    keep.append(vid)
                    continue
                veh.index += 1
                veh.pos = min(overflow, 3.0)
                transferred.append((vid, nxt))
            self.on_edge[eid] = keep
        for vid, nxt in transferred:
            self.on_edge[nxt].append(vid)
            self._record_edge_entry(nxt)

        self.time += dt
        wait_now = sum(v.wait for v in self.vehicles.values())
        switched = sum(a.switches for a in self.bank.agents.values()) - prev_switches
        throughput = self.completed - prev_completed
        queue_sum = sum(self.queues().values())
        reward = (
            -0.35 * max(0.0, wait_now - prev_wait)
            - 0.25 * queue_sum
            + 1.8 * throughput
            - 0.6 * switched
            - 2.0 * (self.collisions - prev_collisions)
        )
        self.reward_sum += reward
        self.last_completed = throughput
        return reward

    def run(self, duration: float) -> dict:
        steps = int(duration / self.dt)
        for _ in range(steps):
            self.step()
        return self.metrics()

    def metrics(self) -> dict:
        queues = list(self.queues().values())
        waiting = sum(1 for v in self.vehicles.values() if v.speed < 0.6)
        avg_wait = (self.wait_sum / self.completed) if self.completed else (sum(v.wait for v in self.vehicles.values()) / max(1, len(self.vehicles)))
        avg_travel = (self.travel_sum / self.completed) if self.completed else 0.0
        occupancy = min(1.0, len(self.vehicles) / max(12, 0.25 * len(self.edges)))
        mean_speed = sum(v.speed for v in self.vehicles.values()) / max(1, len(self.vehicles))
        avg_q = sum(queues) / max(1, len(queues))
        payload = empty_metrics()
        payload.update({
            "sim_time": round(self.time, 1),
            "average_waiting_time": round(avg_wait, 2),
            "average_travel_time": round(avg_travel, 2),
            "average_queue_length": round(avg_q, 2),
            "max_queue_length": max(queues) if queues else 0,
            "total_vehicles": len(self.vehicles),
            "vehicles_completed": self.completed,
            "vehicles_stranded": self.stranded,
            "vehicles_waiting": waiting,
            "throughput_per_hour": round(self.completed / max(self.time, 1.0) * 3600.0, 1),
            "congestion_level": congestion_level(avg_q, occupancy, avg_wait),
            "active_signals": sum(1 for a in self.bank.agents.values() if not a.failed),
            "risk_events": len(self.risk_events),
            "simulated_collisions": self.collisions,
            "mean_speed": round(mean_speed, 2),
            "occupancy": round(occupancy, 3),
            "reward": round(self.reward_sum, 2),
            "spawned": self.spawned,
        })
        return payload

    def snapshot(self, max_vehicles: int = 280) -> dict:
        vehicles = []
        for vid, veh in list(self.vehicles.items())[:max_vehicles]:
            if veh.index >= len(veh.route):
                continue
            edge = self.edges[veh.route[veh.index]]
            lat, lon, heading = point_along(edge, veh.pos)
            vehicles.append({
                "id": vid, "type": veh.kind, "lat": lat, "lon": lon, "heading": heading,
                "speed": round(veh.speed, 2), "edge_id": edge["id"], "waiting": veh.speed < 0.6,
            })
        signals = []
        queues = self.queues()
        for jid, agent in self.bank.agents.items():
            junction = self.network["junctions"][jid]
            incoming = []
            for eid in junction["incoming"]:
                incoming.append({
                    "edge_id": eid,
                    "queue": queues.get(eid, 0),
                    "state": "green" if eid in agent.current_green() else ("yellow" if agent.in_yellow else "red"),
                    "name": self.edges[eid]["name"],
                })
            signals.append({
                "id": jid, "name": junction["name"], "lat": junction["lat"], "lon": junction["lon"],
                "phase": agent.phase, "elapsed": round(agent.elapsed, 1), "display": agent.display(),
                "failed": agent.failed, "incoming": incoming, "switches": agent.switches,
            })
        risk_hotspots = self.risk_events[-40:]
        return {
            "time": round(self.time, 1),
            "metrics": self.metrics(),
            "edge_traffic": self.edge_traffic(),
            "vehicles": vehicles,
            "signals": signals,
            "risk_events": risk_hotspots,
            "closed_edges": list(self.closed),
        }

    def network_view(self) -> dict:
        edges = []
        for edge in self.edges.values():
            edges.append({
                "id": edge["id"], "name": edge["name"], "highway": edge["highway"],
                "lanes": edge["lanes"], "lane_source": edge.get("lane_source", "estimated from road class"),
                "length": edge["length"], "speed": edge["speed"],
                "speed_source": edge.get("speed_source", "estimated from road class"),
                "from": edge["from"], "to": edge["to"],
                "closed": edge["id"] in self.closed,
                "polyline": [{"lat": p["lat"], "lon": p["lon"]} for p in edge["polyline"]],
            })
        junctions = [{
            "id": j["id"], "name": j["name"], "lat": j["lat"], "lon": j["lon"],
            "controllable": j.get("controllable", False),
        } for j in self.network["junctions"].values()]
        return {
            "area": self.network["area"],
            "synthetic": self.network.get("synthetic", False),
            "source": self.network["source"],
            "license": self.network["license"],
            "bounds": self.network["bounds"],
            "stats": self.network["stats"],
            "edges": edges,
            "junctions": junctions,
        }
