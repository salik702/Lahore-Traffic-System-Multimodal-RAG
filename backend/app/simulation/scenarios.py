"""Named what-if configurations. These modify the simulation model, not real-world observations."""

SCENARIOS = {
    "normal": {
        "id": "normal",
        "label": "Normal traffic",
        "description": "Daytime demand mix for the imported Kalma Chowk network.",
        "demand_scale": 1.0,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "peak": {
        "id": "peak",
        "label": "Peak-hour traffic",
        "description": "Higher arrival rates approximating a peak commute. Not calibrated to a counted Lahore peak.",
        "demand_scale": 1.8,
        "mix": {"car": 0.50, "motorcycle": 0.38, "bus": 0.08, "other": 0.04},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "high_traffic": {
        "id": "high_traffic",
        "label": "High traffic",
        "description": "Dense synthetic traffic for visual stress testing (3× normal arrivals); not calibrated to measured Lahore counts.",
        "demand_scale": 3.0,
        "mix": {"car": 0.50, "motorcycle": 0.43, "bus": 0.05, "other": 0.02},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "motorcycle_heavy": {
        "id": "motorcycle_heavy",
        "label": "Increased motorcycle traffic",
        "description": "Same total demand with a motorcycle-majority mix.",
        "demand_scale": 1.15,
        "mix": {"car": 0.28, "motorcycle": 0.62, "bus": 0.05, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "closed_road": {
        "id": "closed_road",
        "label": "Closed road",
        "description": "Closes a major approach so vehicles must reroute. Modelled as a capacity removal, not a real closure.",
        "demand_scale": 1.0,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": ["auto_major"],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "blocked_lane": {
        "id": "blocked_lane",
        "label": "Blocked lane",
        "description": "Reduces lane count on a major approach.",
        "demand_scale": 1.0,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": ["auto_major"],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "incident": {
        "id": "incident",
        "label": "Capacity-reducing incident",
        "description": "Slows a corridor to model an incident. This is not a recorded collision.",
        "demand_scale": 1.1,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": ["auto_major"],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "demand_increase": {
        "id": "demand_increase",
        "label": "Increased traffic demand",
        "description": "2.2× baseline arrivals with the normal mix.",
        "demand_scale": 2.2,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.0,
    },
    "signal_failure": {
        "id": "signal_failure",
        "label": "Traffic signal failure",
        "description": "One controllable junction stays in all-red / caution and discharges slowly.",
        "demand_scale": 1.0,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": ["auto_signal"],
        "phase_scale": 1.0,
    },
    "longer_green": {
        "id": "longer_green",
        "label": "Longer fixed-time greens",
        "description": "Scales fixed-time phase duration by 1.5×.",
        "demand_scale": 1.0,
        "mix": {"car": 0.55, "motorcycle": 0.32, "bus": 0.08, "other": 0.05},
        "closed_edges": [],
        "blocked_lane_edges": [],
        "incident_edges": [],
        "signal_failure": [],
        "phase_scale": 1.5,
    },
}

VEHICLE_TYPES = {
    "car": {"length": 4.5, "max_speed": 13.9, "accel": 2.0, "decel": 4.5, "gap": 2.0},
    "motorcycle": {"length": 2.2, "max_speed": 16.7, "accel": 2.6, "decel": 5.0, "gap": 1.2},
    "bus": {"length": 12.0, "max_speed": 11.1, "accel": 1.1, "decel": 3.5, "gap": 3.0},
    "other": {"length": 3.2, "max_speed": 12.0, "accel": 1.8, "decel": 4.0, "gap": 1.6},
}

# A visible research/demo flow across the imported network. This is synthetic
# demand and must not be interpreted as a measured Lahore traffic count.
BASE_SPAWN_PER_HOUR = 7200


def resolve_scenario(name: str, network: dict, overrides: dict | None = None) -> dict:
    base = dict(SCENARIOS.get(name, SCENARIOS["normal"]))
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                base[key] = value
    major = pick_major_edge(network)
    signal = pick_signal(network)
    closed = []
    for item in base["closed_edges"]:
        closed.append(major if item == "auto_major" and major else item)
    blocked = []
    for item in base["blocked_lane_edges"]:
        blocked.append(major if item == "auto_major" and major else item)
    incidents = []
    for item in base["incident_edges"]:
        incidents.append(major if item == "auto_major" and major else item)
    failures = []
    for item in base["signal_failure"]:
        failures.append(signal if item == "auto_signal" and signal else item)
    base["closed_edges"] = [item for item in closed if item]
    base["blocked_lane_edges"] = [item for item in blocked if item]
    base["incident_edges"] = [item for item in incidents if item]
    base["signal_failure"] = [item for item in failures if item]
    return base


def pick_major_edge(network: dict) -> str | None:
    edges = [e for e in network["edges"].values() if not network.get("synthetic")]
    if not edges:
        edges = list(network["edges"].values())
    named = [e for e in edges if any(token in e["name"].lower() for token in ("feroz", "فیروز", "liberty", "kalma", "boulevard", "mall"))]
    pool = named or edges
    if not pool:
        return None
    return max(pool, key=lambda item: item["length"] * item["lanes"])["id"]


def pick_signal(network: dict) -> str | None:
    controllable = [j for j in network["junctions"].values() if j.get("controllable")]
    if not controllable:
        return None
    named = [j for j in controllable if "kalma" in j["name"].lower() or "کلمہ" in j["name"]]
    return (named or controllable)[0]["id"]
