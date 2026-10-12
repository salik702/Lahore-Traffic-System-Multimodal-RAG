"""Before/after comparison of equivalent simulation runs."""
from __future__ import annotations

from app.simulation.metrics import percentage_change

COMPARE_KEYS = [
    "average_waiting_time",
    "average_travel_time",
    "max_queue_length",
    "average_queue_length",
    "vehicles_completed",
    "throughput_per_hour",
]


def compare_metrics(baseline: dict, treatment: dict) -> dict:
    rows = []
    for key in COMPARE_KEYS:
        before = float(baseline.get(key) or 0)
        after = float(treatment.get(key) or 0)
        rows.append({
            "metric": key,
            "fixed_time": before,
            "ai_controlled": after,
            "absolute_change": round(after - before, 3),
            "percent_change": percentage_change(after, before),
            "kind": "measured_simulation",
        })
    wait_delta = percentage_change(float(treatment.get("average_waiting_time") or 0), float(baseline.get("average_waiting_time") or 0))
    claim = "insufficient_evidence"
    if wait_delta is not None:
        if wait_delta < -3:
            claim = "ai_lower_waiting_time"
        elif wait_delta > 3:
            claim = "fixed_time_lower_waiting_time"
        else:
            claim = "no_meaningful_difference"
    return {
        "rows": rows,
        "claim": claim,
        "note": "Both runs used the same network, demand, duration and random seed family. Results are simulated, not field measurements.",
    }
