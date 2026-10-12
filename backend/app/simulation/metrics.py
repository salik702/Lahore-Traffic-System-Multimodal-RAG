"""Pure metric helpers used by the engine and comparison views."""
from __future__ import annotations


def percentage_change(after: float, before: float) -> float | None:
    if before == 0:
        return None if after == 0 else None
    return round(100.0 * (after - before) / before, 2)


def congestion_level(avg_queue: float, occupancy: float, wait: float) -> str:
    score = 0.45 * min(avg_queue / 12.0, 1.5) + 0.35 * min(occupancy, 1.2) + 0.20 * min(wait / 60.0, 1.5)
    if score < 0.25:
        return "low"
    if score < 0.55:
        return "moderate"
    if score < 0.85:
        return "high"
    return "severe"


def empty_metrics() -> dict:
    return {
        "sim_time": 0.0,
        "average_waiting_time": 0.0,
        "average_travel_time": 0.0,
        "average_queue_length": 0.0,
        "max_queue_length": 0,
        "total_vehicles": 0,
        "vehicles_completed": 0,
        "vehicles_stranded": 0,
        "vehicles_waiting": 0,
        "throughput_per_hour": 0.0,
        "congestion_level": "low",
        "active_signals": 0,
        "risk_events": 0,
        "simulated_collisions": 0,
        "mean_speed": 0.0,
        "occupancy": 0.0,
        "reward": 0.0,
    }


def summarize_run(metrics: dict, extra: dict | None = None) -> dict:
    payload = {**metrics}
    if extra:
        payload.update(extra)
    return payload
