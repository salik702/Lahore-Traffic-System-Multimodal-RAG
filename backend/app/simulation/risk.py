"""Experimental near-miss indicators from simulated trajectories.

These events are model-based risk flags, not real-world accidents and not
proof of a collision unless closing speed produced an overlap in the model.
"""
from __future__ import annotations

TTC_THRESHOLD = 1.5
FOLLOW_GAP_THRESHOLD = 6.0
BRAKE_THRESHOLD = -4.0


def following_risk(leader_speed: float, follower_speed: float, gap: float, follower_decel: float) -> dict | None:
    if gap <= 0:
        return {"kind": "simulated_collision", "ttc": 0.0, "gap": gap, "relative_speed": follower_speed - leader_speed}
    relative = follower_speed - leader_speed
    ttc = gap / relative if relative > 0.15 else None
    harsh = follower_decel <= BRAKE_THRESHOLD
    near_miss = (ttc is not None and ttc < TTC_THRESHOLD and gap < FOLLOW_GAP_THRESHOLD) or (gap < 2.5 and relative > 1.0)
    if not near_miss and not harsh:
        return None
    kind = "simulated_collision" if gap <= 0.4 else "near_miss" if near_miss else "harsh_braking"
    return {
        "kind": kind,
        "ttc": None if ttc is None else round(ttc, 2),
        "gap": round(gap, 2),
        "relative_speed": round(relative, 2),
        "limitation": "Experimental kinematic indicator on simulated trajectories.",
    }
