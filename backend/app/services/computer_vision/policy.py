"""Conservative defaults; these thresholds have NOT been statistically calibrated.

Override through VISION_POLICY_PATH after evaluating a held-out, labeled dataset.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

DEFAULTS = {
    "class_thresholds": {"car": .50, "motorcycle": .40, "person": .40,
                         "bus": .40, "truck": .40, "bicycle": .40,
                         "stop sign": .45, "traffic light": .45},
    "helmet_positive": .60, "helmet_negative": .75,
    "ocr_threshold": .30, "vehicle_image_size": 960,
    "max_motorcycle_crops": 8, "max_plate_crops": 12,
    "max_pose_refinements": 4, "max_head_refinements": 4,
}


@lru_cache(maxsize=1)
def policy():
    result = {**DEFAULTS, "class_thresholds": dict(DEFAULTS["class_thresholds"])}
    filename = os.getenv("VISION_POLICY_PATH")
    if filename:
        custom = json.loads(Path(filename).read_text(encoding="utf-8"))
        unknown = set(custom) - set(DEFAULTS)
        if unknown:
            raise RuntimeError(f"Unknown vision policy fields: {sorted(unknown)}")
        for key, value in custom.items():
            if key == "class_thresholds":
                if set(value) - set(result[key]):
                    raise RuntimeError("Unknown object classes in vision policy")
                result[key].update(value)
            else:
                result[key] = value
    probabilities = [*result["class_thresholds"].values(), result["helmet_positive"],
                     result["helmet_negative"], result["ocr_threshold"]]
    if any(not isinstance(v, (float, int)) or not .01 <= v <= 1 for v in probabilities):
        raise RuntimeError("Confidence thresholds must be within [0.01, 1]")
    for key, limit in (("max_motorcycle_crops", 32), ("max_plate_crops", 32), ("vehicle_image_size", 1280),
                       ("max_pose_refinements", 8), ("max_head_refinements", 8)):
        if not isinstance(result[key], int) or not 1 <= result[key] <= limit:
            raise RuntimeError(f"Invalid {key}")
    return result


def capabilities():
    return [
        {"id": "helmet_violation", "name": "Motorcycle helmet", "status": "implemented_unvalidated",
         "reason": "Requires an explicit bare-head prediction, clear head region, unique motorcycle association and seated pose."},
        {"id": "motorcycle_overloading", "name": "Motorcycle passenger limit", "status": "implemented_unvalidated",
         "reason": "Requires at least three distinct people with supported seated associations; Punjab Section 89 permits driver plus one passenger."},
        {"id": "plate_ocr", "name": "Visible number plate OCR", "status": "implemented_unvalidated",
         "reason": "Latin letters and digits only; two OCR views must agree. No Pakistan benchmark or registration validation."},
        *[{"id": kind, "name": name, "status": "unsupported", "reason": reason}
          for kind, name, reason in [
            ("seatbelt_violation", "Driver seatbelt", "An evaluated occupant/seatbelt detector, driver role and notified-road context are unavailable."),
            ("passenger_seatbelt", "Front passenger seatbelt", "Passenger position, applicable duty and an evaluated restraint detector are unavailable."),
            ("mobile_phone", "Hand-held phone use", "A phone box cannot establish driver identity, hand use or driving."),
            ("missing_plate", "Missing number plate", "No model establishes an adequately observed, empty plate mounting area."),
            ("obscured_plate", "Obscured number plate", "Unreadable text does not prove obstruction; a labeled occlusion detector is required."),
            ("nonstandard_plate", "Nonstandard plate", "Appearance/OCR cannot verify official security features or plate issuance."),
            ("commercial_loading", "Unsafe passengers or loading", "An evaluated loading detector and legal vehicle limits are unavailable."),
            ("red_light", "Red-light / stop-line crossing", "Requires tracked crossing, the controlling signal phase and calibrated stop line."),
            ("wrong_way", "Wrong-way driving", "Requires established road direction and vehicle movement evidence."),
            ("illegal_parking", "Illegal parking", "Requires location, restrictions and sufficient duration evidence."),
            ("lane_violation", "Lane / pedestrian crossing", "Requires road markings, permitted maneuvers and crossing evidence."),
            ("illegal_uturn", "Illegal U-turn", "Requires trajectory and applicable road restrictions."),
            ("dangerous_driving", "Dangerous driving", "Requires temporal behavior and scene context."),
            ("speeding", "Speeding", "Requires calibrated speed measurement or trusted external telemetry."),
            ("license_registration", "License / registration status", "Requires authoritative records; cannot be determined visually."),
        ]],
    ]
