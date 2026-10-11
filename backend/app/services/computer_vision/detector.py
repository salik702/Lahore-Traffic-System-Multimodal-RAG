"""Cached local YOLO inference; provision weights with scripts/setup_vision.py."""
from pathlib import Path
from threading import RLock
import os
from .policy import policy
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
VEHICLES = {"motorcycle", "car", "bus", "truck", "bicycle"}
INFERENCE_LOCK = RLock()
_models = {}

def get_model(kind):
    from ultralytics import YOLO
    filenames = {"vehicle": "vehicle_detector.pt", "helmet": "helmet_detector.pt", "pose": "rider_pose.pt"}
    with INFERENCE_LOCK:
        if kind not in _models:
            path = Path(os.getenv(f"{kind.upper()}_MODEL_PATH", str(MODEL_DIR / filenames[kind])))
            if not path.is_file():
                raise RuntimeError(f"{kind.title()} model unavailable. Run scripts/setup_vision.py.")
            try:
                model = YOLO(path)
                labels = {str(v).lower().replace("_", " ").replace("-", " ") for v in model.names.values()}
                if kind == "vehicle" and not VEHICLES.issubset(labels):
                    raise ValueError("Expected COCO vehicle classes absent")
                if kind == "helmet" and not {"with helmet", "without helmet"}.issubset(labels):
                    raise ValueError("Expected motorcycle helmet classes absent")
                _models[kind] = model
            except Exception as exc:
                raise RuntimeError(f"{kind.title()} model could not be initialized") from exc
        return _models[kind]

def predict(kind, image, confidence=.35, size=640):
    import torch
    with INFERENCE_LOCK:
        model = get_model(kind)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        result = model.predict(image, conf=confidence, imgsz=size, iou=.45,
                               device=device, verbose=False, max_det=150)[0]
        boxes = [{"id": f"{kind}_{i+1}", "class": str(result.names[int(box.cls[0])]).lower().replace('_',' ').replace('-',' '),
                  "confidence": round(float(box.conf[0]),4),
                  "bbox": [round(float(v),1) for v in box.xyxy[0].tolist()]}
                 for i,box in enumerate(result.boxes)]
        return boxes,result

def detect_objects(image, confidence=.3):
    cfg = policy()
    boxes, _ = predict("vehicle", image, min(confidence, min(cfg["class_thresholds"].values())), cfg["vehicle_image_size"])
    selected = resolve_vehicle_classes([b for b in boxes if
        b["confidence"] >= cfg["class_thresholds"].get(b["class"], 1.1)])
    return selected, "Ultralytics YOLO11s" if not os.getenv("VEHICLE_MODEL_PATH") else "Ultralytics configured vehicle model"


def resolve_vehicle_classes(boxes):
    """Remove duplicates; withhold vehicle-specific rules on competing labels."""
    from .rider_association import iou
    selected = []
    for box in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
        if box["class"] not in VEHICLES:
            if not any(b["class"] == box["class"] and iou(b["bbox"], box["bbox"]) > .65 for b in selected):
                selected.append(box)
            continue
        duplicate = next((b for b in selected if b["class"] in VEHICLES and iou(b["bbox"], box["bbox"]) > .65), None)
        if duplicate:
            if duplicate["class"] != box["class"]:
                duplicate["classification_status"] = "uncertain"
                duplicate.setdefault("competing_classes", []).append({"class": box["class"], "confidence": box["confidence"]})
            continue
        selected.append({**box, "classification_status": "supported", "competing_classes": []})
    return selected
