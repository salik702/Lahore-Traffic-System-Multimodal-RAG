"""Provision pinned public models, verify hashes/classes, and preserve official fine snapshots.

Run from the repository root: uv run --project backend python scripts/setup_vision.py
This is a setup utility, not an automated test suite. It never downloads on an API request.
"""
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "backend" / "models"
ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
HELMET_COMMIT = "2b58c02c91395daa9add86fdad311dc37fd39eed"
MODEL_SPECS = [
    {"filename":"vehicle_detector.pt","name":"Ultralytics YOLO11s COCO","source":ASSETS+"yolo11s.pt","version":"v8.3.0 assets",
     "license":"AGPL-3.0; enterprise licensing available from Ultralytics","sha256":"85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5","classes":["person","bicycle","car","motorcycle","bus","truck"],"limitations":"COCO general detector, not calibrated for Punjab surveillance"},
    {"filename":"helmet_detector.pt","name":"AutonomousUI motorcycle helmet YOLOv8n","source":f"https://raw.githubusercontent.com/AutonomousUI/traffic-helmet-detection/{HELMET_COMMIT}/weights/best.pt","version":HELMET_COMMIT,
     "license":"Publisher repository MIT; underlying Ultralytics implementation AGPL-3.0","sha256":"4836dca38e3aff04cb2056262838a47a72531a178ab016052f99492bba029ce4","classes":["with helmet","without helmet","rider","number plate"],"limitations":"Trained on Kaggle aneesarom/rider-with-helmet-without-helmet-number-plate; author metrics are not an independent Pakistan evaluation; hats and occlusion can confuse the model"},
    {"filename":"plate_detector.onnx","name":"open-image-models YOLOv9 tiny 640 license plates end-to-end","source":"https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx","version":"assets release, SHA256 pinned",
     "license":"Publisher open-image-models MIT; YOLOv9 architecture provenance https://github.com/WongKinYiu/yolov9 (GPL-3.0)","sha256":"c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076","classes":["license plate"],"limitations":"Localization only; no orientation or mounting-area classes. No Pakistan-specific accuracy claim; native YOLOv9 .pt is not Ultralytics-compatible"},
    {"filename":"rider_pose.pt","name":"Ultralytics YOLO11n pose COCO","source":ASSETS+"yolo11n-pose.pt","version":"v8.3.0 assets",
     "license":"AGPL-3.0; enterprise licensing available from Ultralytics","sha256":"869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0","classes":["person"],"limitations":"Seated pose supports association; cannot establish vehicle movement"},
]


def main():
    MODELS.mkdir(parents=True,exist_ok=True)
    with httpx.Client(follow_redirects=True,timeout=90,headers={"User-Agent":"LahoreTrafficIntelligenceAcademic/1.0"}) as client:
        for spec in MODEL_SPECS:
            path = MODELS / spec["filename"]
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()!=spec["sha256"]:
                raise RuntimeError(f"Existing {path.name} differs from the pinned model. Preserved; use a separate model path instead of overwriting weights.")
            if not path.is_file():
                response = client.get(spec["source"])
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest()!=spec["sha256"]:
                    raise RuntimeError(f"Checksum mismatch for {path.name}; no model installed")
                temporary = path.with_suffix(path.suffix+".download")
                temporary.write_bytes(response.content)
                temporary.replace(path)
            print(f"Verified {path.name}: {path.stat().st_size:,} bytes")
        licenses = {
            "helmet_LICENSE":f"https://raw.githubusercontent.com/AutonomousUI/traffic-helmet-detection/{HELMET_COMMIT}/LICENSE",
            "plate_LICENSE":"https://raw.githubusercontent.com/ankandrew/open-image-models/main/LICENSE",
        }
        for filename,url in licenses.items():
            response = client.get(url); response.raise_for_status()
            (MODELS/filename).write_bytes(response.content)
        # Save a NEW dated source; do not overwrite the original indexed document or fine dataset.
        url = "https://trafficpolice.punjab.gov.pk/offences_penalties"
        response = client.get(url); response.raise_for_status()
        if "Riding a two-wheel motorcycle without crash helmet" not in response.text:
            raise RuntimeError("Official penalties page has no expected schedule; verification was not updated")
        filename = f"punjab_offences_penalties_{date.today().isoformat()}.html"
        snapshot = ROOT/"data"/"raw"/"documents"/filename
        snapshot.write_bytes(response.content)
        verification = {"source_url":url,"verified_on":date.today().isoformat(),"snapshot_file":filename,
                        "sha256":hashlib.sha256(response.content).hexdigest(),"effective_date":None,"original_index_preserved":True}
        (ROOT/"data"/"processed"/"metadata"/"vision_fine_verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    sys.path.insert(0,str(ROOT/"backend"))
    from app.services.computer_vision.detector import get_model
    for kind in ("vehicle","helmet","pose"):
        model = get_model(kind)
        print(kind,"loaded; classes:",model.names)
    import onnxruntime as ort
    session = ort.InferenceSession(str(MODELS/"plate_detector.onnx"),providers=["CPUExecutionProvider"])
    print("plate ONNX loaded:",session.get_inputs()[0].shape,"output",session.get_outputs()[0].shape)
    (MODELS/"manifest.json").write_text(json.dumps({"models":MODEL_SPECS,"traffic_sign_model":None,
       "traffic_sign_limitation":"No verified, downloadable model suitable for Pakistan sign classes was provisioned; catalogue remains available.",
       "seatbelt_model":None},indent=2),encoding="utf-8")
    from refresh_vision_fines import main as refresh_fines
    refresh_fines()
    print("Vision setup complete; local vehicle/helmet/plate models and the structured legal registry are ready. OCR is bundled with the pinned backend dependency.")


if __name__=="__main__":
    main()
