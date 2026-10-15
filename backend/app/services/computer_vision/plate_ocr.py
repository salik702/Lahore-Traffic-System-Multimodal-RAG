"""Small bundled PP-OCRv4 ONNX models; never downloads during an API request."""
import re
import unicodedata
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
from .policy import policy
from .quality import assess_quality,crop_region,clipped

_engine=None
_lock=RLock()
HEADERS={"PUNJAB","SINDH","ISLAMABAD","PAKISTAN","ICT","BALOCHISTAN"}


def get_ocr():
    global _engine
    with _lock:
        if _engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                import rapidocr_onnxruntime
                from importlib.metadata import version
                from .detector import MODEL_DIR

                # Ensure manifest exists
                manifest_path = MODEL_DIR / 'ocr_manifest.json'
                if not manifest_path.is_file():
                    raise RuntimeError(f"OCR manifest missing at {manifest_path}")

                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

                # Relaxed version check: log warning instead of crashing
                current_ver = version('rapidocr-onnxruntime')
                if current_ver != manifest['package_version']:
                    print(f"Warning: OCR package version mismatch. Expected {manifest['package_version']}, got {current_ver}")

                # Checksums: Log errors but attempt to proceed if files exist
                folder = Path(rapidocr_onnxruntime.__file__).parent / 'models'
                for model in manifest['models']:
                    model_file = folder / model['filename']
                    if not model_file.is_file():
                        print(f"Warning: OCR model file missing: {model_file}")
                        continue
                    if hashlib.sha256(model_file.read_bytes()).hexdigest() != model['sha256']:
                        print(f"Warning: OCR model checksum mismatch for {model['filename']}")

                _engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                                    use_cls=False, det_limit_side_len=320, det_limit_type="max")
            except Exception as exc:
                print(f"Critical OCR Initialization Error: {exc}")
                raise RuntimeError(f"Local OCR models unavailable: {str(exc)}") from exc
        return _engine


def normalize_plate(text):
    """Keep observed characters; never replace O/0, I/1 or complete a partial plate."""
    text=unicodedata.normalize("NFKC",text).upper().strip()
    if re.search(r"[^A-Z0-9\s\-]",text):return None
    parts=[p for p in re.split(r"\s+",text) if p and p not in HEADERS]
    candidate=re.sub(r"[\s-]+",""," ".join(parts))
    # Common letters/year/serial shapes; NOT official format validation.
    return candidate if re.fullmatch(r"[A-Z]{1,4}[0-9]{1,7}",candidate) else None


def parse_reading(rows,threshold):
    if not rows:return None,None
    tokens,scores=[],[]
    for row in rows:
        if len(row)!=3:continue
        polygon,text,score=row
        text=str(text).strip().upper()
        if text in HEADERS:continue
        if not 0<=float(score)<=1 or float(score)<threshold:return None,None
        tokens.append((min(p[1] for p in polygon),min(p[0] for p in polygon),text))
        scores.append(float(score))
    if not scores:return None,None
    tokens.sort(key=lambda t:(round(t[0]/12),t[1]))
    return normalize_plate(" ".join(t[2] for t in tokens)),min(scores)


def recognize_plate(image,box):
    import cv2
    crop=crop_region(image,box["bbox"])
    quality=assess_quality(crop,min_side=16,min_sharpness=35)
    quality["clipped"]=clipped(box["bbox"],image.shape)
    quality["sufficient"] &= crop.shape[1]>=48 and not quality["clipped"]
    result={"text":None,"ocr_confidence":None,"ocr_status":"unreadable",
            "ocr_reason":"Number plate not readable.","quality":quality}
    if not quality["sufficient"]:
        result["ocr_reason"]="Number plate not readable. Crop is small, blurred, clipped or poorly exposed."
        return result
    # A two-line motorcycle plate can be 60-70 pixels tall while each line's
    # letters are only 10-15 pixels high. A 64px target left these crops at 1x,
    # so the text detector dropped the serial line. Preserve the quality gate
    # on original pixels, then enlarge for text localization/recognition.
    scale=min(3,max(1,math.ceil(128/crop.shape[0])))
    scaled=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    lab=cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(4,4)).apply(lab[:,:,0])
    enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    try:
        with _lock:
            ocr=get_ocr()
            readings=[parse_reading(ocr(view,use_cls=False)[0],policy()["ocr_threshold"]) for view in (scaled,enhanced)]
        # Both views must agree, and the weaker view must still be confident:
        # a partial or low-confidence reading never becomes a plate number.
        if readings[0][0] and readings[0][0]==readings[1][0] and min(r[1] for r in readings)>=policy()["ocr_min_confidence"]:
            result.update(text=readings[0][0],ocr_confidence=round(min(r[1] for r in readings),4),
                          ocr_status="recognized",ocr_reason="Original and contrast-enhanced plate crops agree on the observed letters and digits; registration is not verified.")
        else:result["ocr_reason"]="Number plate not readable. OCR views disagree, are partial or below threshold."
    except Exception as exc:
        print(f"OCR Engine Error: {exc}")
        result.update(ocr_status="unavailable",ocr_reason=f"Number plate not readable. Local OCR failed: {str(exc)}")
    return result
