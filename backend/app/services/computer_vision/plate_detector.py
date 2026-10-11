"""Publisher's end-to-end YOLOv9 ONNX export (includes NMS), FP32 CPU inference."""
import os
import numpy as np
from .detector import MODEL_DIR, INFERENCE_LOCK
from .rider_association import area, intersection
from .policy import policy
from .plate_ocr import recognize_plate
from .quality import clipped,crop_region,image_url

_session = None


def predict_plates(image):
    import cv2
    import onnxruntime as ort
    global _session
    with INFERENCE_LOCK:
        if _session is None:
            path = os.getenv("PLATE_MODEL_PATH", str(MODEL_DIR / "plate_detector.onnx"))
            try:
                options = ort.SessionOptions()
                options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                _session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
                if _session.get_inputs()[0].shape != [1, 3, 640, 640] or len(_session.get_outputs()[0].shape)!=2 or _session.get_outputs()[0].shape[-1]!=7:
                    _session = None
                    raise ValueError("Unexpected plate model input shape")
            except Exception as exc:
                raise RuntimeError("Plate model initialization failed. Run scripts/setup_vision.py.") from exc
        h,w = image.shape[:2]
        ratio = min(640/h,640/w)
        nw,nh = round(w*ratio),round(h*ratio)
        dw,dh = (640-nw)/2,(640-nh)/2
        resized = cv2.resize(image,(nw,nh))
        padded = cv2.copyMakeBorder(resized,round(dh-.1),round(dh+.1),round(dw-.1),round(dw+.1),cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        predictions = _session.run(None,{_session.get_inputs()[0].name:tensor})[0]
        boxes = []
        for row in predictions:
            if row[6]<.4: continue
            x1,y1,x2,y2 = (row[1:5]-np.array([dw,dh,dw,dh]))/ratio
            coords = [max(0,float(x1)),max(0,float(y1)),min(w,float(x2)),min(h,float(y2))]
            if coords[2]<=coords[0] or coords[3]<=coords[1]: continue
            boxes.append({"id":f"plate_{len(boxes)+1}","class":"number plate","bbox":[round(v,1) for v in coords],"confidence":round(float(row[6]),4)})
        return boxes


def detect_plates(image,vehicles,helmet_boxes):
    import cv2
    boxes = predict_plates(image)
    # Inspect bounded vehicle crops to preserve small plate detail lost in a full-frame resize.
    from .rider_association import iou
    for vehicle in sorted(vehicles,key=lambda v:area(v["bbox"]),reverse=True)[:8]:
        if vehicle["class"]=="bicycle": continue
        x1,y1,x2,y2 = map(int,vehicle["bbox"])
        if min(x2-x1,y2-y1)<100 or any(intersection(b["bbox"],vehicle["bbox"])/area(b["bbox"])>.85 for b in boxes): continue
        crop = image[max(0,y1):y2,max(0,x1):x2]
        if not crop.size: continue
        for detection in predict_plates(crop):
            b = detection["bbox"]
            detection["bbox"] = [b[0]+max(0,x1),b[1]+max(0,y1),b[2]+max(0,x1),b[3]+max(0,y1)]
            if any(iou(detection["bbox"],b["bbox"])>.4 for b in boxes): continue
            boxes.append({**detection,"id":f"plate_{len(boxes)+1}"})
    # Only already context-associated specialized detections may supplement plates.
    for detection in helmet_boxes:
        if detection["class"] != "number plate": continue
        owner=next((v for v in vehicles if v['id']==detection.get('candidate_vehicle_id') and v['class']=='motorcycle'),None)
        if not owner or intersection(detection['bbox'],owner['bbox'])/area(detection['bbox'])<.85:continue
        if any(iou(detection["bbox"],b["bbox"])>.4 for b in boxes): continue
        boxes.append({**detection,"id":f"plate_{len(boxes)+1}"})
    associations = {v["id"]:[] for v in vehicles if v["class"]!="bicycle"}
    for box in boxes:
        candidates = [v for v in vehicles if v["class"]!="bicycle" and intersection(box["bbox"],v["bbox"])/area(box["bbox"])>.85]
        box["vehicle_id"] = candidates[0]["id"] if len(candidates)==1 else None
        if box["vehicle_id"]: associations[box["vehicle_id"]].append(box)
    # Bound OCR work, keep localization distinct from recognition confidence.
    for index,box in enumerate(sorted(boxes,key=lambda b:b["confidence"],reverse=True)):
        if box.get('vehicle_id') and index<policy()['max_plate_crops']:
            box.update(recognize_plate(image,box))
            crop=crop_region(image,box['bbox'])
            if crop.size:box['crop_url']=image_url(crop,max_side=520)
        else:
            box.update(text=None,ocr_confidence=None,ocr_status='unavailable',
                       ocr_reason='Number plate not readable. Ownership is ambiguous or the per-image OCR budget is exhausted.')
    results = []
    for vehicle in vehicles:
        if vehicle["class"]=="bicycle": continue
        matches = associations[vehicle["id"]]
        x1,y1,x2,y2 = map(int,vehicle["bbox"])
        crop = image[max(0,y1):y2,max(0,x1):x2]
        quality = bool(min(x2-x1,y2-y1)>=80 and crop.size and cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()>=35)
        recognized=[b for b in matches if b.get('text')]
        texts={b['text'] for b in recognized}
        unique=len(texts)==1
        best=max(recognized if unique else matches,key=lambda b:b['confidence'],default=None)
        is_clipped=bool(best and clipped(best['bbox'],image.shape))
        status='recognized' if unique else 'outside_image' if is_clipped else 'unreadable' if matches else 'visibility_not_established'
        results.append({"vehicle_id":vehicle["id"],"status":status,
            "text":next(iter(texts)) if unique else None,
            "ocr_confidence":min(b['ocr_confidence'] for b in recognized) if unique else None,
            "confidence":max((b["confidence"] for b in matches),default=None),
            "plate_ids":[b['id'] for b in matches],"mounting_area_verified":False,
            "vehicle_clipped":clipped(vehicle['bbox'],image.shape),"vehicle_quality_sufficient":quality,
            "crop_url":best.get('crop_url') if best else None,
            "reason":best['ocr_reason'] if unique else 'Number plate not readable. Conflicting plate readings for this vehicle.' if len(texts)>1 else
                     'Number plate not readable. The localized plate is clipped by the image border.' if is_clipped else
                     (best.get('ocr_reason','Number plate not readable.') if best else
                      'No plate localized. The mounting area, viewing direction, and possible obstruction are unverified; this is not evidence of a missing plate.')})
    return results,boxes
