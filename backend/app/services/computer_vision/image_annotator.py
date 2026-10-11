"""OpenCV annotations returned in memory with vehicle IDs and owned evidence."""
import base64

def annotate(image,detections,helmets,plates,redact_plates=False):
    import cv2
    canvas=image.copy()
    for plate in plates if redact_plates else []:
        x1,y1,x2,y2=map(int,plate["bbox"])
        x1,y1=max(0,x1-3),max(0,y1-3);x2,y2=min(canvas.shape[1],x2+3),min(canvas.shape[0],y2+3)
        canvas[y1:y2,x1:x2]=(45,45,45)
    helmet_ids={b["id"] for b in helmets};plate_ids={b["id"] for b in plates}
    for item in detections+helmets+plates:
        label=item["class"].lower().replace("_"," ");color=(230,130,45)
        if item["id"] in helmet_ids:
            color=(60,190,70) if label in {"helmet","with helmet"} else (65,65,225) if label in {"no helmet","without helmet"} else (30,180,245)
        elif item["id"] in plate_ids:color=(205,90,185)
        elif label=="person":color=(30,180,245)
        x1,y1,x2,y2=map(int,item["bbox"]);cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
        owner=item.get('vehicle_id') or (item['id'] if label in {'car','motorcycle','bus','truck','bicycle'} else '')
        text=f'{owner} {label} {item["confidence"]:.0%}'.strip()
        (w,h),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.48,1)
        tx=min(max(0,x1),max(0,canvas.shape[1]-w-6));ty=min(max(h+6,y1),canvas.shape[0]-baseline-2)
        cv2.rectangle(canvas,(tx,ty-h-5),(tx+w+5,ty+baseline),color,-1)
        cv2.putText(canvas,text,(tx+2,ty-2),cv2.FONT_HERSHEY_SIMPLEX,.48,(15,15,15),1,cv2.LINE_AA)
    ok,encoded=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise RuntimeError("Annotated image encoding failed")
    return "data:image/jpeg;base64,"+base64.b64encode(encoded).decode('ascii')
