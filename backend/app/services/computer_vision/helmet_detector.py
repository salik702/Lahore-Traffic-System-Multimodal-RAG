"""Head inference within motorcycle/person context, never a source of vehicle identity."""
from .detector import predict
from .policy import policy
from .quality import assess_quality,crop_region,clipped
from .rider_association import intersection,area,iou

POSITIVE={"helmet","with helmet","helmet on"}
NEGATIVE={"no helmet","without helmet","helmet off","nohelmet"}


def head_region(person):
    return [person[0],person[1],person[2],person[1]+(person[3]-person[1])*.38]


def match_heads(image,riders,boxes,people=None):
    cfg=policy()
    heads=[b for b in boxes if b["class"].lower() in POSITIVE | NEGATIVE]
    accepted=[]
    for rider in riders:
        rider.update(helmet_status="unable_to_determine",helmet_confidence=None,
                     helmet_detection_id=None,potential_violation=False,
                     helmet_reason="No unambiguous explicit helmet or bare-head observation")
        region=head_region(rider["person_bbox"])
        matches=[b for b in heads if intersection(region,b["bbox"])/area(b["bbox"])>.6]
        positive=[b for b in matches if b["class"].lower() in POSITIVE and b['confidence']>=cfg['helmet_positive']]
        negative=[b for b in matches if b["class"].lower() in NEGATIVE and b['confidence']>=cfg['helmet_negative']]
        if positive and negative:
            rider["helmet_reason"]="Conflicting helmet and bare-head observations; manual review required"
            continue
        if not matches:continue
        box=max(positive or negative or matches,key=lambda b:b["confidence"])
        owners=[p for p in (people or [{"id":r["person_id"],"bbox":r["person_bbox"]} for r in riders])
                if intersection(head_region(p["bbox"]),box["bbox"])/area(box["bbox"])>.6]
        if len(owners)!=1 or owners[0]["id"]!=rider["person_id"]:
            rider["helmet_reason"]="Head may belong to multiple people; association withheld"
            continue
        quality=assess_quality(crop_region(image,box["bbox"]),min_side=20,min_sharpness=35)
        quality["clipped"]=clipped(box["bbox"],image.shape)
        quality["sufficient"] &= not quality["clipped"]
        rider["head_quality"]=quality
        if not quality["sufficient"]:
            rider["helmet_reason"]="Head is too small, blurred, poorly exposed or clipped"
            continue
        is_positive=box['class'].lower() in POSITIVE
        threshold=cfg["helmet_positive"] if is_positive else cfg["helmet_negative"]
        if box["confidence"]<threshold:
            rider["helmet_reason"]="Explicit head observation is below the configured class threshold"
            if not is_positive:
                rider.update(helmet_status='uncertain_no_helmet',helmet_confidence=box['confidence'],
                             helmet_detection_id=box['id'])
                accepted.append({**box,'vehicle_id':rider['vehicle_id'],'rider_id':rider['id'],'association':rider['association']})
            continue
        rider.update(helmet_confidence=box["confidence"],helmet_detection_id=box["id"])
        if is_positive:
            rider.update(helmet_status="helmet_detected",helmet_reason="Specialized model explicitly localized a helmet on this person's head")
        else:
            rider.update(helmet_status="possible_no_helmet",potential_violation=rider["association"]=="supported",
                         helmet_reason="Specialized model explicitly localized a bare head; motorcycle association must also be supported")
        accepted.append({**box,"vehicle_id":rider["vehicle_id"],"rider_id":rider["id"],"association":rider["association"]})
    return riders,accepted


def detect_helmets(image,riders,people=None):
    if not riders:return [],[]
    boxes=[]
    groups={}
    for rider in riders:groups.setdefault(rider["vehicle_id"],[]).append(rider)
    def add_predictions(local,x1,y1,vehicle_id):
        for item in local:
            if item["class"].lower() not in POSITIVE | NEGATIVE | {'number plate'}:continue
            b=item["bbox"]
            translated={**item,"bbox":[b[0]+x1,b[1]+y1,b[2]+x1,b[3]+y1],"candidate_vehicle_id":vehicle_id,
                        "id":f"helmet_{len(boxes)+1}"}
            duplicate=next((j for j,other in enumerate(boxes) if other['class']==item['class'] and iou(other['bbox'],translated['bbox'])>.5),None)
            if duplicate is not None:
                if translated['confidence']>boxes[duplicate]['confidence']:
                    boxes[duplicate]={**translated,'id':boxes[duplicate]['id']}
            else:boxes.append(translated)
    for vehicle_id,group in list(groups.items())[:policy()["max_motorcycle_crops"]]:
        regions=[r["person_bbox"] for r in group]
        h,w=image.shape[:2]
        x1=max(0,int(min(b[0] for b in regions))-12)
        y1=max(0,int(min(b[1] for b in regions))-12)
        x2=min(w,int(max(b[2] for b in regions))+12)
        y2=min(h,int(max(b[3] for b in regions))+12)
        if x2<=x1 or y2<=y1:continue
        local,_=predict("helmet",image[y1:y2,x1:x2],.4,640)
        add_predictions(local,x1,y1,vehicle_id)
    riders,_=match_heads(image,riders,boxes,people)
    # A large union crop may shrink an individual head; retry unresolved people
    # separately, without lowering class thresholds or inferring object absence.
    unresolved=[r for r in riders if r['helmet_status'] in {'unable_to_determine','uncertain_no_helmet'}
                and len(groups[r['vehicle_id']])>1]
    unresolved.sort(key=lambda r:area(r['person_bbox']),reverse=True)
    for rider in unresolved[:policy()['max_head_refinements']]:
        b=rider['person_bbox'];h,w=image.shape[:2]
        x1,y1=max(0,int(b[0])-12),max(0,int(b[1])-12)
        x2,y2=min(w,int(b[2])+12),min(h,int(b[3])+12)
        if x2<=x1 or y2<=y1:continue
        local,_=predict('helmet',image[y1:y2,x1:x2],.4,640)
        add_predictions(local,x1,y1,rider['vehicle_id'])
    riders,heads=match_heads(image,riders,boxes,people)
    return riders,heads+[b for b in boxes if b['class']=='number plate']
