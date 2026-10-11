"""Spatial association corroborated by seated pose; ambiguous associations stay uncertain."""
import math

def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(b):
    return max(1,(b[2]-b[0])*(b[3]-b[1]))
def iou(a,b):
    overlap=intersection(a,b)
    return overlap/max(1,area(a)+area(b)-overlap)
def seated(points,bike):
    if not points or len(points)<17:return False
    for hip,knee,ankle in [(11,13,15),(12,14,16)]:
        h,k,a=[points[i] for i in (hip,knee,ankle)]
        if min(h[2],k[2],a[2])<.5:continue
        u,v=(h[0]-k[0],h[1]-k[1]),(a[0]-k[0],a[1]-k[1])
        denom=math.hypot(*u)*math.hypot(*v)
        angle=math.degrees(math.acos(max(-1,min(1,(u[0]*v[0]+u[1]*v[1])/max(1,denom)))))
        if angle<145 and bike[0]<=h[0]<=bike[2] and bike[1]-(bike[3]-bike[1])*.35<=h[1]<=bike[3]:return True
    return False

def associate_riders(detections,poses=None):
    motorcycles=[d for d in detections if d["class"]=="motorcycle" and d.get("classification_status", "supported")=="supported"]
    people=[d for d in detections if d["class"]=="person"]
    riders=[]
    for person in people:
        p=person["bbox"];candidates=[]
        # Seated poses also occur in cars; enclosing vehicle context takes priority.
        if enclosing_vehicles(person, detections):
            continue
        for bike in motorcycles:
            b=bike["bbox"];overlap=intersection(p,b)/area(p)
            horizontal=abs((p[0]+p[2]-b[0]-b[2])/2)/max(1,b[2]-b[0])
            if overlap>=.12 and horizontal<.7 and p[1]<b[1]+.4*(b[3]-b[1]):
                candidates.append((overlap+.3*(1-horizontal),bike))
        if not candidates:continue
        candidates.sort(key=lambda v:v[0],reverse=True);score,bike=candidates[0]
        own_poses=[q for q in poses or [] if q.get('person_id')==person['id']]
        pose=max(own_poses or poses or [],key=lambda x:iou(x["bbox"],p),default=None)
        points=pose["points"] if pose and iou(pose["bbox"],p)>.45 else None
        ambiguous=len(candidates)>1 and score-candidates[1][0]<.12
        supported=not ambiguous and seated(points,bike["bbox"])
        pose_confidence=seated_confidence(points,bike["bbox"]) if supported else None
        riders.append({"id":f"rider_{len(riders)+1}","vehicle_id":bike["id"],"person_id":person["id"],"person_bbox":p,
                       "association":"supported" if supported else "uncertain",
                       "association_confidence":round(min(person["confidence"],bike["confidence"],pose_confidence),4) if supported else None,
                       "geometry_score":round(score,4),"pose_confidence":pose_confidence,
                       "association_evidence":("Unique motorcycle, seated leg pose"+(' resolved on a person crop' if pose and pose.get('source')=='person_crop' else '')+", spatial overlap; no enclosing car/bus/truck") if supported else "Spatial overlap only; standing person or ambiguous motorcycle cannot be excluded"})
    return riders


def enclosing_vehicles(person, detections):
    p=person["bbox"]
    torso=[p[0],p[1],p[2],p[1]+(p[3]-p[1])*.65]
    return [v for v in detections if v["class"] in {"car","bus","truck","van","rickshaw"}
            and intersection(torso,v["bbox"])/area(torso)>.80
            and intersection(p,v["bbox"])/area(p)>.75]


def seated_confidence(points,bike):
    """Minimum actual keypoint confidence of a qualifying leg."""
    if not points or len(points)<17:return None
    values=[]
    for indices in ((11,13,15),(12,14,16)):
        leg=[[0,0,0] for _ in range(17)]
        for i in indices:leg[i]=points[i]
        if seated(leg,bike):values.append(min(points[i][2] for i in indices))
    return max(values) if values else None


def associate_occupants(detections):
    """Containment cannot establish driver/front-passenger role or image depth."""
    results=[]
    for person in (p for p in detections if p["class"]=="person"):
        matches=enclosing_vehicles(person,detections)
        if not matches:continue
        results.append({"person_id":person["id"],"vehicle_id":matches[0]["id"] if len(matches)==1 else None,
                        "candidate_vehicle_ids":[v["id"] for v in matches],"role":"unknown",
                        "association":"uncertain","confidence":None,
                        "reason":"Person contained in enclosed-vehicle region; occupant role and depth cannot be established from boxes alone."})
    return results
