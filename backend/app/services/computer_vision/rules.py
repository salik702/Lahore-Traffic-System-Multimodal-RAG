"""Vehicle-specific evidence decisions. Legal amounts are resolved separately.

Detection absence is never an affirmative observation. Unsupported modules cannot
produce findings. This module has no model, OCR, network or LLM dependency.
"""
from .fines import fine_summary
from .policy import capabilities,policy
from .rider_association import iou


def assessment(kind,status,reason,confidence=None,evidence_ids=None):
    return {"type":kind,"status":status,"reason":reason,"confidence":confidence,
            "evidence_ids":evidence_ids or []}


def helmet_rule(vehicle,riders,scene,fines,jurisdiction):
    if vehicle['class']!='motorcycle':return []
    if not scene['image_quality']['sufficient'] or vehicle.get('classification_status','supported')!='supported':
        return [assessment('helmet_violation','not_assessable','Cannot determine from this image. Image quality or vehicle classification is insufficient.')]
    if not riders:
        return [assessment('helmet_violation','not_assessable','Cannot determine from this image. No rider-to-motorcycle relationship was established.')]
    results=[]
    for rider in riders:
        ids=[rider['person_id']]+([rider['helmet_detection_id']] if rider.get('helmet_detection_id') else [])
        bare=rider.get('helmet_status') in {'possible_no_helmet','uncertain_no_helmet'}
        quality=rider.get('head_quality',{}).get('sufficient',False)
        supported=rider['association']=='supported'
        score=rider.get('helmet_confidence')
        if bare and supported and quality and score is not None and score>=policy()['helmet_negative'] and rider.get('association_confidence') is not None:
            results.append({**assessment('helmet_violation','potential_violation',rider['association_evidence']+'. '+rider['helmet_reason'],
                            min(score,rider['association_confidence']),ids),'rider_id':rider['id']})
        elif bare:
            results.append({**assessment('helmet_violation','uncertain','Explicit bare-head candidate; head confidence, rider association or image quality is insufficient for a finding.',score,ids),'rider_id':rider['id']})
        elif rider.get('helmet_status')=='helmet_detected' and supported and quality:
            results.append({**assessment('helmet_violation','not_detected','Helmet localized. This does not certify helmet standard, fit or full legal compliance.',score,ids),'rider_id':rider['id']})
        else:
            results.append({**assessment('helmet_violation','not_assessable','Cannot determine from this image. '+rider.get('helmet_reason','Head evidence unavailable.'),None,ids),'rider_id':rider['id']})
    return results


def occupancy_rule(vehicle,riders,scene,fines,jurisdiction):
    if vehicle['class']!='motorcycle':return []
    maximum=fines.max_motorcycle_occupants(jurisdiction)
    supported=[r for r in riders if r['association']=='supported' and r.get('association_confidence') is not None]
    unique=[]
    for rider in supported:
        if rider['person_id'] in {r['person_id'] for r in unique}:continue
        if any(iou(rider['person_bbox'],r['person_bbox'])>.65 for r in unique):continue
        unique.append(rider)
    if not scene['image_quality']['sufficient'] or vehicle.get('classification_status','supported')!='supported':
        return [assessment('motorcycle_overloading','not_assessable','Cannot determine from this image. Quality or classification is insufficient.')]
    if maximum is None:
        return [assessment('motorcycle_overloading','not_assessable','Cannot determine from this image. Applicable passenger-limit rule is unverified for this jurisdiction.')]
    if len(unique)>maximum:
        return [assessment('motorcycle_overloading','potential_violation',
            f'{len(unique)} distinct people have supported seated associations. Punjab Section 89 permits driver plus one additional passenger.',
            min(r['association_confidence'] for r in unique),[r['person_id'] for r in unique])]
    return [assessment('motorcycle_overloading','not_assessable',
        f'{len(unique)} people have supported seated associations. Cannot determine the complete occupancy from this image; occluded passengers may be present.')]


# Each module may return several independent observations. The registry allows
# evaluated detectors/rules to be added without changing model or fine storage.
RULES={'helmet_violation':helmet_rule,'motorcycle_overloading':occupancy_rule}


def analyze_evidence(scene,fines,jurisdiction='Punjab'):
    violations=[]
    vehicle_results=[]
    all_assessments=[]
    for vehicle in scene['vehicles']:
        riders=[r for r in scene['riders'] if r['vehicle_id']==vehicle['id']]
        assessments=[]
        if vehicle['class']=='bicycle':
            assessments.append(assessment('motor_vehicle_rules','not_assessable','Motor-vehicle penalties are not applied to a bicycle.'))
        else:
            for evaluator in RULES.values():assessments.extend(evaluator(vehicle,riders,scene,fines,jurisdiction))
            for capability in capabilities():
                kind=capability['id']
                if capability['status']!='unsupported':continue
                if kind in {'seatbelt_violation','passenger_seatbelt'} and vehicle['class']=='motorcycle':continue
                assessments.append(assessment(kind,'not_assessable','Cannot determine from this image. '+capability['reason']))
        by_type={}
        for item in assessments:
            item['vehicle_id']=vehicle['id']
            if item['status']=='potential_violation':by_type.setdefault(item['type'],[]).append(item)
        vehicle_violations=[]
        for kind,items in by_type.items():
            legal=fines.lookup(kind,vehicle,jurisdiction)
            finding={'id':f'violation_{len(violations)+1}','type':kind,'name':
                     'Potential helmet violation' if kind=='helmet_violation' else 'Potential motorcycle passenger-limit violation',
                     'status':'potential_violation','vehicle_id':vehicle['id'],
                     'rider_id':items[0].get('rider_id'),'rider_ids':[i['rider_id'] for i in items if i.get('rider_id')],
                     'confidence':min(i['confidence'] for i in items),
                     'confidence_basis':'Minimum of actual detector/keypoint scores; not a calibrated probability of guilt.',
                     'visual_evidence':list(dict.fromkeys(i['reason'] for i in items)),
                     'evidence_ids':list(dict.fromkeys(e for i in items for e in i['evidence_ids'])),
                     'fine_amount':legal['fine_amount'],'fine_verified':legal['verified'],
                     'fine_status':legal['fine_status'],'rule':legal['rule'],'sources':legal['sources'],
                     'legal_reference':legal.get('legal_reference'),
                     'effective_date':legal['effective_date'],'last_verified_on':legal['last_verified_on'],
                     'jurisdiction':jurisdiction,'currency':'PKR'}
            vehicle_violations.append(finding)
            violations.append(finding)
        plate=next((p for p in scene['plate_results'] if p['vehicle_id']==vehicle['id']),None)
        summary=fine_summary(vehicle_violations,assessments)
        vehicle_results.append({'vehicle_id':vehicle['id'],'vehicle_type':vehicle['class'],
            'detection_confidence':vehicle['confidence'],'classification_status':vehicle.get('classification_status','supported'),
            'bbox':vehicle['bbox'],'riders':riders,'occupants':[p for p in scene.get('occupants',[]) if p['vehicle_id']==vehicle['id']],
            'plate':plate,'assessments':assessments,'violations':vehicle_violations,'summary':summary})
        all_assessments.extend(assessments)
    return {'potential_violations':violations,'vehicle_results':vehicle_results,
            'assessments':all_assessments,'summary':fine_summary(violations,all_assessments)}
