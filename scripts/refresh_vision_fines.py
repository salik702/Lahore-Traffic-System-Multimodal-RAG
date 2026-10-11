"""Fetch official Punjab schedule into an ADDITIVE dated snapshot and legal registry.

Does not replace the pre-existing fines catalogue or rebuild any RAG/model index.
Run: backend/.venv/Scripts/python.exe scripts/refresh_vision_fines.py
"""
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.services.computer_vision.fines import SOURCE_URL,parse_schedule

IDS={1:'speeding',2:'commercial_passengers',3:'red_light',4:'commercial_loading',5:'night_lights',
     6:'wrong_way',7:'tinted_windows',8:'lane_violation',9:'restricted_access',10:'obstruction',
     11:'dangerous_driving',12:'unlicensed',13:'prohibited_horn',14:'excessive_smoke',15:'unregistered',
     16:'age_limit',17:'fitness_certificate',18:'route_permit',19:'helmet_violation',
     20:'motorcycle_overloading',21:'mobile_phone',22:'seatbelt_violation',23:'illegal_parking',
     24:'registration_plate_display',25:'other_ordinance'}
REQUIREMENTS={
    1:'Calibrated speed measurement or trusted telemetry plus posted limit.',
    2:'Vehicle seating capacity, public service status and independently associated occupants.',
    3:'Controlling red signal, calibrated stop line and tracked crossing after red phase.',
    4:'Certified load limits and measured loading conditions.',
    6:'Established road direction and tracked vehicle movement.',
    8:'Calibrated lane/crossing markings, permitted maneuvers and trajectory.',
    11:'Temporal behavior evidence and scene context.',
    19:'Explicit bare head, adequate head quality, two-wheel motorcycle and unique seated-person association.',
    20:'At least three distinct supported occupants and applicable Punjab Section 89.',
    21:'Verified driver role, visible hand-held use and evidence of driving.',
    22:'Verified driver role, explicit unworn-belt evidence, clear torso and notified-road applicability.',
    23:'Verified location, applicable parking restriction and sufficient duration evidence.',
    24:'Observed mounting area and explicit non-display; official plate issuance/security features where relevant. Unreadable OCR is insufficient.',
}


def main():
    response=httpx.get(SOURCE_URL,follow_redirects=True,timeout=45,
                       headers={'User-Agent':'LahoreTrafficIntelligenceResearch/2.0'})
    response.raise_for_status()
    rows=parse_schedule(response.text)
    verified=date.today().isoformat()
    digest=hashlib.sha256(response.content).hexdigest()
    filename=f'punjab_offences_penalties_{verified}_{digest[:8]}.html'
    documents=ROOT/'data/raw/documents'
    path=documents/filename
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Existing dated snapshot differs; refusing to overwrite')
    path.write_bytes(response.content)
    law_path=documents/'punjab_motor_vehicles_ordinance_1965.pdf'
    import pymupdf
    with pymupdf.open(law_path) as law:
        pages=[(i+1,p.get_text()) for i,p in enumerate(law) if '89.' in p.get_text() and 'pillion' in p.get_text().lower()]
        match=next(((n,t) for n,t in pages if 'one person in addition' in t.lower() or 'one person in addition' in ' '.join(t.lower().split())),None)
        if not match:raise RuntimeError('Section 89 must be manually reviewed; legal text not recognized')
        page,text=match
        section=text[text.index('89.'):]
        section=re.split(r'89\.[\s]*\(A\)|89-A|90\.',section,maxsplit=1)[0]
    sources=json.loads((ROOT/'data/sources/sources.json').read_text(encoding='utf-8'))
    law_source=next(s for s in sources if s['id']=='punjab-mvo-1965')
    records=[]
    for code,row in rows.items():
        vehicle_types=['motorcycle'] if code in (19,20) else ['car','bus','truck','van','rickshaw','motorcycle']
        if code==22:vehicle_types=['car','bus','truck','van']
        records.append({'violation_id':IDS[code],'violation_name':row['description'],
            'description':row['description'],'schedule_code':code,'applicable_vehicle_types':vehicle_types,
            'jurisdiction':'Punjab','amounts':row['amounts'],'currency':'PKR',
            'legal_reference':f'Provincial Motor Vehicles Ordinance 1965, Twelfth Schedule, item {code}'+('; Section 89' if code==20 else '; Section 89-A' if code==19 else ''),
            'source_url':SOURCE_URL,'effective_date':None,'last_verified_on':verified,
            'evidence_requirements':REQUIREMENTS.get(code,'Authoritative records or a properly implemented and evaluated detector; cannot infer from object absence.'),
            'applicability_conditions':['Punjab jurisdiction must be supplied, not inferred from image.','Vehicle legal category must be established.']+(['Seatbelt rule refers to a notified road.'] if code==22 else []),
            'required_context':['notified_road'] if code==22 else []})
    db={'schema_version':1,'source_url':SOURCE_URL,'snapshot_file':filename,'snapshot_sha256':digest,
        'last_verified_on':verified,'effective_date':None,'verification_max_age_days':90,
        'records':records,'occupancy_rule':{'max_occupants':2,'legal_reference':'Section 89',
            'source_url':'https://ctplahore.gop.pk/system/files/mvo_1965_0.pdf',
            'snapshot_file':law_path.name,'sha256':hashlib.sha256(law_path.read_bytes()).hexdigest(),
            'last_verified_on':law_source['retrieval_date'],'last_reviewed_on':verified,
            'pdf_page':page,'source_text':section.strip()}}
    destination=ROOT/'data/raw/structured/vision_fines.json'
    temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(db,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(destination)
    print(f'Verified {len(records)} Punjab records on {verified}; effective date unstated; original catalogue preserved.')


if __name__=='__main__':main()
