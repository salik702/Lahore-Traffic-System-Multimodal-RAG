"""Jurisdiction-aware, auditable legal records independent of neural models or RAG.

The current amount must agree with its official HTML snapshot. Stale, altered,
unmapped or inapplicable records never contribute to a potential-fine total.
"""
import hashlib
import json
import re
from datetime import date
from html import unescape
from pathlib import Path

SOURCE_URL="https://trafficpolice.punjab.gov.pk/offences_penalties"
COLUMNS=("motorcycle","three_wheeler","motorcar_under_2000cc","motorcar_over_2000cc","transport_vehicle")
UNVERIFIED="Official fine amount requires verification."


def parse_schedule(html):
    rows={}
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>',html,re.S|re.I):
        cells=[re.sub(r'\s+',' ',unescape(re.sub('<[^>]+>',' ',c))).strip()
               for c in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',row,re.S|re.I)]
        if len(cells)!=7 or not re.fullmatch(r'\d+\.?',cells[0]):continue
        values={}
        for key,cell in zip(COLUMNS,cells[2:]):
            if cell in {'-','\u2014',''}:values[key]=None
            elif re.fullmatch(r'\d+',cell.replace(',','')):values[key]=int(cell.replace(',',''))
            else:raise ValueError("Unrecognized official monetary cell")
        code=int(cells[0].rstrip('.'))
        if code in rows:raise ValueError("Duplicate official schedule code")
        rows[code]={"description":cells[1],"amounts":values}
    if len(rows)!=25 or 'helmet' not in rows[19]['description'].lower() or 'pillion' not in rows[20]['description'].lower():
        raise ValueError("Official schedule layout changed; manual review required")
    return rows


class FineDatabase:
    def __init__(self,data_dir,today=None):
        self.today=today or date.today()
        self.records={}
        self.error=None
        self.valid=False
        try:
            data_dir=Path(data_dir)
            db=json.loads((data_dir/'raw/structured/vision_fines.json').read_text(encoding='utf-8'))
            filename=db['snapshot_file']
            if Path(filename).name!=filename or db['source_url']!=SOURCE_URL:raise ValueError('Invalid provenance')
            snapshot=data_dir/'raw/documents'/filename
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=db['snapshot_sha256']:raise ValueError('Official snapshot checksum mismatch')
            rows=parse_schedule(snapshot.read_text(encoding='utf-8'))
            age=(self.today-date.fromisoformat(db['last_verified_on'])).days
            if not 0<=age<=db.get('verification_max_age_days',90):raise ValueError('Official verification is stale or dated in the future')
            for record in db['records']:
                official=rows[record['schedule_code']]
                if record['amounts']!=official['amounts'] or record['description']!=official['description']:
                    raise ValueError('Structured record disagrees with the official snapshot')
                if record['jurisdiction']!='Punjab' or record['source_url']!=SOURCE_URL:
                    raise ValueError('Jurisdiction/source mismatch')
                if record['last_verified_on']!=db['last_verified_on']:
                    raise ValueError('Record verification date mismatch')
                if record['violation_id'] in self.records:raise ValueError('Duplicate violation ID')
                self.records[record['violation_id']]=record
            # Keep the exact verified legal text/checksum for the only occupancy rule.
            law=db['occupancy_rule']
            legal_path=data_dir/'raw/documents'/law['snapshot_file']
            if Path(law['snapshot_file']).name!=law['snapshot_file'] or hashlib.sha256(legal_path.read_bytes()).hexdigest()!=law['sha256']:
                raise ValueError('Occupancy legal source checksum mismatch')
            self.occupancy_rule=law
            self.valid=True
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.error=str(exc)
            self.records={}

    def max_motorcycle_occupants(self,jurisdiction):
        return self.occupancy_rule['max_occupants'] if self.valid and jurisdiction=='Punjab' else None

    def lookup(self,kind,vehicle,jurisdiction='Punjab',conditions=None):
        record=self.records.get(kind)
        result={"type":kind,"fine_amount":None,"currency":"PKR","fine_status":UNVERIFIED,
                "verified":False,"jurisdiction":jurisdiction,"rule":None,"sources":[],
                "effective_date":None,"last_verified_on":None,"retrieval_status":"Structured official schedule",
                "applicability_conditions":[],"amount_options":{}}
        if not self.valid or not record or jurisdiction!=record['jurisdiction']:return result
        sources=[{"title":"Punjab Offences & Penalties - Twelfth Schedule","url":SOURCE_URL,
                  "organization":"Traffic Police Punjab","verified_on":record['last_verified_on'],
                  "official":True,"scope":"Published Punjab schedule; effective date not stated"}]
        result.update(rule=record['description'],sources=sources,last_verified_on=record['last_verified_on'],
                      effective_date=record['effective_date'],legal_reference=record['legal_reference'],
                      evidence_requirements=record['evidence_requirements'],applicability_conditions=record['applicability_conditions'])
        if kind in {'motorcycle_overloading','helmet_violation'}:
            section='89' if kind=='motorcycle_overloading' else '89-A'
            sources.append({"title":f"Provincial Motor Vehicles Ordinance, Section {section}","url":self.occupancy_rule['source_url'],
                            "organization":"City Traffic Police Lahore","verified_on":self.occupancy_rule['last_verified_on'],
                            "official":True,"scope":"Driver plus one additional passenger" if kind=='motorcycle_overloading' else "Motorcycle driver and pillion helmet duty"})
        category=vehicle.get('legal_vehicle_category')
        if not category:
            category={'motorcycle':'motorcycle','rickshaw':'three_wheeler'}.get(vehicle['class'])
        if vehicle['class'] not in record['applicable_vehicle_types']:return result
        if not category:
            # Car displacement and private/commercial use are not visual attributes.
            result['amount_options']={k:v for k,v in record['amounts'].items() if v is not None}
            result['fine_status']=UNVERIFIED+' Legal vehicle category (including engine capacity) is unknown.'
            return result
        if any(not (conditions or {}).get(key) for key in record.get('required_context',[])):
            result['fine_status']=UNVERIFIED+' Required applicability context is unavailable.'
            return result
        amount=record['amounts'].get(category)
        if amount is not None:
            result.update(fine_amount=amount,verified=True,
                          fine_status='Verified official published schedule; effective date not stated')
        return result


def fine_summary(violations,assessments):
    # One schedule charge per vehicle/violation. Several rider heads remain evidence
    # of the same finding; neither uncertain items nor unknown fines are charged.
    unique={}
    for v in violations:
        if v['status']!='potential_violation':continue
        unique.setdefault((v['vehicle_id'],v['type']),v)
    values=list(unique.values())
    return {"supported_findings":len(values),
            "uncertain_findings":sum(a['status']=='uncertain' for a in assessments),
            "not_assessable":sum(a['status']=='not_assessable' for a in assessments),
            "verified_potential_fine":sum(v['fine_amount'] for v in values if v.get('fine_verified') and v.get('fine_amount') is not None),
            "unverified_penalties":sum(not v.get('fine_verified') or v.get('fine_amount') is None for v in values),
            "currency":"PKR","label":"Estimated potential penalties, not an official challan"}
