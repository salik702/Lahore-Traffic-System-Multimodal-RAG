"""Run genuine inference against reviewed labels; report unavailable metrics as null.

Custom manifests may add independently labeled representative images using the same
schema. Never use test mocks or publisher metrics for application accuracy claims.
"""
import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.services.computer_vision.violation_analyzer import analyze,model_metadata


def metrics(pairs):
    if not pairs:return {'samples':0,'precision':None,'recall':None,'f1':None,'false_positive_rate':None}
    tp=sum(truth and predicted for truth,predicted in pairs)
    fp=sum(not truth and predicted for truth,predicted in pairs)
    fn=sum(truth and not predicted for truth,predicted in pairs)
    tn=sum(not truth and not predicted for truth,predicted in pairs)
    return {'samples':len(pairs),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None}


def edit_distance(expected,observed):
    previous=list(range(len(observed)+1))
    for i,left in enumerate(expected,1):
        current=[i]
        for j,right in enumerate(observed,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]


def ocr_metrics(readings):
    if not readings:
        return {'samples':0,'exact_match_accuracy':None,'character_accuracy':None,
                'reason':'No manually labeled, readable foreground plates in this manifest. Synthetic OCR tests are not an accuracy benchmark.'}
    errors=sum(edit_distance(truth,observed or '') for truth,observed in readings)
    characters=sum(len(truth) for truth,_ in readings)
    return {'samples':len(readings),'exact_match_accuracy':sum(truth==observed for truth,observed in readings)/len(readings),
            'character_accuracy':max(0,1-errors/characters),'character_error_rate':errors/characters,
            'recognized_samples':sum(bool(observed) for _,observed in readings),
            'scope':'Manually transcribed readable foreground plates only; abstentions count as errors. Seen during implementation; not held-out OCR evaluation.'}


async def run(args):
    import torch
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    output={'evaluated_at':datetime.now(timezone.utc).isoformat(),'scope':manifest['scope'],
            'limitations':manifest.get('limitations',[]),'model_versions':model_metadata(),
            'hardware':{'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available(),
                        'cpu_threads_used':min(4,__import__('os').cpu_count() or 1)},'images':[],'skipped':[]}
    pairs={kind:[] for kind in ['helmet_violation','motorcycle_overloading']}
    primary_pairs={kind:[] for kind in pairs}
    types={kind:[] for kind in ['car','motorcycle','bus','truck']}
    latencies=[]
    plate_readings=[]
    with tempfile.TemporaryDirectory(prefix='traffic-evaluation-') as folder:
        for item in manifest['images']:
            source=ROOT/item['path']
            if not source.is_file():
                output['skipped'].append(item['path']);continue
            with Image.open(source) as image:
                normalized=ImageOps.exif_transpose(image).convert('RGB');normalized.thumbnail((1920,1920))
                path=Path(folder)/'image.png';normalized.save(path,format='PNG',compress_level=1)
            # One explicitly excluded warm-up per scene precedes repeated timing.
            started=time.perf_counter()
            result=await analyze(path,None,None,SimpleNamespace(data_dir=ROOT/'data'))
            warmup_ms=(time.perf_counter()-started)*1000
            scene_times=[]
            for _ in range(args.repeat):
                started=time.perf_counter()
                result=await analyze(path,None,None,SimpleNamespace(data_dir=ROOT/'data'))
                scene_times.append((time.perf_counter()-started)*1000)
            latencies.extend(scene_times)
            predicted={v['type'] for v in result['potential_violations']}
            primary=next((v for v in result['vehicle_results'] if v['vehicle_id']==result['primary_vehicle_id']),None)
            primary_findings={v['type'] for v in primary['violations']} if primary else set()
            if item.get('primary_plate_text'):
                observed=(primary.get('plate') or {}).get('text') if primary else None
                plate_readings.append((item['primary_plate_text'],observed))
            classes={v['class'] for v in result['vehicles']}
            for kind in pairs:
                if item.get(kind) is not None:pairs[kind].append((item[kind],kind in predicted))
                if item.get('primary_'+kind) is not None:primary_pairs[kind].append((item['primary_'+kind],kind in primary_findings))
            for kind in types:types[kind].append((kind in item['vehicle_types'],kind in classes))
            output['images'].append({'path':item['path'],'ground_truth':item,'predicted_types':sorted(classes),
                'violations':[{'type':v['type'],'vehicle_id':v['vehicle_id'],'confidence':v['confidence']} for v in result['potential_violations']],
                'primary_vehicle':{'id':primary['vehicle_id'],'type':primary['vehicle_type'],
                    'violations':sorted(primary_findings),'potential_fine':primary['summary']['verified_potential_fine']} if primary else None,
                'helmet_observations':len(result['helmet_detections']),
                'plate_readings':[{'vehicle_id':p['vehicle_id'],'status':p['status'],'text':p['text']} for p in result['plate_results']],
                'warmup_ms':round(warmup_ms,2),'latency_ms':list(map(lambda n:round(n,2),scene_times)),
                'median_ms':round(statistics.median(scene_times),2),'warnings':result['warnings']})
    output['image_level_violation_metrics']={kind:metrics(value) for kind,value in pairs.items()}
    output['primary_vehicle_violation_metrics']={kind:metrics(value) for kind,value in primary_pairs.items()}
    output['image_level_vehicle_presence_metrics']={kind:metrics(value) for kind,value in types.items()}
    output['plate_ocr_metrics']=ocr_metrics(plate_readings)
    output['latency']={'measured_runs':len(latencies),'median_ms':round(statistics.median(latencies),2) if latencies else None,
                      'min_ms':round(min(latencies),2) if latencies else None,'max_ms':round(max(latencies),2) if latencies else None,
                      'scope':'Warm local image analysis (vehicle, pose, helmet, plate OCR, legal rules); excludes upload, PDF and startup.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'violation_metrics':output['image_level_violation_metrics'],'primary_vehicle_metrics':output['primary_vehicle_violation_metrics'],'latency':output['latency'],
                      'plate_ocr_metrics':output['plate_ocr_metrics'],'images':len(output['images']),'skipped':output['skipped']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=ROOT/'backend/tests/fixtures/vision_smoke_ground_truth.json')
    parser.add_argument('--output',type=Path,default=ROOT/'backend/storage/verification/vision-evaluation.json')
    parser.add_argument('--repeat',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.repeat<=20:parser.error('--repeat must be between 1 and 20')
    asyncio.run(run(args))
