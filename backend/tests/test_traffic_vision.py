"""Evidence, API and report regression tests. Synthetic boxes are TEST fixtures only."""
import asyncio
import copy
import io
import json
import tempfile
import unittest
from datetime import date,timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.api.routes import router
from app.services.computer_vision.detector import resolve_vehicle_classes
from app.services.computer_vision.fines import FineDatabase,fine_summary,UNVERIFIED
from app.services.computer_vision.helmet_detector import detect_helmets,match_heads
from app.services.computer_vision.plate_detector import detect_plates
from app.services.computer_vision.plate_ocr import normalize_plate,recognize_plate
from app.services.computer_vision.quality import assess_quality,image_url
from app.services.computer_vision.rider_association import associate_riders
from app.services.computer_vision.rules import analyze_evidence
from app.services.computer_vision.violation_analyzer import visual_analysis,finish_analysis,DISCLAIMER
from app.services.computer_vision.pdf_report import build_report,local_analysis_time
from app.services.computer_vision.report_store import ReportStore
from app.services.computer_vision.foreground import foreground_vehicle,subject_bounds,report_scope

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'


def box(id,kind,bbox,confidence=.95):
    return {'id':id,'class':kind,'bbox':bbox,'confidence':confidence,'classification_status':'supported'}


def rider(id,vehicle,person,bbox,status='possible_no_helmet'):
    return {'id':id,'vehicle_id':vehicle,'person_id':person,'person_bbox':bbox,'association':'supported',
            'association_confidence':.90,'association_evidence':'Test-fixture seated pose and unique motorcycle association',
            'helmet_status':status,'helmet_confidence':.92,'helmet_detection_id':'head_'+id,
            'helmet_reason':'Test-fixture explicit bare head' if status=='possible_no_helmet' else 'Test-fixture explicit helmet',
            'head_quality':{'sufficient':True}}


def scene(kind='motorcycle',riders=None):
    return {'vehicles':[box('v1',kind,[40,40,560,400])],'riders':riders or [],'plate_results':[],
            'image_quality':{'sufficient':True,'sharpness':100,'reasons':[]},'occupants':[]}


def textured_image():return np.random.default_rng(10).integers(35,220,(480,640,3),dtype=np.uint8)


def report_result(visual):
    visual={**visual,'original_image_url':image_url(textured_image()),'annotated_image_url':image_url(textured_image()),
            'detections':[],'persons':[],'helmet_detections':[],'plate_detections':[],'observations':[],
            'evidence_crops':[],'warnings':[],'limitations':['Test fixture; not a real-world accuracy benchmark.'],
            'timings':{'analysis_ms':10},'report_id':'a'*32,'analyzed_at':'2026-10-11T00:00:00+00:00',
            'filename':'test-input.png','analysis_status':'Completed with limitations','pipeline_version':'2.0.0',
            'confidence_calibration':'Uncalibrated model scores','disclaimer':DISCLAIMER,'models':[]}
    return finish_analysis(visual,SimpleNamespace(data_dir=DATA),'Punjab')


class EvidenceRulesTests(unittest.TestCase):
    def setUp(self):self.db=FineDatabase(DATA)

    def test_car_driver_cannot_receive_motorcycle_helmet_fine_even_with_bad_association(self):
        result=analyze_evidence(scene('car',[rider('r1','v1','driver',[60,60,120,200])]),self.db)
        self.assertEqual(result['potential_violations'],[])
        self.assertFalse(any(a['type']=='helmet_violation' for a in result['assessments']))

    def test_seatbelt_worn_and_visibly_unworn_remain_unsupported(self):
        for observed in ('seatbelt_worn','seatbelt_visibly_unworn'):
            with self.subTest(observed=observed):
                s=scene('car');s['occupants']=[{'vehicle_id':'v1','person_id':'p1','role':'driver','observation':observed}]
                result=analyze_evidence(s,self.db)
                item=next(a for a in result['assessments'] if a['type']=='seatbelt_violation')
                self.assertEqual(item['status'],'not_assessable')
                self.assertEqual(result['potential_violations'],[])

    def test_explicit_helmet_is_not_a_violation(self):
        result=analyze_evidence(scene(riders=[rider('r1','v1','p1',[40,50,100,250],'helmet_detected')]),self.db)
        self.assertEqual(result['potential_violations'],[])
        self.assertIn('not_detected',[a['status'] for a in result['assessments'] if a['type']=='helmet_violation'])

    def test_explicit_bare_head_requires_quality_and_supported_association(self):
        r=rider('r1','v1','p1',[40,50,100,250])
        self.assertEqual(len(analyze_evidence(scene(riders=[r]),self.db)['potential_violations']),1)
        for field,value in [('association','uncertain'),('head_quality',{'sufficient':False}),('helmet_confidence',.50),('helmet_status','unable_to_determine')]:
            with self.subTest(field=field):
                changed={**r,field:value}
                self.assertEqual(analyze_evidence(scene(riders=[changed]),self.db)['potential_violations'],[])

    def test_absent_helmet_is_not_a_violation(self):
        r=rider('r1','v1','p1',[40,50,100,250],'unable_to_determine');r['helmet_confidence']=None
        self.assertEqual(analyze_evidence(scene(riders=[r]),self.db)['potential_violations'],[])

    def test_multiple_violations_and_no_duplicate_helmet_charge(self):
        rs=[rider(f'r{i}','v1',f'p{i}',[40+140*i,40,130+140*i,300]) for i in range(3)]
        result=analyze_evidence(scene(riders=rs),self.db)
        self.assertEqual({v['type'] for v in result['potential_violations']},{'helmet_violation','motorcycle_overloading'})
        official=sum(self.db.lookup(kind,{'class':'motorcycle'})['fine_amount'] for kind in ('helmet_violation','motorcycle_overloading'))
        self.assertEqual(result['summary']['verified_potential_fine'],official)
        self.assertEqual(len(result['potential_violations'][0]['rider_ids']),3)

    def test_duplicate_people_cannot_create_overloading(self):
        r=rider('r1','v1','p1',[40,50,100,250],'helmet_detected')
        rs=[r,{**r,'id':'r2'},{**r,'id':'r3','person_id':'duplicate_same_box'}]
        self.assertEqual(analyze_evidence(scene(riders=rs),self.db)['potential_violations'],[])

    def test_multiple_cars_motorcycles_and_independent_fines(self):
        s=scene('car');s['vehicles'] += [box('m1','motorcycle',[20,100,150,400]),box('m2','motorcycle',[200,100,330,400])]
        s['riders']=[rider('r1','m1','p1',[20,40,100,300]),rider('r2','m2','p2',[220,40,300,300],'helmet_detected')]
        result=analyze_evidence(s,self.db)
        self.assertEqual([v['vehicle_id'] for v in result['potential_violations']],['m1'])
        self.assertEqual([v['summary']['supported_findings'] for v in result['vehicle_results']],[0,1,0])

    def test_blurry_image_withholds_all_findings(self):
        s=scene(riders=[rider('r1','v1','p1',[40,50,100,250])]);s['image_quality']['sufficient']=False
        self.assertEqual(analyze_evidence(s,self.db)['potential_violations'],[])
        self.assertFalse(assess_quality(np.full((480,640,3),128,np.uint8))['sufficient'])

    def test_plate_missing_obscured_and_outside_image_never_inferred_from_absence(self):
        for status in ('unreadable','visibility_not_established','outside_image','unavailable'):
            s=scene('car');s['plate_results']=[{'vehicle_id':'v1','status':status,'text':None}]
            result=analyze_evidence(s,self.db)
            self.assertEqual(result['potential_violations'],[])
            self.assertTrue(all(a['status']=='not_assessable' for a in result['assessments'] if 'plate' in a['type']))

    def test_competing_vehicle_classes_withhold_rules(self):
        resolved=resolve_vehicle_classes([box('v1','motorcycle',[20,20,400,400]),box('v2','car',[25,25,395,395],.94)])
        self.assertEqual(len(resolved),1);self.assertEqual(resolved[0]['classification_status'],'uncertain')
        s=scene(riders=[rider('r1','v1','p1',[40,50,100,250])]);s['vehicles']=resolved
        self.assertEqual(analyze_evidence(s,self.db)['potential_violations'],[])

    def test_other_jurisdictions_never_receive_punjab_amounts(self):
        result=analyze_evidence(scene(riders=[rider('r1','v1','p1',[40,50,100,250])]),self.db,'Sindh')
        self.assertIsNone(result['potential_violations'][0]['fine_amount'])
        self.assertEqual(result['summary']['verified_potential_fine'],0)


class AssociationAndOCRTests(unittest.TestCase):
    def test_car_containment_overrides_seated_pose_and_overlapping_bike(self):
        detections=[box('car','car',[50,50,400,350],.98),box('bike','motorcycle',[90,200,300,400],.55),box('driver','person',[130,100,270,300],.96)]
        points=[[0,0,0] for _ in range(17)];points[11]=[180,210,.95];points[13]=[210,240,.95];points[15]=[250,235,.95]
        self.assertEqual(associate_riders(detections,[{'bbox':detections[2]['bbox'],'points':points}]),[])

    def test_no_riders_skips_specialized_model(self):
        with patch('app.services.computer_vision.helmet_detector.predict') as inference:
            self.assertEqual(detect_helmets(textured_image(),[]),([],[]));inference.assert_not_called()

    def test_conflicting_or_shared_heads_are_withheld(self):
        r=rider('r1','v1','p1',[100,50,250,400])
        heads=[box('h1','with helmet',[150,60,200,110]),box('h2','without helmet',[150,60,200,110])]
        rs,accepted=match_heads(textured_image(),[r],heads)
        self.assertEqual(accepted,[]);self.assertEqual(rs[0]['helmet_status'],'unable_to_determine')
        rs,accepted=match_heads(textured_image(),[r],heads[1:],[box('p1','person',r['person_bbox']),box('p2','person',r['person_bbox'])])
        self.assertEqual(accepted,[])

    def test_head_without_clear_pixels_does_not_create_confidence(self):
        rs,accepted=match_heads(np.full((480,640,3),128,np.uint8),[rider('r1','v1','p1',[100,50,250,400])],[box('h1','without helmet',[150,60,200,110])])
        self.assertIsNone(rs[0]['helmet_confidence']);self.assertEqual(accepted,[])

    def test_below_class_threshold_helmet_box_does_not_override_clear_bare_head(self):
        r=rider('r1','v1','p1',[100,50,250,400])
        heads=[box('weak','with helmet',[150,60,200,110],.43),box('clear','without helmet',[150,60,200,110],.95)]
        rs,accepted=match_heads(textured_image(),[r],heads)
        self.assertEqual(rs[0]['helmet_status'],'possible_no_helmet')
        self.assertEqual(rs[0]['helmet_confidence'],.95)
        self.assertEqual(accepted[0]['id'],'clear')

    def test_bare_head_below_finding_threshold_is_explicitly_uncertain_without_fine(self):
        r=rider('r1','v1','p1',[100,50,250,400])
        rs,_=match_heads(textured_image(),[r],[box('weak','without helmet',[150,60,200,110],.69)])
        self.assertEqual(rs[0]['helmet_status'],'uncertain_no_helmet')
        result=analyze_evidence(scene(riders=rs),FineDatabase(DATA))
        self.assertEqual(result['potential_violations'],[])
        self.assertTrue(any(a['status']=='uncertain' for a in result['assessments']))
        self.assertEqual(result['summary']['verified_potential_fine'],0)

    def test_normalization_never_completes_or_replaces_characters(self):
        self.assertEqual(normalize_plate('PUNJAB\nLEA 18-1234'),'LEA181234')
        self.assertIsNone(normalize_plate('LEA 18 12?4'))
        self.assertEqual(normalize_plate('LEA O123'),'LEAO123')
        self.assertIsNone(normalize_plate('Punjab'))

    def test_ocr_failure_and_disagreement_return_unreadable(self):
        image=textured_image();plate=box('p','number plate',[100,100,300,170])
        with patch('app.services.computer_vision.plate_ocr.get_ocr',side_effect=RuntimeError('failure')):
            result=recognize_plate(image,plate);self.assertIsNone(result['text']);self.assertEqual(result['ocr_status'],'unavailable')
        polygon=[[0,0],[100,0],[100,30],[0,30]]
        reader=Mock(side_effect=[([[polygon,'ABC123',.99]],None),([[polygon,'ABC124',.99]],None)])
        with patch('app.services.computer_vision.plate_ocr.get_ocr',return_value=reader):
            self.assertIsNone(recognize_plate(image,plate)['text'])

    def test_ocr_low_confidence_and_partial_text_do_not_become_plate_numbers(self):
        polygon=[[0,0],[100,0],[100,30],[0,30]]
        for text,score in [('ABC123',.84),('ABC',.99),('123',.99),('ABC12?',.99)]:
            reader=Mock(return_value=([[polygon,text,score]],None))
            with patch('app.services.computer_vision.plate_ocr.get_ocr',return_value=reader):
                result=recognize_plate(textured_image(),box('p','number plate',[100,100,300,170]))
            self.assertIsNone(result['text']);self.assertEqual(result['ocr_status'],'unreadable')

    def test_plate_ocr_association_is_vehicle_specific(self):
        vehicles=[box('car','car',[0,0,300,480]),box('motorcycle','motorcycle',[350,0,640,480])]
        plates=[box('p1','number plate',[60,320,220,370]),box('p2','number plate',[390,330,550,380])]
        with patch('app.services.computer_vision.plate_detector.predict_plates',return_value=plates),patch('app.services.computer_vision.plate_detector.recognize_plate',side_effect=lambda image,b:{'text':'ABC123' if b['id']=='p1' else 'XYZ456','ocr_confidence':.96,'ocr_status':'recognized','ocr_reason':'Test OCR fixture'}):
            result,boxes=detect_plates(textured_image(),vehicles,[])
        self.assertEqual([(p['vehicle_id'],p['text']) for p in result],[('car','ABC123'),('motorcycle','XYZ456')])

    def test_ambiguous_plate_is_never_assigned_or_read(self):
        vehicles=[box('car1','car',[0,0,400,480]),box('car2','car',[50,50,450,480])]
        plates=[box('p1','number plate',[100,320,260,370])]
        with patch('app.services.computer_vision.plate_detector.predict_plates',return_value=plates),patch('app.services.computer_vision.plate_detector.recognize_plate') as ocr:
            results,_=detect_plates(textured_image(),vehicles,[])
        ocr.assert_not_called();self.assertTrue(all(r['text'] is None for r in results))


class LegalAndPDFTests(unittest.TestCase):
    def test_verified_amounts_and_unknown_car_capacity(self):
        db=FineDatabase(DATA);self.assertTrue(db.valid,db.error)
        row=db.lookup('helmet_violation',{'class':'motorcycle'});self.assertTrue(row['verified']);self.assertIsNone(row['effective_date'])
        seat=db.lookup('seatbelt_violation',{'class':'car'},conditions={'notified_road':True})
        self.assertIsNone(seat['fine_amount']);self.assertFalse(seat['verified'])
        seat=db.lookup('seatbelt_violation',{'class':'car','legal_vehicle_category':'motorcar_under_2000cc'})
        self.assertIsNone(seat['fine_amount'])

    def test_stale_snapshot_never_falls_back_to_index(self):
        db=FineDatabase(DATA,today=date.today()+timedelta(days=100))
        self.assertFalse(db.valid);self.assertIsNone(db.lookup('helmet_violation',{'class':'motorcycle'})['fine_amount'])

    def test_tampered_fine_or_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            from shutil import copy2
            destination=Path(folder)/'data';(destination/'raw/documents').mkdir(parents=True)
            (destination/'raw/structured').mkdir(parents=True)
            original=json.loads((DATA/'raw/structured/vision_fines.json').read_text(encoding='utf-8'))
            for name in [original['snapshot_file'],original['occupancy_rule']['snapshot_file']]:
                copy2(DATA/'raw/documents'/name,destination/'raw/documents'/name)
            original['records'][18]['amounts']['motorcycle']+=1
            (destination/'raw/structured/vision_fines.json').write_text(json.dumps(original),encoding='utf-8')
            self.assertFalse(FineDatabase(destination).valid)

    def test_total_excludes_unverified_and_duplicate_findings(self):
        first={'vehicle_id':'v1','type':'helmet_violation','status':'potential_violation','fine_amount':2000,'fine_verified':True}
        unknown={**first,'type':'other','fine_verified':False,'fine_amount':100000}
        result=fine_summary([first,first,unknown],[])
        self.assertEqual(result['verified_potential_fine'],2000);self.assertEqual(result['unverified_penalties'],1)

    def test_pdf_single_subject_multiple_violations_and_no_violation(self):
        rs=[rider(f'r{i}','v1',f'p{i}',[40+140*i,40,130+140*i,300]) for i in range(3)]
        for positive in (True,False):
            s=scene(riders=rs if positive else []);s['vehicles'].append(box('car','car',[100,100,300,300]))
            result=report_result(s);pdf=build_report(result);reader=PdfReader(io.BytesIO(pdf));text='\n'.join(p.extract_text() for p in reader.pages)
            self.assertIn('Motorcycle',text);self.assertNotIn('Car',text)
            self.assertIn('not official traffic challans',' '.join(text.split()));self.assertEqual(len(reader.pages),1)
            if positive:
                self.assertIn('Potential helmet violation',text);self.assertIn('passenger-limit',text)
                self.assertIn(f"PKR {result['summary']['verified_potential_fine']:,}",text)
            else:self.assertIn('No supported potential violation',text)

    def test_foreground_ignores_detector_order_and_probability_as_distance(self):
        car=box('car','car',[10,10,80,50],.99)
        bike=box('bike','motorcycle',[100,130,590,470],.75)
        self.assertEqual(foreground_vehicle([car,bike],{'width':640,'height':480})['id'],'bike')
        self.assertIsNone(foreground_vehicle([],{'width':640,'height':480}))

    def test_subject_crop_includes_rider_head_but_excludes_unrelated_people(self):
        bike=box('bike','motorcycle',[100,220,500,450])
        rs=[rider('r1','bike','p1',[120,40,300,430]),rider('r2','other','p2',[0,0,640,480])]
        bounds=subject_bounds(bike,rs,(480,640,3))
        self.assertLess(bounds[1],40);self.assertGreater(bounds[3],450)
        self.assertGreater(bounds[0],0)

    def test_report_subject_override_has_independent_fine_and_source_totals(self):
        s=scene(riders=[rider('r','v1','p',[80,20,200,330])]);s['vehicles'].append(box('car','car',[100,100,300,300]))
        result=report_result(s)
        default=report_scope(result)
        self.assertEqual(default['summary']['verified_potential_fine'],2000)
        car=report_scope(result,'car')
        self.assertEqual(car['summary']['verified_potential_fine'],0);self.assertEqual(car['sources'],[])
        text='\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(build_report(result,'car'))).pages)
        self.assertIn('Car',text);self.assertNotIn('Potential helmet violation',text)
        with self.assertRaises(ValueError):report_scope(result,'not-in-image')

    def test_server_evidence_snapshot_is_immutable_and_memory_bounded(self):
        cache=ReportStore(max_bytes=200)
        evidence={'vehicle_results':[{'vehicle_id':'original'}]}
        cache.put('r',b'pdf',evidence)
        evidence['vehicle_results'][0]['vehicle_id']='client-changed'
        self.assertEqual(cache.evidence('r')['vehicle_results'][0]['vehicle_id'],'original')
        with self.assertRaises(RuntimeError):cache.put('oversize',b'pdf',{'image':'x'*300})

    def test_pdf_escapes_plate_markup_and_omits_unrelated_technical_metadata(self):
        result=report_result(scene('car'));result['filename']='unrelated-file.png';result['warnings']=['Long evidence '*90]
        result['vehicle_results'][0]['plate']={'status':'recognized','text':'<b>untrusted</b>'}
        text='\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(build_report(result))).pages)
        self.assertIn('<b>untrusted</b>',text)
        for unwanted in ('unrelated-file.png','Pipeline version','Image quality','model versions','Long evidence','Other assessments'):
            self.assertNotIn(unwanted,text)

    def test_pdf_uses_original_subject_photo_without_annotations(self):
        result=report_result(scene('car'))
        original=np.zeros((240,240,3),dtype=np.uint8);original[:,:,2]=210
        annotated=np.zeros_like(original);annotated[:,:,1]=210
        result['vehicle_results'][0]['image_url']=image_url(original)
        result['vehicle_results'][0]['focus_annotated_image_url']=image_url(annotated)
        pdf=PdfReader(io.BytesIO(build_report(result)))
        self.assertEqual(len(pdf.pages[0].images),1)
        pixels=np.array(pdf.pages[0].images[0].image.convert('RGB'))
        self.assertGreater(float(pixels[:,:,0].mean()),200)
        self.assertLess(float(pixels[:,:,1].mean()),10)

    def test_pdf_displays_analysis_time_in_pakistan_and_withholds_unconfirmed_plate(self):
        result=report_result(scene('motorcycle'))
        result['vehicle_results'][0]['plate']={'status':'unreadable','text':'PARTIAL123'}
        text='\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(build_report(result))).pages)
        self.assertIn('11 Oct 2026, 05:00 AM PKT',text)
        self.assertIn('Number plate not readable.',text);self.assertNotIn('PARTIAL123',text)
        self.assertEqual(local_analysis_time('2026-10-11T00:00:00'),'Unavailable')
        self.assertEqual(local_analysis_time(None),'Unavailable')

    def test_pdf_long_violation_detail_wraps_and_continues_across_pages(self):
        result=report_result(scene(riders=[rider('r','v1','p',[80,20,200,330])]))
        finding=result['vehicle_results'][0]['violations'][0]
        finding['type']='future_supported_category';finding['visual_evidence']=['Observed detail '*350]
        pdf=PdfReader(io.BytesIO(build_report(result)))
        text='\n'.join(p.extract_text() for p in pdf.pages)
        self.assertIn('Observed detail',text);self.assertIn('Verified potential total: PKR 2,000',text)
        self.assertGreater(len(pdf.pages),1)

    def test_report_cache_capacity_and_expiry(self):
        cache=ReportStore(ttl=1,max_reports=1,max_bytes=10)
        cache.put('a',b'123');cache.put('b',b'456');self.assertIsNone(cache.get('a'));self.assertEqual(cache.get('b'),b'456')
        with patch('app.services.computer_vision.report_store.time.monotonic',return_value=10**12):self.assertIsNone(cache.get('b'))


class APIContractTests(unittest.TestCase):
    def setUp(self):
        app=FastAPI();app.include_router(router,prefix='/api/v1')
        app.state.settings=SimpleNamespace(max_upload_mb=1,data_dir=DATA)
        app.state.repository=None;app.state.rag=None
        self.client=TestClient(app)

    def upload(self,content,mime='image/png',name='road.png',jurisdiction='Punjab'):
        return self.client.post('/api/v1/vision/analyze',files={'file':(name,content,mime)},data={'jurisdiction':jurisdiction})

    def test_empty_unsupported_corrupt_and_wrong_format_uploads(self):
        self.assertEqual(self.upload(b'').status_code,400)
        self.assertEqual(self.upload(b'gif','image/gif').status_code,415)
        self.assertEqual(self.upload(b'not an image').status_code,400)
        data=io.BytesIO();Image.new('RGB',(64,64)).save(data,format='JPEG')
        self.assertEqual(self.upload(data.getvalue()).status_code,400)

    def test_size_and_dimension_validation(self):
        self.assertEqual(self.upload(b'x'*(1024*1024+1)).status_code,413)
        data=io.BytesIO();Image.new('RGB',(10,64)).save(data,format='PNG')
        self.assertEqual(self.upload(data.getvalue()).status_code,400)
        self.assertEqual(self.upload(data.getvalue(),jurisdiction='invented').status_code,422)

    def test_report_id_and_expiry_validation(self):
        self.assertEqual(self.client.get('/api/v1/vision/reports/invalid.pdf').status_code,404)
        self.assertEqual(self.client.get('/api/v1/vision/reports/'+'0'*32+'.pdf').status_code,404)

    def test_api_normalization_preserves_decoded_jpeg_pixels(self):
        data=io.BytesIO();Image.fromarray(textured_image()).save(data,format='JPEG',quality=84)
        with Image.open(io.BytesIO(data.getvalue())) as im:expected=np.array(im.convert('RGB'))[:,:,::-1]
        observed=[]
        def capture(image):
            observed.append(image.copy());return [],'Test fixture checks preprocessing only'
        with patch('app.services.computer_vision.violation_analyzer.detect_objects',side_effect=capture),patch('app.services.computer_vision.violation_analyzer.detect_plates',return_value=([],[])):
            response=self.upload(data.getvalue(),'image/jpeg','road.jpg')
        self.assertEqual(response.status_code,200,response.text)
        np.testing.assert_array_equal(observed[0],expected)

    def test_no_vehicles_pipeline_and_pdf_download(self):
        data=io.BytesIO();Image.fromarray(textured_image()).save(data,format='PNG')
        with patch('app.services.computer_vision.violation_analyzer.detect_objects',return_value=([],'Test fixture: no detections')),patch('app.services.computer_vision.violation_analyzer.detect_plates',return_value=([],[])):
            response=self.upload(data.getvalue())
        self.assertEqual(response.status_code,200,response.text)
        result=response.json();self.assertEqual(result['potential_violations'],[])
        report=self.client.get('/api/v1'+result['report_url']);self.assertEqual(report.status_code,200)
        self.assertEqual(report.headers['content-type'],'application/pdf');self.assertTrue(report.content.startswith(b'%PDF'))

    def test_car_driver_helmet_bug_at_full_inference_pipeline_boundary(self):
        detections=[box('car','car',[50,50,400,350],.98),box('bike','motorcycle',[90,200,300,400],.55),box('driver','person',[130,100,270,300],.96)]
        points=[[0,0,0] for _ in range(17)];points[11]=[180,210,.95];points[13]=[210,240,.95];points[15]=[250,235,.95]
        pose=SimpleNamespace(keypoints=SimpleNamespace(data=SimpleNamespace(tolist=lambda:[points])))
        data=io.BytesIO();Image.fromarray(textured_image()).save(data,format='PNG')
        with patch('app.services.computer_vision.violation_analyzer.detect_objects',return_value=(detections,'Test fixture regression')),patch('app.services.computer_vision.violation_analyzer.predict',return_value=([box('pose','person',detections[2]['bbox'])],pose)),patch('app.services.computer_vision.helmet_detector.predict') as head_inference,patch('app.services.computer_vision.violation_analyzer.detect_plates',return_value=([],[])):
            response=self.upload(data.getvalue())
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['riders'],[]);self.assertEqual(response.json()['potential_violations'],[])
        head_inference.assert_not_called()


class RealModelSmokeTests(unittest.TestCase):
    """Real model inference; no detector replacement in these cases."""
    def test_real_saved_upload_api_detects_bare_head_and_pdf_fine_on_foreground_bike(self):
        path=ROOT/'backend/storage/uploads/bdd82670bec54f0eaab14c8ea13e7736.jpg'
        if not path.is_file():self.skipTest('Existing private user upload unavailable')
        app=FastAPI();app.include_router(router,prefix='/api/v1')
        app.state.settings=SimpleNamespace(max_upload_mb=8,data_dir=DATA);app.state.repository=None;app.state.rag=None
        client=TestClient(app)
        response=client.post('/api/v1/vision/analyze',files={'file':('rider.jpg',path.read_bytes(),'image/jpeg')})
        self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        subject=next(v for v in result['vehicle_results'] if v['vehicle_id']==result['primary_vehicle_id'])
        self.assertEqual(subject['vehicle_type'],'motorcycle')
        self.assertTrue(any(v['type']=='helmet_violation' for v in subject['violations']))
        self.assertEqual(subject['summary']['verified_potential_fine'],2000)
        self.assertEqual(subject['plate']['status'],'recognized')
        self.assertEqual(subject['plate']['text'],'SGK3695') # Manually reviewed real photo label; never a production fallback.
        self.assertGreaterEqual(subject['plate']['ocr_confidence'],.85)
        pdf=client.get('/api/v1'+result['report_url'])
        pages=PdfReader(io.BytesIO(pdf.content)).pages
        text='\n'.join(p.extract_text() for p in pages)
        self.assertIn('Potential helmet violation',text);self.assertIn('PKR 2,000',text)
        self.assertIn('SGK3695',text);self.assertEqual(len(pages),1)
        self.assertNotIn('Car',text)
        car=next(v for v in result['vehicle_results'] if v['vehicle_type']=='car')
        alternate=client.get('/api/v1'+result['report_url'],params={'vehicle_id':car['vehicle_id']})
        car_text='\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(alternate.content)).pages)
        self.assertIn('Car',car_text);self.assertNotIn('Potential helmet violation',car_text)
        self.assertNotIn('SGK3695',car_text)
        self.assertEqual(client.get('/api/v1'+result['report_url'],params={'vehicle_id':'fake'}).status_code,404)

    def test_real_foreground_rider_pose_refinement_recovers_crowd_finding(self):
        path=ROOT/'backend/storage/verification/motorcycle-riders.jpg'
        if not path.is_file():self.skipTest('Licensed fixture unavailable')
        result=finish_analysis(visual_analysis(path),SimpleNamespace(data_dir=DATA),'Punjab')
        subject=next(v for v in result['vehicle_results'] if v['vehicle_id']==result['primary_vehicle_id'])
        self.assertTrue(any(r['association']=='supported' and 'person crop' in r['association_evidence'] for r in subject['riders']))
        self.assertTrue(any(v['type']=='helmet_violation' for v in subject['violations']))
    def test_real_ocr_on_printed_latin_plate_pixels(self):
        # A generated typography fixture tests OCR execution, not real plate accuracy.
        from PIL import ImageDraw,ImageFont
        import reportlab
        image=Image.new('RGB',(700,240),(210,210,210));draw=ImageDraw.Draw(image)
        draw.rectangle((50,65,650,180),fill=(245,245,245))
        font=ImageFont.truetype(str(Path(reportlab.__file__).parent/'fonts/VeraBd.ttf'),60)
        draw.text((70,78),'LEA181234',font=font,fill=(10,10,10))
        result=recognize_plate(np.array(image)[:,:,::-1].copy(),{'bbox':[50,65,650,180]})
        self.assertEqual(result['text'],'LEA181234',result)

    def test_real_bus_image_has_no_helmet_observations_or_violations(self):
        path=ROOT/'backend/storage/verification/bus.jpg'
        if not path.is_file():self.skipTest('Licensed fixture not provisioned')
        result=finish_analysis(visual_analysis(path),SimpleNamespace(data_dir=DATA),'Punjab')
        self.assertTrue(any(v['class']=='bus' for v in result['vehicles']))
        self.assertEqual(result['helmet_detections'],[]);self.assertEqual(result['potential_violations'],[])

    def test_real_mixed_car_motorcycle_photo_does_not_cross_assign_helmet_findings(self):
        path=ROOT/'backend/storage/verification/helmeted-couple.jpg'
        if not path.is_file():self.skipTest('Licensed fixture not provisioned')
        result=finish_analysis(visual_analysis(path),SimpleNamespace(data_dir=DATA),'Punjab')
        types={v['id']:v['class'] for v in result['vehicles']}
        self.assertIn('car',types.values());self.assertIn('motorcycle',types.values())
        self.assertTrue(result['helmet_detections'])
        self.assertTrue(all(types[h['vehicle_id']]=='motorcycle' for h in result['helmet_detections']))
        self.assertEqual(result['potential_violations'],[])


if __name__=='__main__':unittest.main()
