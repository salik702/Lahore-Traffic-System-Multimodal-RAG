import asyncio
import base64
import re
import uuid
from datetime import date
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.schemas.api import ChatRequest, ChatResponse, SearchResponse
from app.services.router import detect_intent


router = APIRouter()
ALLOWED_UPLOADS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


@router.get("/health")
async def health(request: Request):
    rag = request.app.state.rag
    settings = request.app.state.settings
    return {
        "status": "ok", "rag_anything_ready": rag.ready, "rag_error": rag.error,
        "groq_configured": bool(settings.groq_api_key), "model": settings.groq_model,
        "knowledge_indexed": request.app.state.repository.knowledge_status()["indexed"],
        "image_embeddings": request.app.state.repository.image_index.status(),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    return await request.app.state.chat.ask(payload.message, payload.session_id, payload.top_k)


@router.get("/search", response_model=SearchResponse)
async def search(request: Request, q: str = Query(min_length=2), top_k: int = Query(6, ge=1, le=20)):
    import asyncio
    repository = request.app.state.repository
    results = await asyncio.to_thread(repository.search, q, top_k * 3)
    return {"query": q, "results": (await asyncio.to_thread(repository.rank_image_evidence, q, results, detect_intent(q)))[:top_k]}


@router.get("/rules")
async def rules(request: Request, q: str = "", category: str = ""):
    return {"items": request.app.state.repository.list_rules(q, category)}


@router.get("/fines")
async def fines(request: Request, q: str = ""):
    return {"items": request.app.state.repository.list_fines(q), "currency": "PKR", "source_id": "punjab-fines-2026"}


@router.get("/signs")
async def signs(request: Request, q: str = "", category: str = ""):
    # Functional/visual fallback may use local encoders; keep other requests responsive.
    return {"items": await asyncio.to_thread(request.app.state.repository.list_signs, q, category)}


@router.get("/sources")
async def sources(request: Request):
    return {"items": request.app.state.repository.sources}


@router.get("/documents")
async def documents(request: Request):
    return {"items": [source for source in request.app.state.repository.sources if "PDF" in source["data_type"]]}


@router.get("/statistics")
async def statistics(request: Request):
    return request.app.state.repository.statistics()


@router.get("/accidents")
async def accidents(request: Request, q: str = "", city: str = "", province: str = "", accident_type: str = "", vehicle_type: str = "", has_image: bool | None = None, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), date_from: date | None = None, date_to: date | None = None):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "Start date must precede end date")
    semantic_ids = []
    if q and request.app.state.rag.ready:
        try:
            semantic_ids = [hit.get("record_id") or hit["id"] for hit in await request.app.state.rag.retrieve_accidents(q, 60)]
        except Exception:
            pass  # Structured factual filters remain available during index downtime.
    items, total = request.app.state.repository.search_accidents(q, city=city, province=province, accident_type=accident_type, vehicle_type=vehicle_type, has_image=has_image, limit=limit, offset=offset, semantic_ids=semantic_ids, date_from=date_from.isoformat() if date_from else None, date_to=date_to.isoformat() if date_to else None)
    return {"items": items, "total": total, "limit": limit, "offset": offset, "last_indexed_date": request.app.state.repository.statistics()["latest_indexed_accident"]}


@router.get("/accidents/search")
async def accident_search(request: Request, q: str = Query(min_length=2), limit: int = Query(12, ge=1, le=50)):
    return await accidents(request, q=q, limit=limit, offset=0)


@router.get("/accidents/images")
async def accident_images(request: Request, q: str = "", limit: int = Query(12, ge=1, le=40)):
    semantic_ids = []
    if q and request.app.state.rag.ready:
        try:
            semantic_ids = [hit.get("record_id") or hit["id"] for hit in await request.app.state.rag.retrieve_accidents(q, 60)]
        except Exception:
            pass
    items = request.app.state.repository.list_accident_images(q, limit, semantic_ids)
    return {"query": q, "items": items, "total": len(items)}


@router.get("/accidents/facets")
async def accident_facets(request: Request):
    return request.app.state.repository.accident_facets()


@router.get("/accidents/statistics")
async def accident_statistics(request: Request):
    return {"items": request.app.state.repository.accident_statistics, "type": "accident_statistics"}


@router.get("/accidents/{record_id}")
async def accident_detail(record_id: str, request: Request):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", record_id):
        raise HTTPException(400, "Invalid accident id")
    item = request.app.state.repository.get_accident(record_id)
    if not item:
        raise HTTPException(404, "Accident record not found")
    return item


@router.get("/knowledge-base/status")
async def knowledge_base_status(request: Request):
    return request.app.state.repository.knowledge_status()


@router.get("/sessions/{session_id}")
async def session_history(session_id: str, request: Request):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
        raise HTTPException(400, "Invalid session id")
    return {"session_id": session_id, "messages": request.app.state.store.history(session_id)}


@router.post("/visual-search")
async def visual_search(request: Request, file: UploadFile = File(...), question: str = "Explain this traffic sign"):
    settings = request.app.state.settings
    if file.content_type not in ALLOWED_UPLOADS:
        raise HTTPException(415, "Only JPEG, PNG and WebP images are accepted")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB")
    try:
        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(content)); image.verify()
    except Exception as exc:
        raise HTTPException(400, "The uploaded file is not a valid image") from exc

    suffix = ALLOWED_UPLOADS[file.content_type]
    safe_path = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    safe_path.write_bytes(content)
    try:
        import asyncio
        import logging
        from app.services.evidence import INTERNAL, INJECTION
        analysis = None
        engine = "visual metadata fallback"
        if request.app.state.rag.ready:
            try:
                analysis = await asyncio.wait_for(request.app.state.rag.query_image(question, safe_path), timeout=20)
                engine = "RAG-Anything multimodal"
            except Exception:
                logging.getLogger(__name__).warning("visual_rag_unavailable", exc_info=True)
                analysis = None
        if analysis and (INTERNAL.search(analysis) or INJECTION.search(analysis)):
            analysis = None
        if not analysis and settings.groq_api_key:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=settings.groq_api_key, timeout=15, max_retries=0)
            encoded = base64.b64encode(content).decode("ascii")
            try:
                response = await client.chat.completions.create(
                    model=settings.groq_vision_model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Identify this road sign conservatively. Describe visible shape, color, symbol and likely meaning. If uncertain, say so. Never infer location or legal penalties from appearance."},
                        {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{encoded}"}},
                    ]}], temperature=0.1, max_completion_tokens=500,
                )
                analysis = response.choices[0].message.content
                if analysis and (INTERNAL.search(analysis) or INJECTION.search(analysis)):
                    analysis = None
                if analysis:
                    engine = "Groq vision + verified sign catalogue"
            except Exception:
                logging.getLogger(__name__).warning("visual_provider_unavailable", exc_info=True)
    
        from app.services.repository import tokens
        descriptive = tokens(question) - {"explain", "traffic", "road", "sign", "signs", "mean"}
        candidates = request.app.state.repository.list_signs(question)[:6] if descriptive else []
        if analysis:
            candidates = request.app.state.repository.list_signs(analysis)[:6]
        else:
            engine = "visual analysis unavailable"
        return {
            "analysis": analysis or "I could not verify the sign from this image because visual analysis is unavailable. Describe the shape, colour and symbol, or try again later. Any catalogue suggestions are text matches, not a visual identification.",
            "intent": detect_intent(question, has_image=True), "engine": engine,
            "related_signs": candidates,
            "trace": [
                {"label": "Secure upload", "detail": f"Validated {file.content_type}; {len(content)} bytes"},
                {"label": "Multimodal analysis", "detail": engine, "status": "complete" if analysis else "fallback"},
                {"label": "Catalogue retrieval", "detail": f"{len(candidates)} official NHMP sign candidates"},
            ],
        }
    finally:
        safe_path.unlink(missing_ok=True)


# Bound the entire CV request so queued uploads cannot exhaust CPU or memory.
import asyncio
import logging
_vision_slot = asyncio.Semaphore(1)


@router.post("/vision/analyze")
async def vision_analyze(request: Request, file: UploadFile = File(...), jurisdiction: str = Form('Punjab')):
    from io import BytesIO
    from PIL import Image, ImageOps
    from fastapi.responses import JSONResponse
    from tempfile import TemporaryDirectory
    from datetime import datetime,timezone
    import time
    from app.services.computer_vision.violation_analyzer import analyze
    from app.services.computer_vision.pdf_report import build_report
    from app.services.computer_vision.report_store import REPORTS
    started=time.perf_counter()
    settings = request.app.state.settings
    if jurisdiction not in {'Punjab','Sindh','Khyber Pakhtunkhwa','Balochistan','Islamabad','National highways','Unknown'}:
        await file.close()
        raise HTTPException(422,'Unsupported jurisdiction selection')
    if file.content_type not in ALLOWED_UPLOADS:
        await file.close()
        raise HTTPException(415, "Only JPEG, PNG and WebP images are accepted")
    try:
        content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    finally:
        await file.close()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB")
    try:
        image = Image.open(BytesIO(content))
        expected_format = {"image/jpeg":"JPEG", "image/png":"PNG", "image/webp":"WEBP"}[file.content_type]
        if image.format != expected_format or image.width * image.height > 20_000_000 or min(image.size)<32:
            raise ValueError("Unsupported image dimensions or format")
        image.verify()
        with Image.open(BytesIO(content)) as decoded:
            normalized = ImageOps.exif_transpose(decoded).convert("RGB")
            input_size={'width':normalized.width,'height':normalized.height}
            normalized.thumbnail((1920,1920))
    except Exception as exc:
        raise HTTPException(400, "Choose a valid JPEG, PNG or WebP image, at least 32 pixels wide and high, up to 20 megapixels") from exc
    try:
        await asyncio.wait_for(_vision_slot.acquire(), timeout=5)
    except asyncio.TimeoutError as exc:
        raise HTTPException(429,"Another image is being analyzed. Please retry shortly.") from exc
    try:
        # Temporary files have no static route. Wait for synchronous inference to finish
        # before cleanup, including when a client disconnects/cancels its request.
        with TemporaryDirectory(prefix="traffic-vision-") as folder:
            # Recompressing a JPEG changed clear bare-head scores from .77 to .69
            # in the reproduced upload. Preserve decoded pixels after orientation.
            path = Path(folder) / "image.png"
            normalized.save(path,format='PNG',compress_level=1)
            async def complete():
                result=await analyze(path,request.app.state.repository,request.app.state.rag,settings,jurisdiction)
                result.update(report_id=uuid.uuid4().hex,analyzed_at=datetime.now(timezone.utc).isoformat(),
                    filename=(file.filename or 'uploaded-image').replace('\\','/').rsplit('/',1)[-1][:180],
                    analysis_status='Completed with limitations' if result['warnings'] or any(c['status']=='unsupported' for c in result['capabilities']) else 'Completed',
                    input_size=input_size)
                result['timings']['analysis_ms']=round((time.perf_counter()-started)*1000,1)
                pdf_started=time.perf_counter()
                pdf=await asyncio.to_thread(build_report,result)
                REPORTS.put(result['report_id'],pdf,result)
                result['timings']['pdf_ms']=round((time.perf_counter()-pdf_started)*1000,1)
                result['timings']['request_ms']=round((time.perf_counter()-started)*1000,1)
                result['report_url']=f"/vision/reports/{result['report_id']}.pdf"
                result['report_expires_in_seconds']=REPORTS.ttl
                return result
            task = asyncio.create_task(complete())
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
        return JSONResponse(result,headers={"Cache-Control":"no-store, private","Pragma":"no-cache","X-Content-Type-Options":"nosniff"})
    except RuntimeError as exc:
        logging.getLogger(__name__).exception("Vision model failed")
        raise HTTPException(503,"Computer vision model unavailable or inference failed. Check backend model setup.") from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Vision analysis failed")
        raise HTTPException(500,"Image analysis failed. Please try another image or check backend logs.") from exc
    finally:
        _vision_slot.release()


@router.get('/vision/capabilities')
async def vision_capabilities(request: Request):
    from app.services.computer_vision.policy import capabilities,policy
    return {'capabilities':capabilities(),'thresholds':policy(),
            'max_upload_mb':request.app.state.settings.max_upload_mb,
            'max_megapixels':20,'formats':['image/jpeg','image/png','image/webp'],
            'calibration':'Conservative thresholds; no Pakistan-specific statistical calibration.'}


@router.get('/vision/reports/{report_id}.pdf')
async def vision_report(report_id: str,vehicle_id: str | None = Query(None,max_length=80)):
    from fastapi.responses import Response
    from app.services.computer_vision.report_store import REPORTS
    if not re.fullmatch(r'[0-9a-f]{32}',report_id):raise HTTPException(404,'Report unavailable')
    content=REPORTS.get(report_id)
    if content is None:raise HTTPException(404,'Report expired or unavailable. Analyze the image again to generate a new report.')
    if vehicle_id is not None:
        evidence=REPORTS.evidence(report_id)
        if evidence is None:raise HTTPException(404,'Report evidence expired. Analyze the image again.')
        if not any(v['vehicle_id']==vehicle_id for v in evidence['vehicle_results']):
            raise HTTPException(404,'This vehicle does not belong to the analyzed image')
        from app.services.computer_vision.pdf_report import build_report
        content=await asyncio.to_thread(build_report,evidence,vehicle_id)
    return Response(content,media_type='application/pdf',headers={
        'Content-Disposition':f'attachment; filename="LTI-violation-report-{report_id[:12]}.pdf"',
        'Cache-Control':'no-store, private','Pragma':'no-cache','X-Content-Type-Options':'nosniff'})
