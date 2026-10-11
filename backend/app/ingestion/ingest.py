import asyncio
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from app.core.config import get_settings
from app.rag.engine import RAGAnythingEngine


def pdf_content(path: Path) -> list[dict]:
    document = pymupdf.open(path)
    return [{"type": "text", "text": page.get_text("text"), "page_idx": index} for index, page in enumerate(document) if page.get_text("text").strip()]


def structured_content(settings) -> list[dict]:
    base = settings.data_dir / "raw" / "structured"
    rules = json.loads((base / "rules.json").read_text(encoding="utf-8"))
    fines = json.loads((base / "fines.json").read_text(encoding="utf-8"))
    signs = json.loads((base / "signs.json").read_text(encoding="utf-8"))
    content = [{"type": "text", "text": f"{r['title']}\n{r['summary']}\nSource section: {r['section']}; page {r['page']}; region {r['region']}", "page_idx": r["page"]} for r in rules]
    headers = ["Code", "Violation", "Motorcycle", "Three wheeler", "Car under 2000cc", "Car over 2000cc", "Transport vehicle"]
    rows = [[str(f.get(k, "")) for k in ("code", "violation", "motorcycle", "three_wheeler", "motorcar_under_2000cc", "motorcar_over_2000cc", "transport_vehicle")] for f in fines]
    table = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    content.append({"type": "table", "table_body": table, "table_caption": ["Official Traffic Police Punjab offences and penalties by vehicle class"], "table_footnote": ["Amounts are PKR; blank means not listed."], "page_idx": 0})
    indexed_images = {"traffic-lights", "pedestrian-crossing", "no-entry", "speed-limit-50", "no-parking"}
    for sign in signs:
        if sign["id"] not in indexed_images:
            continue
        image_path = settings.data_dir / "raw" / "traffic_signs" / sign["image"]
        if image_path.exists():
            content.append({"type": "image", "img_path": str(image_path.resolve()), "image_caption": [f"{sign['name']}: {sign['meaning']}"], "image_footnote": [f"Category: {sign['category']}; source: {sign.get('source_id', 'NHMP road safety syllabus')}"], "page_idx": sign.get("source_page", 37)})
    return content


async def ingest() -> None:
    settings = get_settings()
    worker = None
    with socket.socket() as probe:
        worker_running = probe.connect_ex(("127.0.0.1", 8765)) == 0
    if not worker_running:
        worker_env = os.environ.copy()
        worker_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        worker = subprocess.Popen([settings.semantic_python_executable, str(settings.project_root / "backend" / "embedding_worker.py")], cwd=settings.project_root / "backend", env=worker_env, creationflags=flags)
        # On this Windows/OneDrive workstation a cold sentence-transformers import
        # takes about three minutes, so allow a bounded six-minute warm-up.
        for _ in range(360):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", 8765)) == 0:
                    break
            time.sleep(1)
        else:
            worker.terminate()
            raise RuntimeError("Local semantic embedding worker did not become ready")
    engine = RAGAnythingEngine(settings)
    await engine.initialize()
    if not engine.ready:
        raise RuntimeError(engine.error or "RAG-Anything could not initialize")
    documents = sorted((settings.data_dir / "raw" / "documents").glob("*.pdf"))
    catalogue_path = settings.data_dir / "raw" / "structured" / "rules.json"
    rules = json.loads((settings.data_dir / "raw" / "structured" / "rules.json").read_text(encoding="utf-8"))
    fines = json.loads((settings.data_dir / "raw" / "structured" / "fines.json").read_text(encoding="utf-8"))
    signs = json.loads((settings.data_dir / "raw" / "structured" / "signs.json").read_text(encoding="utf-8"))
    accident_records = json.loads((settings.data_dir / "processed" / "accidents" / "records.json").read_text(encoding="utf-8"))
    accident_images = json.loads((settings.data_dir / "processed" / "accidents" / "image_metadata.json").read_text(encoding="utf-8"))
    accident_statistics = json.loads((settings.data_dir / "processed" / "accidents" / "statistics" / "rescue1122_districts.json").read_text(encoding="utf-8"))
    chunks = [f"RULE | {item['title']} | {item['summary']} | {item['section']} | page {item['page']} | source {item['source_id']}" for item in rules]
    chunks += [f"FINE | {item['violation']} | code {item['code']} | motorcycle {item.get('motorcycle')} | three wheeler {item.get('three_wheeler')} | car under 2000cc {item.get('motorcar_under_2000cc')} | car over 2000cc {item.get('motorcar_over_2000cc')} | transport {item.get('transport_vehicle')} | PKR | source Traffic Police Punjab" for item in fines]
    chunks += [f"IMAGE | {item['name']} | {item['meaning']} | category {item['category']} | file {item['image']} | source {item.get('source_id', 'nhmp-road-safety-syllabus')}" for item in signs]
    for item in accident_records:
        fields = [
            item["id"], item["title"], item["description"], f"date {item.get('date')}",
            f"location {item.get('location')}", f"city {item.get('city')}", f"province {item.get('province')}",
            f"vehicles {', '.join(item.get('vehicle_types') or []) or 'not recorded'}",
            f"severity {item.get('severity')}", f"source {item['source_id']}",
            f"verified image {item.get('image_verified_for_record', False)}",
        ]
        if item.get("photographed_at"):
            fields.append(f"photograph date {item['photographed_at']}; incident date unknown")
        chunks.append("ACCIDENT REPORT | " + " | ".join(str(value) for value in fields))
    chunks += [f"ACCIDENT STATISTICS | district {item['district']} | road accidents {item['road_accidents']} | reporting period ending {item.get('reporting_period_end')} | source rescue1122-punjab-performance" for item in accident_statistics]
    print(f"Seeding RAG-Anything/LightRAG semantic store with {len(chunks)} verified chunks")
    chunks += [f"ACCIDENT IMAGE | {image['id']} | record {image.get('record_id')} | {image['title']} | {image['description']} | city {image.get('city')} | province {image.get('province')} | country basis {image.get('geography_basis', 'source-supported location')} | verified record match {image.get('verified_record_match', False)} | source {image['source_id']} | source URL {image['source_url']} | license {image['license']}" for image in accident_images]
    failures = []
    completed = []
    try:
        await engine.seed_vector_chunks(chunks, "verified-semantic-corpus-v3-accidents")
        completed.append({"doc_id": "verified-semantic-corpus-v3-accidents", "file": "data/processed/accidents/rag_documents/accidents.jsonl", "blocks": len(chunks)})
    except Exception as exc:
        failures.append({"doc_id": "verified-semantic-corpus-v3-accidents", "error": f"{type(exc).__name__}: {exc}"})
    image_content = []
    try:
        await engine.ingest_accident_provenance(accident_records, accident_images)
        completed.append({"doc_id": "verified-accident-provenance-v1", "file": "data/processed/accidents/image_metadata.json", "blocks": len(accident_images)})
    except Exception as exc:
        failures.append({"doc_id": "verified-accident-provenance-v1", "error": f"{type(exc).__name__}: {exc}"})
    for image in accident_images:
        image_path = settings.project_root / image["image_path"]
        if image_path.is_file():
            caption = f"Image ID: {image['id']}. {image['title']}. {image['description']} Location: {image.get('city') or image.get('district') or image.get('province') or 'unknown'}. Geography basis: {image.get('geography_basis', 'source-supported location')}. Photograph date (not independently verified incident date): {image.get('date')}. Accident type: {image.get('accident_type')}. Record: {image.get('record_id')}. Verified record-image match: {image.get('verified_record_match', False)}. Source: {image['source_url']}. License: {image['license']}."
            image_content.append({"type": "image", "img_path": str(image_path.resolve()), "image_caption": [caption], "image_footnote": [f"Source ID: {image['source_id']}; author: {image['author']}"], "page_idx": 0})
    if image_content:
        try:
            revision = hashlib.sha256(("source-grounded-photo-date-policy-v3-verified-graph\n" + json.dumps(accident_images, sort_keys=True)).encode()).hexdigest()[:12]
            await engine.ingest_content(image_content, settings.data_dir / "processed" / "accidents" / "image_metadata.json", f"pakistan-accident-images-{revision}", verified_metadata_only=True)
            completed.append({"doc_id": f"pakistan-accident-images-{revision}", "file": "data/processed/accidents/image_metadata.json", "blocks": len(image_content)})
        except Exception as exc:
            failures.append({"doc_id": "pakistan-accident-images-v1", "error": f"{type(exc).__name__}: {exc}"})
    status = "complete" if not failures else "partial"
    manifest = {"status": status, "scope": "curated traffic rules, fines, signs, normalized Pakistan accident reports/statistics, and verified record-associated accident images", "framework": "RAG-Anything / LightRAG", "version": "1.4.1", "index_kind": "semantic vector chunks plus multimodal image content", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "query_mode": settings.rag_query_mode, "completed_at": datetime.now(timezone.utc).isoformat(), "discovered_raw_pdfs": [str(path.relative_to(settings.project_root)) for path in documents], "documents": completed, "failures": failures, "totals": {"documents": len(completed), "blocks": len(chunks) + len(image_content), "rules": len(rules), "fines": len(fines), "traffic_sign_images": len(signs), "accident_records": len(accident_records), "accident_images": len(accident_images), "verified_accident_image_pairs": len(accident_images), "accident_statistics": len(accident_statistics)}}
    output = settings.data_dir / "processed" / "metadata" / "ingestion_manifest.json"
    manifest["totals"]["verified_accident_image_pairs"] = sum(bool(image.get("verified_record_match") and image.get("record_id")) for image in accident_images)
    manifest["totals"]["generic_accident_images"] = sum(not image.get("record_id") for image in accident_images)
    manifest["totals"]["blocks"] = sum(document["blocks"] for document in completed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {output} ({manifest['status']})")
    if worker and worker.poll() is None:
        worker.terminate()
    if failures:
        raise RuntimeError(f"Ingestion completed with {len(failures)} failed document(s)")


def main() -> None:
    asyncio.run(ingest())


if __name__ == "__main__":
    main()
