"""Process the project's discovered traffic PDFs through RAG-Anything.

Run from the repository root with ``backend/.venv/Scripts/python.exe
scripts/ingest_rag_documents.py``. Processing is sequential and writes the
normal RAG-Anything/LightRAG storage plus a separate, auditable manifest.

The whole ``backend/rag_storage`` directory is checksummed and copied before the
run and restored if the run cannot finish, so a failed attempt never leaves the
existing semantic index half-written. ``RAG_LLM_MAX_ASYNC`` raises the extraction
concurrency for this bulk job only; the API keeps its conservative default.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import pymupdf

from app.core.config import get_settings
from app.rag.engine import RagQuotaExceeded, RAGAnythingEngine

BACKUP_PREFIX = "rag_document_ingestion_backup"


def graph_counts(path: Path) -> tuple[int, int]:
    import xml.etree.ElementTree as ET

    if not path.is_file():
        return 0, 0
    root = ET.parse(path).getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    return len(root.findall(".//g:node", ns)), len(root.findall(".//g:edge", ns))


def storage_digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*")) if path.is_file()
    }


def make_backup(directory: Path) -> tuple[Path, dict[str, str]]:
    """Copy the live storage aside so a failed run can be rolled back exactly."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = directory.parent / f"{BACKUP_PREFIX}_{stamp}"
    shutil.copytree(directory, backup)
    return backup, storage_digests(directory)


def restore_backup(backup: Path, directory: Path, expected: dict[str, str]) -> None:
    for path in directory.glob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted(backup.glob("*")):
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    restored = storage_digests(directory)
    if restored != expected:
        raise RuntimeError("Restored storage does not match the pre-run checksums")


def start_embedding_worker(settings):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8765)) == 0:
            return None
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    worker = subprocess.Popen(
        [settings.semantic_python_executable, str(ROOT / "backend" / "embedding_worker.py")],
        cwd=ROOT / "backend", env=env, creationflags=flags,
    )
    timeout = max(1, int(settings.semantic_startup_timeout))
    for _ in range(timeout):
        if worker.poll() is not None:
            raise RuntimeError(f"Embedding worker exited with status {worker.returncode}")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8765)) == 0:
                return worker
        time.sleep(1)
    worker.terminate()
    raise RuntimeError("Local semantic embedding worker did not become ready")


def chunk_vector_count(path: Path) -> int:
    """Count chunk vectors in the persisted LightRAG vector store."""
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("data", []) if isinstance(data, dict) else data
    return len(records)


async def ingest_documents() -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    # Groq's free tier cannot cover this corpus, so only Groq needs the request
    # bounds and sequential extraction that its per-minute budget requires.
    if (settings.rag_llm_provider or "").strip().casefold() != "groq":
        settings.rag_llm_max_completion_tokens = max(
            settings.rag_llm_max_completion_tokens, 8000)
        settings.rag_chunk_token_size = max(settings.rag_chunk_token_size, 1800)
        settings.rag_llm_max_async = max(1, int(os.environ.get("RAG_LLM_MAX_ASYNC", "4")))
    else:
        settings.rag_llm_max_async = max(1, int(os.environ.get("RAG_LLM_MAX_ASYNC", "1")))
    docs = sorted((settings.data_dir / "raw" / "documents").glob("*.pdf"))
    graph_path = settings.rag_storage / "graph_chunk_entity_relation.graphml"
    storage_dir = settings.rag_storage
    before_nodes, before_edges = graph_counts(graph_path)
    backup, baseline = make_backup(storage_dir)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "RAG-Anything / LightRAG",
        "raganything_version": "1.4.1",
        "requested_parser": settings.rag_parser,
        "parse_method": "auto",
        "source_root": "data/raw/documents",
        "graph_storage": "backend/rag_storage/graph_chunk_entity_relation.graphml",
        "storage_backup": str(backup.relative_to(ROOT).as_posix()),
        "storage_baseline_sha256": baseline,
        "chunk_vectors_before": chunk_vector_count(storage_dir / "vdb_chunks.json"),
        "llm_provider": settings.rag_llm_provider,
        "llm_model": settings.gemini_model if (settings.rag_llm_provider or "").casefold() == "gemini" else settings.rag_extraction_model,
        "vision_model": settings.gemini_vision_model if (settings.rag_llm_provider or "").casefold() == "gemini" else settings.groq_vision_model,
        "llm_max_async": settings.rag_llm_max_async,
        "llm_max_completion_tokens": settings.rag_llm_max_completion_tokens,
        "chunk_token_size": settings.rag_chunk_token_size,
        "graph_before": {"entities": before_nodes, "relationships": before_edges},
        "documents": [],
    }
    worker = None
    engine = RAGAnythingEngine(settings)
    rolled_back = False
    try:
        worker = start_embedding_worker(settings)
        await engine.initialize()
        manifest["parser"] = engine.parser
        manifest["parser_note"] = engine.parser_note
        if not engine.ready:
            raise RuntimeError(engine.error or "RAG-Anything could not initialize")
        for path in docs:
            relative = path.relative_to(ROOT).as_posix()
            with pymupdf.open(path) as pdf:
                pages = len(pdf)
                text_pages = sum(bool(page.get_text("text").strip()) for page in pdf)
            item = {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pages_discovered": pages,
                "pages_with_text_layer": text_pages,
                "status": "processing",
            }
            started = time.time()
            try:
                item["lightrag_doc_id"] = await engine.prepare_document(path)
                item["processing"] = await engine.ingest_document(path)
                item["status"] = "processed"
            except RagQuotaExceeded as exc:
                # The provider allowance cannot be waited out mid-corpus. Stop here
                # so the run ends with an honest manifest instead of an hour of
                # retries against a dead daily limit.
                manifest["documents"].append({
                    **item, "status": "not_attempted", "error": f"RagQuotaExceeded: {exc}",
                })
                raise
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"
            item["seconds"] = round(time.time() - started, 1)
            # RAG-Anything logs extraction failures and returns normally, so the
            # persisted doc_status is the only proof that the document landed.
            if item["status"] == "processed":
                persisted = await engine.document_status(item["lightrag_doc_id"])
                item["lightrag_status"] = persisted
                if not persisted.get("persisted"):
                    # doc_status says "processed" even when extraction failed and
                    # nothing was written; only a real chunk count proves an insert.
                    item["status"] = "failed"
                    item["error"] = (
                        f"LightRAG reported status {persisted.get('status')!r} but persisted no "
                        "chunks, so no entities or relationships were written."
                    )
            manifest["documents"].append(item)
            nodes, edges = graph_counts(graph_path)
            manifest.setdefault("graph_after_each_document", []).append({
                "path": relative, "entities": nodes, "relationships": edges,
            })
            print(f"{relative}: {item['status']} in {item['seconds']}s "
                  f"(graph {nodes} entities / {edges} relationships)", flush=True)
        if engine.rag is not None:
            await engine.rag.finalize_storages()
        after_nodes, after_edges = graph_counts(graph_path)
        manifest["graph_after"] = {"entities": after_nodes, "relationships": after_edges}
        manifest["chunk_vectors_after"] = chunk_vector_count(storage_dir / "vdb_chunks.json")
        manifest["summary"] = {
            "discovered": len(docs),
            "processed": sum(d["status"] == "processed" for d in manifest["documents"]),
            "failed": sum(d["status"] == "failed" for d in manifest["documents"]),
        }
        manifest["status"] = "complete" if manifest["summary"]["failed"] == 0 else "partial"
        manifest["lightrag_documents_before"] = sorted(await engine.known_document_ids())
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        # A half-finished run can leave chunks without vectors or entities without
        # edges. Put the verified pre-run storage back before reporting the failure.
        try:
            restore_backup(backup, storage_dir, baseline)
            rolled_back = True
        except Exception as restore_exc:
            manifest["rollback_error"] = f"{type(restore_exc).__name__}: {restore_exc}"
    finally:
        if worker and worker.poll() is None:
            worker.terminate()
        manifest["storage_rolled_back"] = rolled_back
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        output = settings.data_dir / "processed" / "metadata" / "rag_document_ingestion.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest.get("status") != "complete":
        raise RuntimeError("One or more RAG-Anything document operations failed; see the manifest")
    return manifest


if __name__ == "__main__":
    asyncio.run(ingest_documents())