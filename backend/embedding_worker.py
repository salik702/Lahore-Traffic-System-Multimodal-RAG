"""Semantic embedding sidecar preserving the dimension of the existing LightRAG index."""
import json
import logging
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "gemini-embedding-001"
from pathlib import Path
import re
_vector_path = Path(__file__).resolve().parent / "rag_storage" / "vdb_chunks.json"
OUTPUT_DIMENSIONALITY = 384
if _vector_path.is_file():
    with _vector_path.open(encoding="utf-8") as _handle:
        _match = re.search(r'"embedding_dim"\s*:\s*(\d+)', _handle.read(256))
    if _match:
        OUTPUT_DIMENSIONALITY = int(_match[1])
LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_local_model = None
ENCODE_LOCK = threading.Lock()


def load_project_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_path = os.path.abspath(env_path)
    if not os.path.isfile(env_path):
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if os.environ.get("GEMINI_API_KEY") or not os.path.isfile(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_project_env()


def embed(texts):
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_MODEL, device="cpu", local_files_only=True)
    return _local_model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ok", "model": LOCAL_MODEL if OUTPUT_DIMENSIONALITY == 384 else MODEL, "dimensions": OUTPUT_DIMENSIONALITY}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 8 * 1024 * 1024:
                raise ValueError("Embedding batch exceeds limit")
            payload = json.loads(self.rfile.read(length))
            texts = [str(value) for value in payload.get("texts", [])]
            with ENCODE_LOCK:
                vectors = embed(texts)
            body = json.dumps({"embeddings": vectors}).encode()
            self.send_response(200)
        except Exception:
            logging.getLogger(__name__).exception("Embedding request failed")
            body = json.dumps({"error": "Embedding provider unavailable; check worker logs and cached model"}).encode()
            self.send_response(503)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *_):
        return


if __name__ == "__main__":
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
