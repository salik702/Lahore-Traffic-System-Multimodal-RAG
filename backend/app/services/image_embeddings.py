"""Additive, content-addressed CLIP index. Never mix CLIP and MiniLM vectors."""
import hashlib
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.core.config import Settings

logger = logging.getLogger(__name__)
MODEL_ID = "openai/CLIP-RN50"
MODEL_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
MODEL_URL = f"https://openaipublic.azureedge.net/clip/models/{MODEL_SHA256}/RN50.pt"
DIMENSION = 1024
PREPROCESSING = "clip-rn50-white-square-v1"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION or not np.isfinite(vectors).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid or incompatible image/text embeddings")
    return vectors / norms


def read_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".svg":
        # Only corpus SVGs with no external resources are accepted.
        import pymupdf as fitz
        import xml.etree.ElementTree as ET
        svg = path.read_bytes()
        for element in ET.fromstring(svg).iter():
            for key, value in element.attrib.items():
                if key.rsplit("}", 1)[-1] == "href" and not value.startswith(("#", "data:")):
                    raise ValueError("External SVG resource")
        with fitz.open(stream=svg, filetype="svg") as document:
            with fitz.open("pdf", document.convert_to_pdf()) as pdf:
                page = pdf[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(512 / max(page.rect.width, page.rect.height), 512 / max(page.rect.width, page.rect.height)), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")
    # Preserve the entire symbol/photo instead of cropping off arrows or borders.
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


class ClipEncoder:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError("Image encoder not provisioned; run scripts/index_images.py --download-model")
        if file_hash(path) != MODEL_SHA256:
            raise ValueError("Image encoder checksum mismatch")
        import clip
        import torch
        self.torch = torch
        self.clip = clip
        # Reuse installed PyTorch; bounded CPU parallelism avoids oversubscription.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model, self.preprocess = clip.load(str(path), device="cpu", jit=False)
        self.model.eval()

    def images(self, paths: list[Path]) -> np.ndarray:
        batch = self.torch.stack([self.preprocess(read_image(path)) for path in paths])
        with self.torch.inference_mode():
            return normalized(self.model.encode_image(batch).float().cpu().numpy())

    def text(self, query: str) -> np.ndarray:
        with self.torch.inference_mode():
            return normalized(self.model.encode_text(self.clip.tokenize([query], truncate=True)).float().cpu().numpy())[0]


class ImageEmbeddingIndex:
    def __init__(self, settings: Settings, encoder_factory=ClipEncoder):
        self.settings = settings
        self.encoder_factory = encoder_factory
        self._encoder = None
        self._vectors = None
        self._manifest = {}
        self._hash_rows = {}
        self._query_cache = OrderedDict()
        self._lock = threading.RLock()
        self._cache_stamp = None
        self.error = None

    def _load(self) -> bool:
        if not self.settings.image_index_path.is_file():
            return False
        stat = self.settings.image_index_path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._vectors is not None and stamp == self._cache_stamp:
            return True
        try:
            with np.load(self.settings.image_index_path, allow_pickle=False) as saved:
                manifest = json.loads(str(saved["manifest"].item()))
                if (manifest.get("model"), manifest.get("model_sha256"), manifest.get("preprocessing")) != (MODEL_ID, MODEL_SHA256, PREPROCESSING):
                    raise ValueError("Image index model/preprocessing mismatch")
                vectors = normalized(saved["vectors"])
                if len(vectors) != len(manifest["assets"]) or len({a["sha256"] for a in manifest["assets"]}) != len(vectors):
                    raise ValueError("Image index association mismatch")
                self._manifest, self._vectors = manifest, vectors
                self._hash_rows = {asset["sha256"]: i for i, asset in enumerate(manifest["assets"])}
                self._cache_stamp = stamp
                self.error = None
                return True
        except Exception as exc:
            self.error = type(exc).__name__
            logger.warning("image_index_load_failed %s", self.error)
            return False

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = self.encoder_factory(self.settings.image_embedding_model_path)
        return self._encoder

    def build(self, batch_size: int = 4) -> dict:
        """Explicit repeatable migration: reuse unchanged pixels; atomically add cache."""
        with self._lock:
            self._load()
            root = self.settings.data_dir.resolve()
            assets, skipped = {}, []
            for folder in ("raw/traffic_signs", "raw/images", "raw/accidents/images"):
                for path in sorted((root / folder).rglob("*")):
                    if path.is_file() and path.suffix.lower() in EXTENSIONS and path.resolve().is_relative_to(root):
                        digest = file_hash(path)
                        assets.setdefault(digest, {"sha256": digest, "files": []})["files"].append(path.relative_to(root).as_posix())
            completed, vectors, pending = [], [], []
            reused = 0
            for digest, asset in assets.items():
                if digest in self._hash_rows:
                    completed.append(asset)
                    vectors.append(self._vectors[self._hash_rows[digest]])
                    reused += 1
                else:
                    pending.append(asset)
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                encoder = self._get_encoder()
                try:
                    embeddings = encoder.images([root / item["files"][0] for item in batch])
                    if len(embeddings) != len(batch):
                        raise ValueError("Image embedding batch association mismatch")
                    completed.extend(batch)
                    vectors.extend(normalized(embeddings))
                except Exception:
                    # One broken image must not discard the other images in its batch.
                    for asset in batch:
                        try:
                            vector = normalized(encoder.images([root / asset["files"][0]]))[0]
                            completed.append(asset)
                            vectors.append(vector)
                        except Exception as exc:
                            skipped.append({"files": asset["files"], "error": type(exc).__name__})
                logger.info("image_index_progress %s", json.dumps({"completed": len(completed), "total": len(assets)}))
            if not vectors:
                raise ValueError("No valid dataset images were embedded; original cache retained")
            manifest = {"model": MODEL_ID, "model_sha256": MODEL_SHA256, "dimension": DIMENSION,
                        "preprocessing": PREPROCESSING, "assets": completed, "skipped": skipped,
                        "file_count": sum(len(a["files"]) for a in completed), "reused_vectors": reused}
            target = self.settings.image_index_path
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, manifest=np.array(json.dumps(manifest)), vectors=normalized(np.stack(vectors)))
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            self._vectors = None
            self._query_cache.clear()
            self._load()
            return manifest

    def scores(self, query: str, candidates: dict[str, list[Path]]) -> dict[str, float]:
        """Score only caller-authorized record associations, never invented sources."""
        if not self.settings.image_embeddings_enabled or not candidates:
            return {}
        with self._lock:
            if not self._load():
                return {}
            try:
                root = self.settings.data_dir.resolve()
                rows = {}
                for record_id, paths in candidates.items():
                    indices = [self._hash_rows[digest] for path in paths
                               if path.is_file() and path.resolve().is_relative_to(root)
                               and (digest := file_hash(path)) in self._hash_rows]
                    if indices:
                        rows[record_id] = indices
                if not rows:
                    return {}
                if query not in self._query_cache:
                    vector = self._get_encoder().text(query)
                    self._query_cache[query] = normalized(np.asarray(vector)[None, :])[0]
                    if len(self._query_cache) > 64:
                        self._query_cache.popitem(last=False)
                all_scores = self._vectors @ self._query_cache[query]
                return {record_id: float(np.max(all_scores[indices])) for record_id, indices in rows.items()}
            except Exception as exc:
                self.error = type(exc).__name__
                logger.warning("image_query_failed_using_catalogue %s", self.error)
                return {}

    def status(self) -> dict:
        # Health calls run on the API event loop. Never block them on a cold
        # encoder load/inference in another thread; report the cached snapshot.
        acquired = self._lock.acquire(blocking=False)
        try:
            loaded = self._load() if acquired else self._vectors is not None
            return {"enabled": self.settings.image_embeddings_enabled, "indexed": loaded,
                    "model_available": self.settings.image_embedding_model_path.is_file(), "model": MODEL_ID,
                    "unique_images": len(self._hash_rows), "indexed_files": self._manifest.get("file_count", 0),
                    "skipped_images": len(self._manifest.get("skipped", [])), "error": self.error}
        finally:
            if acquired:
                self._lock.release()
