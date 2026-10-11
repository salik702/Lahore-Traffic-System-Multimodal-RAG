"""Provision genuine CLIP weights and build/reuse the additive image-only index."""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.config import Settings
from app.services.image_embeddings import ImageEmbeddingIndex, MODEL_SHA256, MODEL_URL, file_hash


def download_model(target: Path):
    if target.is_file() and file_hash(target) == MODEL_SHA256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".download")
    try:
        # Bounded range requests avoid stalled whole-file transfers on Windows.
        # Every range and the final official checksum are validated before publish.
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(MODEL_URL, headers={"Range": "bytes=0-0"})
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get("content-range", "").startswith("bytes 0-0/"):
                raise ValueError("Model host does not support validated range downloads")
            size = int(response.headers["content-range"].split("/")[-1])
            chunk_size = 4 * 1024 * 1024

            def fetch(start):
                end = min(start + chunk_size, size) - 1
                result = client.get(MODEL_URL, headers={"Range": f"bytes={start}-{end}"})
                result.raise_for_status()
                if result.status_code != 206 or result.headers.get("content-range") != f"bytes {start}-{end}/{size}" or len(result.content) != end - start + 1:
                    raise ValueError("Incomplete or mismatched model download range")
                return start, result.content

            with partial.open("wb") as handle, ThreadPoolExecutor(max_workers=4) as pool:
                handle.truncate(size)
                pending = [pool.submit(fetch, start) for start in range(0, size, chunk_size)]
                received = 0
                for future in as_completed(pending):
                    start, content = future.result()
                    handle.seek(start)
                    handle.write(content)
                    received += len(content)
                    logging.info("image_model_download %s%%", round(received / size * 100))
        if file_hash(partial) != MODEL_SHA256:
            raise ValueError("Downloaded model checksum does not match the official release")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true", help="Provision the official 256 MB CPU CLIP model once")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    if args.download_model:
        download_model(settings.image_embedding_model_path)
    index = ImageEmbeddingIndex(settings)
    manifest = index.build()
    report = {**index.status(), "reused_vectors": manifest["reused_vectors"], "skipped": manifest["skipped"]}
    (settings.image_index_path.parent / "image-index-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
