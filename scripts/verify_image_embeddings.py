"""Live pixel retrieval, source boundaries and additive-cache acceptance checks."""
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.config import Settings
from app.services.image_embeddings import DIMENSION, EXTENSIONS, file_hash


async def main():
    settings = Settings()
    with np.load(settings.image_index_path, allow_pickle=False) as saved:
        manifest = json.loads(str(saved["manifest"].item()))
        vectors = saved["vectors"]
    assert vectors.shape == (len(manifest["assets"]), DIMENSION)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    files = {filename for asset in manifest["assets"] for filename in asset["files"]}
    expected_files = {path.relative_to(settings.data_dir).as_posix()
                      for folder in ("raw/traffic_signs", "raw/images", "raw/accidents/images")
                      for path in (settings.data_dir / folder).rglob("*")
                      if path.is_file() and path.suffix.lower() in EXTENSIONS}
    assert files == expected_files and not manifest["skipped"]
    for asset in manifest["assets"]:
        assert all(file_hash(settings.data_dir / filename) == asset["sha256"] for filename in asset["files"])
    rows = []
    cases = [("show me roundabout sign", {"compulsory-roundabout"}),
             ("show me road clousre sign", {"road-closed"}),
             ("show me a sign with circular arrows", {"compulsory-roundabout"}),
             ("What does it mean?", {"compulsory-roundabout"}),
             ("show me a red roundabout sign", {"compulsory-roundabout"}),
             ("show me a blue roundabout sign", set()),
             ("show me accident photos from Lahore", {"pk-lahore-m2-fog-crash-2008-12-29"}),
             ("show me a photo of a bus in a river", {"pk-gilgit-bus-stream-2004-08-08"}),
             ("show me accident photos from Manchester", set()),
             ("show me the teleportation sign", set()),
             ("show me truck accident photos", {"pk-rawalpindi-westridge-truck-2011-04-14", "pk-islamabad-truck-accident-2011-06-22"}),
             ("What happens after a road accident?", None),
             ("What is the helmet fine in Lahore?", None)]
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/api/v1", timeout=100) as client:
        health = (await client.get("/health")).json()
        assert health["rag_anything_ready"] and health["image_embeddings"]["indexed"]
        assert health["image_embeddings"]["indexed_files"] == len(files)
        session = None
        for index, (question, expected) in enumerate(cases):
            if index in {3, 11, 12}:
                await asyncio.sleep(15)
            response = await client.post("/chat", json={"message": question, "session_id": session if index == 3 else None})
            response.raise_for_status()
            data = response.json()
            ids = {e["id"] for e in data["evidence"]}
            if expected is not None:
                assert ids == expected, (question, ids, data["answer"])
                assert bool(data["images"]) == bool(expected), data
            else:
                assert not data["images"] and data["citations"], data
                assert "1122" in data["answer"] if index == 11 else "2,000" in data["answer"], data
            if index == 2:
                session = data["session_id"]
            if index == 6:
                assert all(image["region"] == "Lahore" for image in data["images"])
            if index == 1:
                assert "project illustration" in data["answer"].lower() and not data["citations"][0]["official"]
            assert not re.search(r"[A-Za-z]:[\\/]|Image Path|Visual Analysis|PROVENANCE|image_similarity|image_rank_fusion", data["answer"])
            assert all(set(e["metadata"]) <= {"page", "section", "category", "citation_number"} for e in data["evidence"])
            assert set(map(int, re.findall(r"\[(\d+)\]", data["answer"]))) <= set(range(1, len(data["citations"]) + 1))
            for image in data["images"]:
                media = await client.get("http://127.0.0.1:8000" + image["url"])
                assert media.status_code == 200 and media.headers["content-type"].startswith("image/"), image
                assert image["source_url"].startswith(("http://", "https://"))
            rows.append({"question": question, "evidence_ids": sorted(ids), "answer": data["answer"], "images": data["images"], "passed": True})
            print(f"PASS {question}", flush=True)
    report = {"health": health, "indexed_files_checked": len(files), "unique_pixel_vectors": len(vectors), "cases": rows}
    (settings.image_index_path.parent / "image-integration-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} passing live cases; {len(files)} file associations checked.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
