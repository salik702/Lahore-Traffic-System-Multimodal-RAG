"""Acquire a bounded RESEARCH-ONLY accident-class sample for provenance review.

Uses ordinary HTTP byte ranges to read the public ZIP, avoiding a 1.8 GB
download of unrelated fighting/snatching classes. No access controls bypassed.
The reviewed source was rejected for public ingestion: the deposit's CC BY 4.0
does not establish per-image rights/provenance for news/international samples.
Run with backend/.venv/Scripts/python.exe scripts/acquire_accident_imagery.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
import yaml
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
API = "https://zenodo.org/api/records/13832363"
SOURCE = "https://zenodo.org/records/13832363"
SOURCE_ID = "zenodo-pakistan-road-anomaly-13832363"


class PublicRangeReader:
    def __init__(self, client, url, size):
        self.client, self.url, self.size, self.position = client, url, size, 0
        self.blocks = {}

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size)
        if count <= 0:
            return b""
        pieces = []
        while count:
            block_start = self.position // 1_048_576 * 1_048_576
            if block_start not in self.blocks:
                block_end = min(block_start + 1_048_576, self.size) - 1
                # Stream so a server ignoring Range never downloads the entire archive.
                for attempt in range(3):
                    try:
                        with self.client.stream("GET", self.url, headers={"Range": f"bytes={block_start}-{block_end}"}) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {block_start}-"):
                                raise RuntimeError("Public archive does not support byte ranges; refusing an unbounded download")
                            block = response.read()
                        break
                    except httpx.TransportError:
                        if attempt == 2:
                            raise
                        time.sleep(1)
                if len(block) != block_end - block_start + 1:
                    raise RuntimeError("Incomplete archive byte range")
                self.blocks[block_start] = block
            block = self.blocks[block_start]
            offset = self.position - block_start
            take = min(count, len(block) - offset)
            pieces.append(block[offset:offset + take])
            self.position += take
            count -= take
        return b"".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    raw = ROOT / "data/raw/accidents/reports/zenodo-road-anomaly"
    raw.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "LahoreTrafficIntelligence/1.0 academic dataset acquisition"}) as client:
        metadata_snapshot = raw / "source_metadata.json"
        if metadata_snapshot.exists():
            release = json.loads(metadata_snapshot.read_text(encoding="utf-8"))
        else:
            response = client.get(API)
            response.raise_for_status()
            release = response.json()
        if release["metadata"].get("license", {}).get("id") != "cc-by-4.0":
            raise RuntimeError("Expected CC BY 4.0 license could not be verified")
        (raw / "source_metadata.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
        file = release["files"][0]
        reader = PublicRangeReader(client, file["links"]["self"], file["size"])
        with zipfile.ZipFile(reader) as archive:
            entries = archive.infolist()
            inventory = [{"name": entry.filename, "bytes": entry.file_size, "crc32": entry.CRC} for entry in entries]
            (raw / "archive_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
            schemas = [entry for entry in entries if Path(entry.filename).suffix.lower() in {".yaml", ".yml"}]
            names = {}
            for schema_entry in schemas:
                schema_text = archive.read(schema_entry).decode("utf-8")
                schema = yaml.safe_load(schema_text)
                if isinstance(schema, dict) and "names" in schema:
                    names = schema["names"]
                    names = dict(enumerate(names)) if isinstance(names, list) else names
                    (raw / "class_schema.yaml").write_text(schema_text, encoding="utf-8")
                    break
            accident_ids = {str(key) for key, name in names.items() if "accident" in str(name).lower()}
            candidates = [entry for entry in entries if Path(entry.filename).suffix.lower() in {".jpg", ".jpeg", ".png"} and Path(entry.filename).name.lower().startswith("acci-")]
            print(f"Source classes: {names}", flush=True)
            print(f"Archive entries: {len(entries)}; accident-class candidates: {len(candidates)}", flush=True)
            print("\n".join(entry.filename for entry in candidates[:40]), flush=True)
            if args.inspect:
                return
            if not candidates:
                raise RuntimeError("No explicitly named accident class found; inspect inventory before acquisition")
            images = []
            hashes = set()
            source_frames = set()
            for entry in sorted(candidates, key=lambda item: item.filename):
                if len(images) >= args.limit:
                    break
                if entry.file_size > 15_000_000:
                    continue
                frame = Path(entry.filename).name.split(".rf.")[0]
                if frame in source_frames:
                    continue
                label_member = entry.filename.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                label_text = archive.read(label_member).decode("utf-8")
                if not label_text.strip() or (accident_ids and not any(line.split() and line.split()[0] in accident_ids for line in label_text.splitlines())):
                    continue
                content = archive.read(entry)  # zipfile verifies the source CRC
                digest = hashlib.sha256(content).hexdigest()
                if digest in hashes:
                    continue
                with Image.open(io.BytesIO(content)) as picture:
                    picture.verify()
                hashes.add(digest)
                source_frames.add(frame)
                filename = f"zenodo-accident-{digest[:16]}{Path(entry.filename).suffix.lower()}"
                destination = raw / "review_sample" / filename
                destination.parent.mkdir(exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(content)
                images.append({
                    "id": f"zenodo-acc-img-{digest[:16]}", "type": "accident_image",
                    "media_role": "generic_dataset_image", "title": "Pakistan road-anomaly dataset — accident-class image",
                    "description": "Accident-class image from the research dataset. No individual incident report, date, city, vehicle type or casualty count is supplied for this image.",
                    "description_type": "dataset_class_summary", "city": None, "district": None, "province": None,
                    "country": "Pakistan", "geography_basis": "Collection-level publisher statement: primary data collected at diverse sites in Pakistan; individual image location unverified.",
                    "accident_type": None, "vehicle_types": [], "date": None, "record_id": None,
                    "source": "Zenodo — Sarfaraz Natha", "source_id": SOURCE_ID, "source_url": SOURCE,
                    "author": "Sarfaraz Natha", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "image_path": destination.relative_to(ROOT).as_posix(), "image_url_original": file["links"]["self"],
                    "archive_member": entry.filename, "sha256": digest, "verified_record_match": False,
                    "annotation_member": label_member, "source_annotation": label_text,
                    "class_assignment_basis": "Source accident-coded filename prefix acci; paired raw YOLO annotation retained. Numeric class-name mapping not supplied." if not names else "Source class schema and YOLO annotation",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"Acquired {len(images)}: {entry.filename}", flush=True)
    # This candidate failed visual provenance review. Never publish or index it.
    (raw / "review_sample_metadata.json").write_text(json.dumps(images, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contact = Image.new("RGB", (960, ((len(images) + 3) // 4) * 200), "white")
    drawing = ImageDraw.Draw(contact)
    for index, item in enumerate(images):
        with Image.open(ROOT / item["image_path"]) as original:
            thumbnail = original.convert("RGB")
            thumbnail.thumbnail((236, 175))
        x, y = index % 4 * 240, index // 4 * 200
        contact.paste(thumbnail, (x, y))
        drawing.text((x + 3, y + 179), f"{index + 1}: {Path(item['archive_member']).name.split('.rf.')[0]}", fill="black")
    contact.save(raw / "review_contact_sheet.jpg")


if __name__ == "__main__":
    main()
