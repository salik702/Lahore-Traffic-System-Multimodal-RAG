"""Licensed international imagery, explicitly separate from Pakistan reports."""
import json
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image
from acquire_accident_data import commons_download_url

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "commons-international-generic-accident-media"
FILES = [
    {"filename": "Car_vs_motorcycle_accident.jpg", "author": "Chmee2", "license": "CC BY-SA 3.0", "license_url": "https://creativecommons.org/licenses/by-sa/3.0/", "date": "2008-02-01", "city": "Písek", "country": "Czech Republic", "description": "Source describes a car versus motorcycle accident at an intersection in Písek, Czech Republic. International reference image; not a Pakistan accident and not linked to a Pakistan record."},
    {"filename": "2009_Car_and_Motorcycle_Accident_Lasalle_and_Groveland.jpg", "author": "adria.richards", "license": "CC BY-SA 2.0", "license_url": "https://creativecommons.org/licenses/by-sa/2.0/", "date": "2009-08-21", "city": None, "country": "United States", "description": "Source photograph of a car and motorcycle accident at Lasalle and Groveland; US location supported by source geotag. International reference image; not a Pakistan accident and not linked to a Pakistan record."},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    metadata_path = ROOT / "data/processed/accidents/image_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rejected = [image for image in metadata if image.get("source_id") == "zenodo-pakistan-road-anomaly-13832363"]
    quarantine = ROOT / "data/raw/accidents/reports/zenodo-road-anomaly/rejected_sample"
    if rejected:
        quarantine.mkdir(parents=True, exist_ok=True)
        (quarantine / "metadata.json").write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")
        for image in rejected:
            path = ROOT / image["image_path"]
            if path.is_file() and path.parent == ROOT / "data/raw/accidents/images" and path.name.startswith("zenodo-accident-"):
                path.rename(quarantine / path.name)
    metadata = [image for image in metadata if image.get("source_id") != "zenodo-pakistan-road-anomaly-13832363"]
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    registry_path = ROOT / "data/sources/sources.json"
    sources = json.loads(registry_path.read_text(encoding="utf-8"))
    sources = [source for source in sources if source["id"] != "zenodo-pakistan-road-anomaly-13832363"]
    registry_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.cleanup_only:
        return
    # Stage replacements in memory; failed acquisition retains accepted metadata.
    metadata = [image for image in metadata if image.get("source_id") != SOURCE_ID]
    retrieved = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "LahoreTrafficIntelligence/1.0 academic licensed-data acquisition"}) as client:
        for index, item in enumerate(FILES, 1):
            original = commons_download_url(item["filename"])
            local = ROOT / "data/raw/accidents/images" / f"international-motorcycle-reference-{index:02d}.jpg"
            if not local.exists():
                for attempt in range(3):
                    response = client.get(original)
                    if response.status_code != 429 or attempt == 2:
                        break
                    delay = response.headers.get("Retry-After", "30")
                    delay = int(delay) if delay.isdigit() else 30
                    print(f"Source rate limit; respecting a {delay}s retry delay", flush=True)
                    time.sleep(delay)
                response.raise_for_status()
                local.write_bytes(response.content)
            with Image.open(local) as photo:
                photo.verify()
            source_url = "https://commons.wikimedia.org/wiki/File:" + item["filename"]
            metadata.append({"id": f"intl-generic-acc-img-{index:03d}", "type": "accident_image", "media_role": "international_generic_reference",
                             "title": f"International motorcycle accident reference — {item['country']}", "description": item["description"], "description_type": "source_supported_summary",
                             "city": item["city"], "district": None, "province": None, "country": item["country"], "date": item["date"], "accident_type": "car and motorcycle collision",
                             "vehicle_types": ["car", "motorcycle"], "record_id": None, "verified_record_match": False,
                             "source": "Wikimedia Commons", "source_id": SOURCE_ID, "source_url": source_url, "author": item["author"], "license": item["license"], "license_url": item["license_url"],
                             "image_path": local.relative_to(ROOT).as_posix(), "image_url_original": original, "retrieved_at": retrieved})
            print(f"Acquired licensed international reference: {local.name}", flush=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sources = json.loads(registry_path.read_text(encoding="utf-8"))
    sources = [source for source in sources if source["id"] not in {SOURCE_ID, "zenodo-pakistan-road-anomaly-13832363"}]
    sources.append({"id": SOURCE_ID, "title": "Licensed international motorcycle accident reference photographs", "organization": "Wikimedia Commons contributors",
                    "original_url": "https://commons.wikimedia.org/wiki/Category:Motorcycle_accidents", "download_url": "", "retrieval_date": retrieved[:10], "data_type": "Generic international accident imagery",
                    "type": "accident_images", "region": "International — not Pakistan", "geographic_scope": "Czech Republic and United States", "official": False,
                    "license": "Per file: CC BY-SA 3.0 or CC BY-SA 2.0", "local_file_path": "data/raw/accidents/images", "description": "International visual fallback only; never attached to Pakistan incident records.",
                    "records": 0, "images": len(FILES), "image_record_relationship": "none"})
    registry_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
