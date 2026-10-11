"""Small source-verified Lahore news fact catalogue; publisher photos not reused."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "app-lahore-accident-facts"
CANDIDATES = [
    {"id": "app-lahore-dha-2026-08-08", "slug": "punjab/motorcyclist-killed-in-road-accident-in-dha/", "date": "2026-08-08", "title": "Fatal motorcycle collision in DHA Phase VI", "description": "APP reports two motorcycles collided in Lahore's DHA Phase VI, with one rider killed, citing Rescue 1122.", "location": "DHA Phase VI", "vehicle_types": ["motorcycle"], "fatalities": 1, "injuries": None},
    {"id": "app-lahore-dha-2026-09-12", "slug": "punjab/motorcyclist-killed-another-injured-on-road/", "date": "2026-09-12", "title": "Motorcycle crash at Pakistan Chowk, DHA Phase V", "description": "APP reports a motorcycle struck a footpath at Pakistan Chowk in Lahore's DHA Phase V. One rider died and another was injured, according to Rescue 1122.", "location": "Pakistan Chowk, DHA Phase V", "vehicle_types": ["motorcycle"], "fatalities": 1, "injuries": 1},
    {"id": "app-lahore-dha-raya-2025-11-01", "slug": "domestic/3-killed-as-garbage-dumper-hits-motorcycle-in-dha/", "date": "2025-11-01", "title": "Dumper and motorcycle collision at DHA Raya", "description": "APP reports three deaths in a dumper and motorcycle collision at DHA Raya, Lahore, citing Rescue 1122.", "location": "DHA Raya", "vehicle_types": ["motorcycle", "truck"], "fatalities": 3, "injuries": None},
]


def main():
    raw = ROOT / "data/raw/accidents/reports/app"
    raw.mkdir(parents=True, exist_ok=True)
    acquired = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for item in CANDIDATES:
            url = "https://www.app.com.pk/" + item["slug"]
            path = raw / (item["id"] + ".html")
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
            html = path.read_text(encoding="utf-8", errors="replace")
            # Validate the researched publication date against source metadata.
            dates = re.findall(r'(?:datePublished|article:published_time)[^\n]{0,100}?(\d{4}-\d{2}-\d{2})', html)
            if item["date"] not in dates:
                raise RuntimeError(f"Source publication date mismatch for {item['id']}: {dates}")
            acquired.append({"id": item["id"], "type": "accident_report", "title": item["title"], "description": item["description"],
                             "description_type": "brief_source_supported_factual_summary", "date": item["date"], "date_basis": "Reported incident date; verified against publication metadata", "time": None,
                             "location": item["location"], "city": "Lahore", "district": "Lahore", "province": "Punjab", "country": "Pakistan", "latitude": None, "longitude": None,
                             "accident_type": "road traffic collision", "vehicle_types": item["vehicle_types"], "fatalities": item["fatalities"], "injuries": item["injuries"], "severity": "Fatal",
                             "source_name": "Associated Press of Pakistan (APP)", "source_id": SOURCE_ID, "source_url": url, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "report_path": path.relative_to(ROOT).as_posix(), "image_path": None, "image_url_original": None, "image_license": None, "image_verified_for_record": False,
                             "image_note": "Publisher image not downloaded: reusable rights and event-photo relationship are not established.", "retrieved_at": datetime.now(timezone.utc).isoformat()})
            print(f"Acquired factual Lahore report: {item['id']}", flush=True)
    path = ROOT / "data/processed/accidents/records.json"
    existing = {item["id"]: item for item in json.loads(path.read_text(encoding="utf-8"))}
    existing.update({item["id"]: item for item in acquired})
    reports = sorted(existing.values(), key=lambda item: (item.get("date") or "", item["id"]), reverse=True)
    path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (ROOT / "data/processed/accidents/rag_documents/accidents.jsonl").open("w", encoding="utf-8") as stream:
        for report in reports:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
    sources_path = ROOT / "data/sources/sources.json"
    sources = [source for source in json.loads(sources_path.read_text(encoding="utf-8")) if source["id"] != SOURCE_ID]
    sources.append({"id": SOURCE_ID, "title": "Lahore accident facts reported by APP", "organization": "Associated Press of Pakistan",
                    "original_url": "https://www.app.com.pk/punjab/", "download_url": "", "retrieval_date": datetime.now(timezone.utc).date().isoformat(), "data_type": "Individual news report factual metadata",
                    "type": "accident_reports", "region": "Lahore, Punjab", "geographic_scope": "Lahore, Punjab, Pakistan", "official": False,
                    "license": "Publisher copyright; factual metadata and brief original paraphrases only. No publisher images reused.", "local_file_path": "data/raw/accidents/reports/app",
                    "description": "Three sourced Lahore event records with verified publication dates; photographs excluded because reuse and event relationships are unverified.", "records": len(acquired), "images": 0, "image_record_relationship": "none"})
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
