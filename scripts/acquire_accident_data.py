"""Acquire and normalize the auditable Pakistan accident corpus.

Run from the repository root with:
    uv run --project backend python scripts/acquire_accident_data.py

Only openly licensed or official public sources are downloaded.  Wikimedia image
metadata below is intentionally explicit so an image can never be associated with
an event merely because it looks relevant.
"""

from __future__ import annotations

import csv
import re
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import httpx


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accidents"
RAW_RECORDS = RAW / "records"
RAW_IMAGES = RAW / "images"
PROCESSED = ROOT / "data" / "processed" / "accidents"
SOURCES_PATH = ROOT / "data" / "sources" / "sources.json"
RETRIEVED_AT = datetime.now(timezone.utc).date().isoformat()

DATAVERSE_FILE = "https://dataverse.harvard.edu/api/access/datafile/11674565?format=original"
DATAVERSE_PAGE = "https://doi.org/10.7910/DVN/L51CR9"
RESCUE_URL = "https://www.rescue.gov.pk/performance.aspx"
RESCUE_VERIFIED_SNAPSHOT = [{
    "district": "Lahore", "emergency_calls": 2931042, "road_accidents": 1183000,
    "medical": 1376935, "fire": 70814, "structure_collapse": 2177,
    "crime_incidents": 67984, "drowning": 1792, "miscellaneous": 228340,
    "patients_rescued": 2863263, "reporting_period_end": "2026-09-30",
    "snapshot_note": "Values transcribed from the official public performance table after direct DNS access failed during acquisition.",
}]


def commons_download_url(filename: str) -> str:
    normalized = filename.replace(" ", "_")
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()  # Wikimedia storage layout
    return f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[:2]}/{quote(normalized)}"


COMMONS_MEDIA = [
    {
        "record_id": "pk-lahore-m2-fog-crash-2008-12-29",
        "title": "Fog crash near Ravi Toll Plaza on the M2",
        "description": "Photograph of a fog-related road crash on the M2 Lahore-Islamabad Motorway near Ravi Toll Plaza.",
        "date": "2008-12-29",
        "location": "M2 near Ravi Toll Plaza",
        "city": "Lahore",
        "district": "Lahore",
        "province": "Punjab",
        "accident_type": "fog-related road collision",
        "vehicle_types": [],
        "latitude": None,
        "longitude": None,
        "filename": "Fog_Crash.jpg",
        "local_filename": "lahore-m2-fog-crash-2008.jpg",
        "author": "Wasim3s",
        "license": "Public domain (dedicated by copyright holder)",
        "source_url": "https://commons.wikimedia.org/wiki/File:Fog_Crash.jpg",
    },
    {
        "record_id": "pk-gilgit-bus-stream-2004-08-08",
        "title": "Bus fell into a stream on the road to Gilgit",
        "description": "Photograph of a bus that fell into a stream while travelling toward Gilgit.",
        "date": "2004-08-08",
        "location": "Road to Gilgit",
        "city": None,
        "district": None,
        "province": "Gilgit-Baltistan",
        "accident_type": "bus road accident",
        "vehicle_types": ["bus"],
        "latitude": None,
        "longitude": None,
        "filename": "A_Bus_Accident_on_the_Karakoram_Highway_on_Way_to_Gilgit.jpg",
        "local_filename": "gilgit-bus-stream-2004.jpg",
        "author": "Anees Jillani",
        "license": "CC BY-SA 3.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:A_Bus_Accident_on_the_Karakoram_Highway_on_Way_to_Gilgit.jpg",
    },
    {
        "record_id": "pk-islamabad-truck-accident-2011-06-22",
        "title": "Truck accident in Islamabad",
        "description": "Photograph titled and geotagged by its author as a truck accident in Islamabad.",
        "date": "2011-06-22",
        "location": "Islamabad (source geotag)",
        "city": "Islamabad",
        "district": "Islamabad",
        "province": "Islamabad Capital Territory",
        "accident_type": "truck accident",
        "vehicle_types": ["truck"],
        "latitude": 33.665689,
        "longitude": 72.989447,
        "filename": "Truck_Accident_-_panoramio.jpg",
        "local_filename": "islamabad-truck-accident-2011.jpg",
        "author": "Jam Mudasir",
        "license": "CC BY 3.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Truck_Accident_-_panoramio.jpg",
    },
    {
        "record_id": "pk-rawalpindi-westridge-truck-2011-04-14",
        "title": "Truck accident in Westridge, Rawalpindi",
        "description": "First of three source-associated photographs of the same truck accident in Westridge, Rawalpindi.",
        "date": "2011-04-14",
        "location": "Westridge, Rawalpindi",
        "city": "Rawalpindi",
        "district": "Rawalpindi",
        "province": "Punjab",
        "accident_type": "truck accident",
        "vehicle_types": ["truck"],
        "latitude": 33.624715,
        "longitude": 73.018658,
        "filename": "Truck_accident_in_westridge_Rawalpindi_-_panoramio.jpg",
        "local_filename": "rawalpindi-westridge-truck-2011-01.jpg",
        "author": "Jam Mudasir",
        "license": "CC BY 3.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Truck_accident_in_westridge_Rawalpindi_-_panoramio.jpg",
    },
    {
        "record_id": "pk-rawalpindi-westridge-truck-2011-04-14",
        "title": "Truck accident in Westridge, Rawalpindi — view 2",
        "description": "Second source-associated photograph of the same truck accident in Westridge, Rawalpindi.",
        "date": "2011-04-14",
        "location": "Westridge, Rawalpindi",
        "city": "Rawalpindi",
        "district": "Rawalpindi",
        "province": "Punjab",
        "accident_type": "truck accident",
        "vehicle_types": ["truck"],
        "latitude": 33.624679,
        "longitude": 73.018664,
        "filename": "Truck_accident_in_westridge_Rawalpindi_-_panoramio_(1).jpg",
        "local_filename": "rawalpindi-westridge-truck-2011-02.jpg",
        "author": "Jam Mudasir",
        "license": "CC BY 3.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Truck_accident_in_westridge_Rawalpindi_-_panoramio_(1).jpg",
    },
    {
        "record_id": "pk-rawalpindi-westridge-truck-2011-04-14",
        "title": "Truck accident in Westridge, Rawalpindi — view 3",
        "description": "Third source-associated photograph of the same truck accident in Westridge, Rawalpindi.",
        "date": "2011-04-14",
        "location": "Westridge, Rawalpindi",
        "city": "Rawalpindi",
        "district": "Rawalpindi",
        "province": "Punjab",
        "accident_type": "truck accident",
        "vehicle_types": ["truck"],
        "latitude": 33.624679,
        "longitude": 73.018664,
        "filename": "Truck_accident_in_westridge_Rawalpindi_-_panoramio_(2).jpg",
        "local_filename": "rawalpindi-westridge-truck-2011-03.jpg",
        "author": "Jam Mudasir",
        "license": "CC BY 3.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Truck_accident_in_westridge_Rawalpindi_-_panoramio_(2).jpg",
    },
    {
        "record_id": "pk-malakand-car-river-2014-03-06",
        "title": "Car fell into a river in Malakand",
        "description": "Source photograph titled as a non-custom-paid car that fell into a river in the Malakand area.",
        "date": "2014-03-06",
        "location": "Malakand area",
        "city": None,
        "district": "Malakand",
        "province": "Khyber Pakhtunkhwa",
        "accident_type": "car left roadway",
        "vehicle_types": ["car"],
        "latitude": None,
        "longitude": None,
        "filename": "NCP_Non-Custom_Paid_Car_fall_into_the_river.jpg",
        "local_filename": "malakand-car-river-2014.jpg",
        "author": "Anaskhankhattak",
        "license": "CC BY-SA 4.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:NCP_Non-Custom_Paid_Car_fall_into_the_river.jpg",
    },
]


class RescueTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.cell = ""
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"td", "th"}:
            self.in_cell = True
            self.cell = ""

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell += data

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(" ".join(self.cell.split()))
            self.in_cell = False
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []


def download(client: httpx.Client, url: str, destination: Path) -> None:
    response = client.get(url)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"{destination.relative_to(ROOT)}: {len(response.content):,} bytes")


def normalize_rawalpindi(csv_path: Path) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["EcNumber"]].append(row)

    vehicle_columns = {
        "BicycleInvovled": "bicycle", "BikesInvolved": "motorcycle", "BusesInvolved": "bus",
        "CarsInvolved": "car", "CartInvovled": "cart", "RickshawsInvolved": "rickshaw",
        "TractorInvovled": "tractor", "TrainsInvovled": "train", "TrucksInvolved": "truck",
        "VansInvolved": "van", "OthersInvolved": "other vehicle",
    }
    records = []
    for event_id, rows in grouped.items():
        first = rows[0]
        vehicles = sorted({label for row in rows for column, label in vehicle_columns.items() if row.get(column) == "1"})
        causes = sorted({row.get("Cause", "").strip() for row in rows if row.get("Cause", "").strip()})
        injuries = sorted({row.get("InjuryType", "").strip() for row in rows if row.get("InjuryType", "").strip()})
        classes = sorted({row.get("ACCLASS", "").strip() for row in rows if row.get("ACCLASS", "").strip()})
        patient_statuses = sorted({row.get("PatientStatus", "").strip() for row in rows if row.get("PatientStatus", "").strip()})
        fatalities = sum(1 for row in rows if row.get("PatientStatus", "").strip().lower() == "dead")
        year, month, day = first.get("Year"), first.get("Month"), first.get("Date")
        try:
            date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except (TypeError, ValueError):
            date = None
        call_time = first.get("CallTime", "").strip()
        time = call_time.split()[-1] if call_time and " " in call_time else None
        severity = "Fatal" if "Fatal" in classes else (classes[0] if len(classes) == 1 else None)
        vehicle_text = ", ".join(vehicles) if vehicles else "vehicle type not recorded"
        cause_text = ", ".join(causes) if causes else "cause not recorded"
        description = f"Structured Rescue 1122 collision record involving {vehicle_text}; recorded cause: {cause_text}."
        records.append({
            "id": f"rwp-rta-{event_id}", "title": f"Rawalpindi road traffic collision {event_id}",
            "description": description, "description_type": "generated_from_verified_structured_fields",
            "date": date, "time": time, "location": "Rawalpindi", "city": "Rawalpindi",
            "district": "Rawalpindi", "province": "Punjab", "country": "Pakistan",
            "latitude": None, "longitude": None, "accident_type": "road traffic collision",
            "vehicle_types": vehicles, "fatalities": fatalities, "injuries": None, "severity": severity,
            "causes": causes, "injury_types": injuries, "patient_statuses": patient_statuses,
            "patients_in_source_rows": len(rows), "source_name": "Harvard Dataverse — Rawalpindi RTA Dataset",
            "source_id": "harvard-rawalpindi-rta-2020-2023", "source_url": DATAVERSE_PAGE,
            "image_path": None, "image_url_original": None, "image_license": None,
            "retrieved_at": RETRIEVED_AT, "image_verified_for_record": False,
        })
    records.sort(key=lambda item: (item.get("date") or "", item["id"]), reverse=True)
    return records


def build_media_records() -> tuple[list[dict], list[dict]]:
    record_map: dict[str, dict] = {}
    image_metadata = []
    for index, item in enumerate(COMMONS_MEDIA, 1):
        image_path = f"data/raw/accidents/images/{item['local_filename']}"
        record = record_map.setdefault(item["record_id"], {
            "id": item["record_id"], "title": item["title"].split(" — view")[0],
            "description": item["description"], "description_type": "source_supported_summary",
            "date": None, "photographed_at": item["date"], "date_basis": "Source photograph date available; incident date not independently established", "time": None, "location": item["location"], "city": item["city"],
            "district": item["district"], "province": item["province"], "country": "Pakistan",
            "latitude": item["latitude"], "longitude": item["longitude"],
            "accident_type": item["accident_type"], "vehicle_types": item["vehicle_types"],
            "fatalities": None, "injuries": None, "severity": None,
            "source_name": "Wikimedia Commons", "source_id": "wikimedia-pakistan-accident-media",
            "source_url": item["source_url"], "image_path": image_path,
            "image_url_original": commons_download_url(item["filename"]), "image_license": item["license"],
            "retrieved_at": RETRIEVED_AT, "image_verified_for_record": True, "image_paths": [],
        })
        record["image_paths"].append(image_path)
        image_metadata.append({
            "id": f"pk-acc-img-{index:03d}", "type": "accident_image", "title": item["title"],
            "description": item["description"], "description_type": "source_supported_summary",
            "city": item["city"], "district": item["district"], "province": item["province"],
            "country": "Pakistan", "accident_type": item["accident_type"], "vehicle_types": item["vehicle_types"],
            "date": item["date"], "record_id": item["record_id"], "source": "Wikimedia Commons",
            "source_id": "wikimedia-pakistan-accident-media", "source_url": item["source_url"],
            "author": item["author"], "license": item["license"], "image_path": image_path,
            "date_kind": "photograph", "license_url": {"CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/", "CC BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/", "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/"}.get(item["license"], "https://commons.wikimedia.org/wiki/Commons:Licensing#Public_domain"),
            "image_url_original": commons_download_url(item["filename"]), "verified_record_match": True,
            "retrieved_at": RETRIEVED_AT,
        })
    return list(record_map.values()), image_metadata


def normalize_rescue(html_path: Path) -> list[dict]:
    parser = RescueTableParser()
    html = html_path.read_text(encoding="utf-8", errors="replace")
    parser.feed(html)
    period = re.search(r"To\s+(\d{2})/(\d{2})/(\d{4})", html, re.IGNORECASE)
    period_end = f"{period[3]}-{period[2]}-{period[1]}" if period else None
    stats = []
    for row in parser.rows:
        if len(row) < 10 or row[0] in {"District", "Total"}:
            continue
        try:
            values = [int(value.replace(",", "")) for value in row[1:10]]
        except ValueError:
            continue
        stats.append(dict(zip(
            ["district", "emergency_calls", "road_accidents", "medical", "fire", "structure_collapse", "crime_incidents", "drowning", "miscellaneous", "patients_rescued"],
            [row[0], *values], strict=True,
        )))
        stats[-1].update({"type": "accident_statistics", "reporting_period_end": period_end, "source_url": RESCUE_URL, "retrieved_at": RETRIEVED_AT})
    return stats


def update_sources(record_count: int, image_count: int, rescue_count: int) -> None:
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    additions = [
        {
            "id": "harvard-rawalpindi-rta-2020-2023", "title": "Road Traffic Accident Dataset, Rawalpindi-Punjab, Pakistan",
            "organization": "Harvard Dataverse (depositor: Said Nabi)", "original_url": DATAVERSE_PAGE,
            "download_url": DATAVERSE_FILE, "retrieval_date": RETRIEVED_AT, "data_type": "Incident-level tabular accident records",
            "type": "accident_reports", "region": "Punjab", "geographic_scope": "Rawalpindi, Punjab, Pakistan",
            "official": False, "license": "CC0 1.0", "local_file_path": "data/raw/accidents/records/rawalpindi_rta.csv",
            "description": "Patient-level Rescue 1122 collision data normalized into event records; the release contains no photographs.",
            "records": record_count, "images": 0, "image_record_relationship": "none",
        },
        {
            "id": "rescue1122-punjab-performance", "title": "Consolidated Report of Emergency Calls and Rescue Operations in Punjab",
            "organization": "Punjab Emergency Service Department (Rescue 1122)", "original_url": RESCUE_URL,
            "download_url": RESCUE_URL, "retrieval_date": RETRIEVED_AT, "data_type": "Official district aggregate HTML table",
            "type": "accident_statistics", "region": "Punjab", "geographic_scope": "Punjab, Pakistan",
            "official": True, "license": "Not stated; official public information",
            "local_file_path": "data/processed/accidents/statistics/rescue1122_districts.json",
            "description": "District-level cumulative emergency and road-accident totals; not incident-level and contains no photographs.",
            "records": rescue_count, "images": 0, "image_record_relationship": "none",
        },
        {
            "id": "wikimedia-pakistan-accident-media", "title": "Openly licensed Pakistan accident photographs",
            "organization": "Wikimedia Commons contributors", "original_url": "https://commons.wikimedia.org/wiki/Category:Accidents_in_Pakistan",
            "download_url": "https://commons.wikimedia.org/wiki/Category:Accidents_in_Pakistan",
            "retrieval_date": RETRIEVED_AT, "data_type": "Accident reports with source-associated photographs",
            "type": "accident_reports_with_images", "region": "Pakistan", "geographic_scope": "Pakistan",
            "official": False, "license": "Per-file: Public domain, CC BY 3.0, CC BY-SA 3.0, or CC BY-SA 4.0",
            "local_file_path": "data/raw/accidents/images",
            "description": "Small curated set whose file descriptions, dates, geotags, and licenses support each record-image relationship.",
            "records": len({item['record_id'] for item in COMMONS_MEDIA}), "images": image_count, "image_record_relationship": "verified",
        },
    ]
    by_id = {source["id"]: source for source in sources}
    by_id.update({source["id"]: source for source in additions})
    SOURCES_PATH.write_text(json.dumps(list(by_id.values()), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for path in (RAW_RECORDS, RAW_IMAGES, RAW / "statistics", PROCESSED / "statistics", PROCESSED / "rag_documents"):
        path.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "LahoreTrafficIntelligence/1.0 (academic project; auditable public-data acquisition)"}
    with httpx.Client(follow_redirects=True, timeout=120, headers=headers) as client:
        csv_path = RAW_RECORDS / "rawalpindi_rta.csv"
        if csv_path.exists() and csv_path.stat().st_size:
            print(f"{csv_path.relative_to(ROOT)}: using existing verified download ({csv_path.stat().st_size:,} bytes)")
        else:
            download(client, DATAVERSE_FILE, csv_path)
        rescue_path = RAW / "statistics" / "rescue1122_performance.html"
        try:
            download(client, RESCUE_URL, rescue_path)
        except httpx.HTTPError as exc:
            print(f"Rescue 1122 direct download unavailable ({exc}); using the verified official-page snapshot")
        for item in COMMONS_MEDIA:
            destination = RAW_IMAGES / item["local_filename"]
            if not destination.exists():
                download(client, commons_download_url(item["filename"]), destination)

    structured_records = normalize_rawalpindi(csv_path)
    media_records, image_metadata = build_media_records()
    records_path = PROCESSED / "records.json"
    previous_records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    other_records = [record for record in previous_records if record.get("source_id") not in {"harvard-rawalpindi-rta-2020-2023", "wikimedia-pakistan-accident-media"}]
    records = sorted([*structured_records, *media_records, *other_records], key=lambda item: (item.get("date") or "", item["id"]), reverse=True)
    rescue_stats = normalize_rescue(rescue_path) if rescue_path.exists() else RESCUE_VERIFIED_SNAPSHOT
    (PROCESSED / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata_path = PROCESSED / "image_metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    other_images = [image for image in previous if image.get("source_id") != "wikimedia-pakistan-accident-media"]
    metadata_path.write_text(json.dumps([*image_metadata, *other_images], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (PROCESSED / "statistics" / "rescue1122_districts.json").write_text(json.dumps(rescue_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (PROCESSED / "rag_documents" / "accidents.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    update_sources(len(structured_records), len(image_metadata), len(rescue_stats))
    print(f"Normalized {len(records):,} records, {len(image_metadata)} verified images, and {len(rescue_stats)} district statistics rows")


if __name__ == "__main__":
    main()
