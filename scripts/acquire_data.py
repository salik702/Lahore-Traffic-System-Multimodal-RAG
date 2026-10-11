"""Download the small, auditable official corpus used by this project.

Run from the repository root: `uv run --project backend python scripts/acquire_data.py`.
The script respects ordinary HTTP access and does not bypass access controls.
"""

from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "documents"
SOURCES = {
    "punjab_motor_vehicles_ordinance_1965.pdf": "https://ctplahore.gop.pk/system/files/mvo_1965_0.pdf",
    "national_highways_safety_ordinance_2000.pdf": "https://nhmp.gov.pk/storage/trafficrules/NHSO-PDF-VERSION-BOOK.pdf",
    "nhmp_road_safety_syllabus.pdf": "https://nhmp.gov.pk/incldocs/All-Classes.pdf",
    "punjab_offences_penalties.html": "https://trafficpolice.punjab.gov.pk/offences_penalties",
}


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        for filename, url in SOURCES.items():
            destination = TARGET / filename
            response = client.get(url, headers={"User-Agent": "LahoreTrafficIntelligence/1.0 (academic project)"})
            response.raise_for_status()
            destination.write_bytes(response.content)
            print(f"{filename}: {len(response.content):,} bytes")


if __name__ == "__main__":
    main()

