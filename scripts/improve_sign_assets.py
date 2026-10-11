"""Render verified vector sign cells from the official NHMP CE handbook.

Run with backend/.venv/Scripts/python.exe scripts/improve_sign_assets.py.
Original syllabus images are retained. Crops were checked against each question's
diagram and answer in the handbook. Raster-only cells are deliberately skipped;
enlarging a low-resolution bitmap would not recover detail.
"""
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/documents/nhmp_ce_sign_handbook.pdf"
TARGET = ROOT / "data/raw/traffic_signs"
MANIFEST = ROOT / "data/processed/metadata/sign_assets.json"
SOURCE_ID = "nhmp-sign-test-handbook"
SOURCE_URL = "https://dla.nhmp.gov.pk/CEEnglish.pdf"
SOURCE_SHA256 = "178646ae10fbf62733426fbfb579bf510241d9eb6dce8b4151e73876a234baba"
# Physical PDF page, question number, and verified sign-cell crop in PDF points.
# Page numbers include the handbook's cover/front matter.
CELLS = {
    "compulsory-roundabout": (10, 22, [132, 59, 201, 112]),
    "no-entry": (7, 4, [132, 319, 201, 382]),
    "wild-animals": (8, 11, [132, 289, 201, 336]),
    "narrow-bridge": (8, 12, [132, 340, 201, 391]),
    "pedestrian-crossing": (8, 13, [132, 395, 201, 443]),
    "level-crossing-with-gate": (8, 14, [132, 447, 201, 520]),
    "dual-carriageway-ahead": (9, 16, [132, 126, 201, 195]),
    "no-u-turn": (9, 18, [132, 276, 201, 337]),
    "carriageway-narrows": (9, 19, [132, 341, 201, 403]),
    "no-horn": (9, 20, [132, 408, 201, 447]),
    "no-parking": (10, 25, [132, 226, 201, 278]),
    "motorway": (12, 36, [132, 117, 201, 165]),
    "speed-limit-50": (13, 43, [132, 60, 201, 128]),
    "no-left-turn": (14, 53, [132, 215, 201, 277]),
    "double-bend-left": (14, 55, [132, 347, 201, 406]),
    "advance-danger": (15, 59, [132, 114, 201, 155]),
    "hospital": (15, 62, [132, 286, 201, 337]),
    "petrol-station": (15, 63, [132, 341, 201, 381]),
    "road-dip": (15, 65, [132, 428, 201, 468]),
    "major-cross-road": (16, 74, [132, 430, 201, 480]),
    "uneven-road": (16, 75, [132, 484, 201, 525]),
    "animals-warning": (17, 76, [132, 60, 201, 101]),
    "level-crossing-without-gate": (17, 77, [132, 105, 201, 177]),
    "go-straight": (17, 80, [132, 331, 201, 374]),
    "turn-left": (17, 81, [132, 379, 201, 420]),
    "turn-right": (17, 82, [132, 425, 201, 466]),
    "straight-or-left": (17, 83, [132, 473, 201, 525]),
    "keep-left": (18, 85, [132, 123, 201, 169]),
    "keep-right": (18, 86, [132, 174, 201, 221]),
    "left-bend": (20, 99, [132, 151, 201, 221]),
    "right-bend": (20, 100, [132, 225, 201, 267]),
    "two-way-traffic": (20, 102, [132, 323, 201, 374]),
}


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Download the official handbook from {SOURCE_URL} to {SOURCE}")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("The source PDF changed. Review sign cells before updating the verified source hash.")
    signs = json.loads((ROOT / "data/raw/structured/signs.json").read_text(encoding="utf-8"))
    known = {s["id"] for s in signs}
    if not set(CELLS) <= known:
        raise ValueError("Crop manifest references unknown catalogue identities")
    assets = {}
    TARGET.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(SOURCE) as document:
        for sign_id, (number, question, coordinates) in CELLS.items():
            page, clip = document[number - 1], pymupdf.Rect(coordinates)
            # Include entire vector paths intersecting the checked cell. Some
            # arrowheads/rings extend beyond the initial visual crop; never clip
            # those or include table lines spanning neighbouring text columns.
            paths = [p for p in page.get_drawings()
                     if p["rect"].x0 >= 131 and p["rect"].x1 <= 202
                     and p["rect"].width > 1 and p["rect"].height > 1
                     and clip.intersects(p["rect"])]
            frames = [p["rect"] for p in paths if p.get("fill")
                      and p["fill"][0] > 0.7 and max(p["fill"][1:]) < 0.3
                      and p["rect"].width > 25 and p["rect"].height > 25]
            if not frames:
                print(f"No vector diagram: {sign_id}")
                continue
            # The nearest red frame identifies the checked diagram, rather than
            # a neighbouring sign that touches the approximate initial cell.
            center = (clip.y0 + clip.y1) / 2
            distance = lambda r: abs((r.y0 + r.y1) / 2 - center)
            nearest = min(map(distance, frames))
            bounds = max((r for r in frames if distance(r) <= nearest + 2), key=lambda r: r.width * r.height)
            clip = bounds + (-0.3, -0.3, 0.3, 0.3)
            if not page.rect.contains(clip):
                raise ValueError(f"Invalid crop: {sign_id}")
            # A vector crop should not include any embedded raster diagram.
            if any(clip.intersects(rect) for image in page.get_images() for rect in page.get_image_rects(image[0])):
                print(f"Kept original bitmap: {sign_id}")
                continue
            # Copy the original vector cell into a white PDF canvas. Margin is
            # added outside the crop, so tight table borders are not copied.
            with pymupdf.open() as canvas:
                sheet = canvas.new_page(width=clip.width + 6, height=clip.height + 6)
                sheet.show_pdf_page(pymupdf.Rect(3, 3, clip.width + 3, clip.height + 3), document, number - 1, clip=clip)
                scale = 512 / max(sheet.rect.width, sheet.rect.height)
                pix = sheet.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            filename = f"{sign_id}-nhmp-512.png"
            pix.save(TARGET / filename)
            assets[sign_id] = {"image": filename, "width": pix.width, "height": pix.height,
                               "source_id": SOURCE_ID, "page": number, "question": question,
                               "crop": list(clip), "extraction": "vector-render"}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"source_url": SOURCE_URL,
        "source_file": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": digest,
        "assets": assets}, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(assets)} verified vector signs; originals retained. Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
