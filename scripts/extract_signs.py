"""Extract the 24 individual sign images embedded on page 37 of the NHMP syllabus."""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "documents" / "nhmp_road_safety_syllabus.pdf"
TARGET = ROOT / "data" / "raw" / "traffic_signs"
NAMES = [
    "turn-left", "go-straight", "turn-right", "turn-right-alt", "turn-left-alt",
    "straight-or-left", "keep-right", "keep-left", "compulsory-roundabout", "lane-control",
    "dual-carriageway-ahead", "dual-carriageway-end", "left-bend", "right-bend",
    "double-bend-right", "double-bend-left", "level-crossing-without-gate",
    "level-crossing-with-gate", "carriageway-narrows", "narrow-bridge", "major-cross-road",
    "uneven-road", "road-dip", "advance-danger",
]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    document = fitz.open(SOURCE)
    images = document[36].get_images(full=True)
    if len(images) < len(NAMES):
        raise RuntimeError(f"Expected at least {len(NAMES)} embedded images, found {len(images)}")
    for name, image in zip(NAMES, images, strict=False):
        extracted = document.extract_image(image[0])
        destination = TARGET / f"{name}.{extracted['ext']}"
        destination.write_bytes(extracted["image"])
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
