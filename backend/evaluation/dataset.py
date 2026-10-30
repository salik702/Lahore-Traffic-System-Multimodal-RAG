"""Ground-truth dataset generation for the Lahore Traffic RAG evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunk_mapping(settings: Any) -> dict[str, str]:
    """Map stable structured IDs to the exact IDs in the persisted chunk index."""
    index_path = settings.rag_storage / "vdb_chunks.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Persisted chunk index not found: {index_path}")
    rows = json.loads(index_path.read_text(encoding="utf-8")).get("data", [])
    mapping: dict[str, str] = {}
    for row in rows:
        content = str(row.get("content", ""))
        chunk_id = str(row.get("__id__", ""))
        if not chunk_id:
            continue
        parts = content.split(" | ")
        if content.startswith("RULE | ") and len(parts) > 1:
            title = parts[1].strip().casefold()
            mapping.setdefault(f"rule:{title}", chunk_id)
        elif content.startswith("FINE | ") and len(parts) > 2:
            mapping.setdefault(f"fine:{parts[2].removeprefix('code ').strip()}", chunk_id)
        elif content.startswith("IMAGE | ") and len(parts) > 1:
            mapping.setdefault(f"sign:{parts[1].strip().casefold()}", chunk_id)
    return mapping


def require_chunk(mapping: dict[str, str], key: str) -> str:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"No persisted chunk found for ground-truth key {key!r}") from exc


def generate_dataset() -> list[dict[str, Any]]:
    settings = get_settings()
    data_dir = settings.data_dir
    chunk_mapping = load_chunk_mapping(settings)

    rules = load_json(data_dir / "raw/structured/rules.json")
    fines = load_json(data_dir / "raw/structured/fines.json")
    signs = load_json(data_dir / "raw/structured/signs.json")

    dataset = []

    # 1. Easy Questions (Direct lookup)
    for rule in rules[:5]:
        chunk_id = require_chunk(chunk_mapping, f"rule:{rule['title'].casefold()}")
        dataset.append({
            "question": f"What does the law say about {rule['title'].lower()}?",
            "ground_truth_answer": rule["summary"],
            "relevant_document_ids": [chunk_id],
            "relevance_labels": {chunk_id: 3},
            "source_documents": [rule.get("source_id", "punjab-mvo-1965")],
            "difficulty": "easy",
            "category": "rule",
            "verified": True
        })

    for fine in fines[:5]:
        chunk_id = require_chunk(chunk_mapping, f"fine:{fine['code']}")
        dataset.append({
            "question": f"What is the penalty for {fine['violation'].lower()}?",
            "ground_truth_answer": f"The penalty is PKR {fine.get('motorcar_under_2000cc')} for motorcars.",
            "relevant_document_ids": [chunk_id],
            "relevance_labels": {chunk_id: 3},
            "source_documents": ["punjab-fines-2026"],
            "difficulty": "easy",
            "category": "fine",
            "verified": True
        })

    # 2. Medium Questions (Specific details)
    for sign in signs[:5]:
        chunk_id = require_chunk(chunk_mapping, f"sign:{sign['name'].casefold()}")
        dataset.append({
            "question": f"What is the meaning of the '{sign['name']}' traffic sign?",
            "ground_truth_answer": sign["meaning"],
            "relevant_document_ids": [chunk_id],
            "relevance_labels": {chunk_id: 3},
            "source_documents": [sign.get("source_id", "nhmp-road-safety-syllabus")],
            "difficulty": "medium",
            "category": "sign",
            "verified": True
        })

    # 3. Complex / Multi-document / Multi-chunk questions
    # Example: Combining a rule and a fine
    license_rule = require_chunk(chunk_mapping, "rule:effective driving licence required")
    license_fine = require_chunk(chunk_mapping, "fine:12")
    dataset.append({
        "question": "What are the rules and penalties for driving without a license in Punjab?",
        "ground_truth_answer": "According to Section 3 of the MVO 1965, an effective license is required. The penalty for driving without a license is PKR 5,000 for motorcars.",
        "relevant_document_ids": [license_rule, license_fine],
        "relevance_labels": {license_rule: 3, license_fine: 3},
        "source_documents": ["punjab-mvo-1965", "punjab-fines-2026"],
        "difficulty": "complex",
        "category": "multi-doc",
        "verified": True
    })

    accident_rule = require_chunk(chunk_mapping, "rule:stop and assist after an accident")
    dataset.append({
        "question": "What should I do after a road accident and who should I call for medical help in Punjab?",
        "ground_truth_answer": "You should stop and assist (Section 92/94). For medical help in Punjab, call Rescue 1122.",
        "relevant_document_ids": [accident_rule],
        "relevance_labels": {accident_rule: 3},
        "source_documents": ["punjab-mvo-1965", "rescue1122-emergency"],
        "difficulty": "complex",
        "category": "multi-doc",
        "verified": True
    })

    # 4. Out of Domain / Answer not present
    dataset.append({
        "question": "What is the best restaurant in Gulberg, Lahore?",
        "ground_truth_answer": "I do not have information about specific restaurant recommendations in Gulberg.",
        "relevant_document_ids": [],
        "relevance_labels": {},
        "source_documents": [],
        "difficulty": "medium",
        "category": "out-of-domain",
        "verified": True
    })

    # Save to file
    output_path = Path(__file__).parent / "dataset.json"
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Generated dataset with {len(dataset)} examples at {output_path}")
    return dataset


if __name__ == "__main__":
    generate_dataset()
