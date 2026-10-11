"""Public evidence boundary: task scope, provenance, deduplication and citations."""
import json
import math
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

from app.services.router import normalize
from app.services.sign_matching import (SEMANTIC_MIN_MARGIN, SEMANTIC_MIN_SCORE, matches_sign,
    sign_subject, visual_description, visual_identity_compatible)

INTERNAL = re.compile(r"[A-Za-z]:[\\/]|file://|/(?:Users|home|tmp|mnt)/|\b(?:data/raw|backend/rag_storage)/|(?:^|\n)\s*(?:Image Path|Visual Analysis|Verified metadata|Section Path|Neighbor Text|Footnotes):|ACCIDENT (?:PROVENANCE|IMAGE|REPORT)\s*\||\"(?:image_path|record_id|file_path)\"", re.I)
INJECTION = re.compile(r"ignore (?:all |any |the |previous )*(?:instructions|prompts)|system\s*(?:message|prompt)\s*:|<\|(?:system|im_start)|(?:reveal|print|expose).*(?:api.?key|secret)|(?:assistant|developer)\s*:", re.I)


def safe_url(value: str | None) -> str | None:
    if not value or any(c in value for c in "\r\n<>\\"):
        return None
    try:
        parsed = urlparse(value)
        return value if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password else None
    except ValueError:
        return None


def evidence_kind(item: dict) -> str:
    meta = item.get("metadata", {})
    if meta.get("kind"):
        return meta["kind"]
    if item.get("id", "").startswith("rule-"):
        return "rule"
    if meta.get("category") == "accident_statistics" or item.get("id", "").startswith("rescue-stat-"):
        return "statistics"
    if "accident" in str(meta.get("category", "")).lower() or meta.get("record_id"):
        return "accident"
    if item.get("source_id") == "punjab-fines-2026":
        return "fine"
    if item.get("modality") == "image":
        return "sign"
    return "rule"


def eligible(item: dict, intent: str, query: str = "") -> bool:
    kind = evidence_kind(item)
    if intent in {"ACCIDENT_GUIDANCE", "EMERGENCY_GUIDANCE"}:
        return kind == "guidance" and item.get("metadata", {}).get("topic") in {"accident", "emergency", "scene_safety"} or kind == "rule" and item.get("metadata", {}).get("category") == "Accidents"
    if intent == "ACCIDENT_STATISTICS":
        return kind == "statistics"
    if intent in {"ACCIDENT_INFORMATION", "ACCIDENT_IMAGE_REQUEST"}:
        return kind == "accident"
    if intent in {"TRAFFIC_SIGN", "VISUAL_SEARCH"}:
        if kind == "sign":
            if item.get("source_id") == "project-illustration" and not sign_subject(query):
                return False  # Broad real-world sign explanations use published specimens.
            meta = item.get("metadata", {})
            visual_match = (meta.get("visual_retrieval_query") == query and visual_description(query)
                            and visual_identity_compatible(query, item))
            similarity, margin = meta.get("semantic_similarity"), meta.get("semantic_margin")
            semantic_match = (meta.get("semantic_retrieval_query") == query
                              and isinstance(similarity, (int, float)) and math.isfinite(similarity) and similarity >= SEMANTIC_MIN_SCORE
                              and isinstance(margin, (int, float)) and math.isfinite(margin) and margin >= SEMANTIC_MIN_MARGIN
                              and visual_identity_compatible(query, item))
            return matches_sign(query, item) or visual_match or semantic_match
        # Signal rules explain a signal; they are not pictures of named signs.
        signal = bool(re.search(r"\b(?:red|green|yellow)\b.*\b(?:traffic )?(?:signal|light)", query, re.I))
        return item.get("id") == "rule-red-signal" and signal and "red" in query.lower() or item.get("id") == "rule-signs" and not sign_subject(query)
    if intent == "TRAFFIC_FINE":
        return kind in {"fine", "rule"}
    if intent in {"OUT_OF_DOMAIN", "GENERAL_CONVERSATION", "GENERAL_KNOWLEDGE"}:
        return False
    return kind in {"rule", "guidance"}


def topic_terms(query: str) -> set[str]:
    q = normalize(query)
    topics = {
        "helmet": r"helmet", "documents": r"documents?|registration|permit|fitness|police.*stop|stopp?ed.*police",
        "license": r"licen[cs]e", "accident": r"accident|crash|collision|injur|hurt|ambulance|emergency",
        "sign": r"sign|symbol|signal|red light", "parking": r"park", "report": r"report.*(?:traffic|violation)|complaint",
        "pillion": r"pillion|passenger|overload", "age": r"age|old.*drive",
    }
    return {topic for topic, pattern in topics.items() if re.search(pattern, q)}


def select_evidence(items: list[dict], intent: str, query: str, source_map: dict, limit: int = 5) -> list[dict]:
    selected = []
    topics = topic_terms(query)
    for item in sorted(items, key=lambda value: value.get("score", 0), reverse=True):
        text = str(item.get("text", "")).strip()
        if not text or INTERNAL.search(text) or INJECTION.search(text) or not eligible(item, intent, query):
            continue
        source = source_map.get(item.get("source_id"))
        if not source or not safe_url(item.get("metadata", {}).get("source_url") or source.get("original_url")):
            continue
        score = float(item.get("score", 0))
        if not math.isfinite(score) or score < 0.32:
            continue
        # A high generic crash similarity cannot turn a different subject into guidance.
        if topics and not topics & topic_terms(item.get("title", "") + " " + text) and evidence_kind(item) not in {"accident", "statistics", "sign"}:
            continue
        key = " ".join(re.findall(r"\w+", text.lower()))
        if any(item.get("id") == other.get("id") or SequenceMatcher(None, key, other["_dedup"]).ratio() >= 0.9 for other in selected):
            continue
        selected.append({**item, "text": text[:1800], "_dedup": key})
        if len(selected) >= limit:
            break
    return [{k: v for k, v in item.items() if k != "_dedup"} for item in selected]


def build_context(evidence: list[dict], source_map: dict, budget: int = 12000) -> tuple[str, list[dict], list[dict]]:
    """Number sources before prompting; the answer and source list share one mapping."""
    citations, kept, blocks, source_numbers = [], [], [], {}
    used = 0
    for item in evidence:
        source = source_map.get(item.get("source_id"), {})
        meta = item.get("metadata", {})
        url = safe_url(meta.get("source_url") or source.get("original_url"))
        if not url:
            continue
        key = (url, meta.get("page"), meta.get("section"))
        number = source_numbers.get(key, len(citations) + 1)
        block = {"citation": number, "title": item["title"], "passage": item["text"], "scope": source.get("region"), "section": meta.get("section"), "page": meta.get("page")}
        if meta.get("example_only"):
            block["scope"] = "Catalogue specimen example; its speed value does not establish the limit on the user's road."
        encoded = json.dumps(block, ensure_ascii=False)
        if used + len(encoded) > budget:
            break
        used += len(encoded)
        if key not in source_numbers:
            source_numbers[key] = number
            citations.append({"id": source.get("id", item["source_id"]), "title": source.get("title", item["title"]), "organization": source.get("organization", "Source"), "url": url, "page": meta.get("page"), "section": meta.get("section"), "region": source.get("region", "Unknown"), "official": bool(source.get("official")), "modality": item.get("modality", "text"), "excerpt": item["text"][:600], "score": round(item["score"], 3)})
        kept.append({**item, "metadata": {**meta, "citation_number": number}})
        blocks.append(encoded)
    return "\n".join(blocks), citations, kept


def normalize_citations(answer: str) -> str:
    """Canonicalize numeric citation typography, without accepting new sources."""
    return re.sub(r"【\s*(\d+)\s*】|［\s*(\d+)\s*］", lambda match: f"[{match[1] or match[2]}]", answer)


def valid_answer(answer: str, citations: list[dict]) -> bool:
    if not answer.strip() or INTERNAL.search(answer) or INJECTION.search(answer) or re.search(r"```(?:json)?|\{\s*\"", answer):
        return False
    if any(int(number) < 1 or int(number) > len(citations) for number in re.findall(r"\[(\d+)\](?!\()", answer)):
        return False
    allowed = {item["url"] for item in citations}
    if any(not safe_url(url) or url not in allowed for url in re.findall(r"\]\(([^\s]+)\)", answer)):
        return False
    return not citations or bool(re.search(r"\[\d+\]", answer))


def public_evidence(evidence: list[dict]) -> list[dict]:
    keys = {"page", "section", "category", "citation_number"}
    return [{**item, "metadata": {k: v for k, v in item.get("metadata", {}).items() if k in keys}} for item in evidence]
