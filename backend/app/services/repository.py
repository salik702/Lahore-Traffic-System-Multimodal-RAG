import json
import logging
import re
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.services.sign_matching import (PROTECTED_WORDS, SEMANTIC_MIN_MARGIN, SEMANTIC_MIN_SCORE,
    catalogue_vocabulary, correct_sign_query, matches_sign, one_edit_apart, sign_subject,
    visual_description, visual_identity_compatible, visual_identity_terms)
from app.services.image_embeddings import ImageEmbeddingIndex, read_image

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {"a", "an", "and", "are", "at", "be", "do", "does", "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "this", "to", "what", "when", "where", "which", "who", "why", "with", "without", "would"}
VISUAL_COLORS = {"red", "blue", "green", "yellow", "white", "black"}


@lru_cache(maxsize=128)
def sign_colors(path: str, modified: int, size: int) -> frozenset[str]:
    pixels = np.asarray(read_image(Path(path)).resize((128, 128)), dtype=np.float32)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    masks = {"red": (r > 100) & (r > g * 1.25) & (r > b * 1.25),
             "blue": (b > 100) & (b > r * 1.25) & (b > g * 1.25),
             "green": (g > 100) & (g > r * 1.25) & (g > b * 1.25),
             "yellow": (r > 100) & (g > 100) & (b < 100),
             "white": pixels.min(axis=-1) > 220, "black": pixels.max(axis=-1) < 55}
    return frozenset(color for color, mask in masks.items() if mask.mean() >= 0.015)


def tokens(value: str) -> set[str]:
    words = set(TOKEN_RE.findall(value.lower())) - STOP_WORDS
    plurals = {"accidents": "accident", "crashes": "crash", "images": "image", "reports": "report", "motorcycles": "motorcycle", "cars": "car", "trucks": "truck", "buses": "bus"}
    plurals.update({"motorbike": "motorcycle", "motorbikes": "motorcycle", "motorcyclist": "motorcycle", "motorcyclists": "motorcycle"})
    words = {plurals.get(word, word) for word in words}
    aliases = {"bike": "motorcycle", "bikes": "motorcycle", "pictures": "image", "picture": "image", "photos": "image", "photo": "image", "challan": "fine", "licence": "license", "phone": "mobile", "redlight": "signal", "topi": "helmet", "gaari": "vehicle", "band": "closed", "hadsa": "accident"}
    return words | {aliases[word] for word in words if word in aliases}


class KnowledgeRepository:
    """Verified corpus with local neural embeddings and lexical backoff."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sources = self._load("sources/sources.json")
        self.rules = self._load("raw/structured/rules.json")
        self.fines = self._load("raw/structured/fines.json")
        self.signs = self._load("raw/structured/signs.json")
        self._sign_vocabulary = catalogue_vocabulary(self.signs)
        asset_manifest = settings.data_dir / "processed/metadata/sign_assets.json"
        self.sign_assets = json.loads(asset_manifest.read_text(encoding="utf-8")).get("assets", {}) if asset_manifest.is_file() else {}
        self.guidance = self._load_optional("raw/structured/guidance.json")
        self.accidents = self._load_optional("processed/accidents/records.json")
        self.accident_images = self._load_optional("processed/accidents/image_metadata.json")
        self.accident_statistics = self._load_optional("processed/accidents/statistics/rescue1122_districts.json")
        self.accident_image_map: dict[str, list[dict[str, Any]]] = {}
        for image in self.accident_images:
            if image.get("record_id") and image.get("verified_record_match") and self._accident_image_url(image.get("image_path")):
                self.accident_image_map.setdefault(image["record_id"], []).append(image)
        self.accident_by_id = {record["id"]: record for record in self.accidents}
        self.accident_location_tokens = set().union(*(tokens(" ".join(str(record.get(field) or "") for field in ("city", "district", "province", "country"))) for record in self.accidents))
        self.source_map = {item["id"]: item for item in self.sources}
        self._items = self._build_items()
        self._embedding_model = None
        self._vectors: np.ndarray | None = None
        self.semantic_error: str | None = None
        self._route_vectors = None
        self.image_index = ImageEmbeddingIndex(settings)

    def _load(self, relative: str) -> list[dict[str, Any]]:
        return json.loads((self.settings.data_dir / relative).read_text(encoding="utf-8"))

    def _load_optional(self, relative: str) -> list[dict[str, Any]]:
        path = self.settings.data_dir / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def _sign_image_url(self, filename: str | None) -> str | None:
        if not filename or Path(filename).name != filename:
            return None
        image_path = self.settings.data_dir / "raw" / "traffic_signs" / filename
        return f"/media/signs/{filename}" if image_path.is_file() else None

    def _accident_image_url(self, image_path: str | None) -> str | None:
        if not image_path:
            return None
        if ".." in image_path.replace("\\", "/").split("/") or re.match(r"^[A-Za-z]:", image_path):
            return None
        filename = Path(image_path).name
        if filename != image_path.replace("\\", "/").split("/")[-1]:
            return None
        path = self.settings.data_dir / "raw" / "accidents" / "images" / filename
        return f"/media/accidents/{filename}" if path.is_file() else None

    def _sign_media(self, sign: dict) -> dict:
        asset = self.sign_assets.get(sign["id"], {})
        preferred = self._sign_image_url(asset.get("image"))
        filename = asset.get("image") if preferred else sign.get("image")
        url = preferred or self._sign_image_url(filename)
        width = height = None
        if url and filename and not filename.endswith(".svg"):
            try:
                with Image.open(self.settings.data_dir / "raw/traffic_signs" / filename) as image:
                    width, height = image.size
            except (OSError, ValueError):
                url = None
        source_id = asset.get("source_id") if preferred else sign.get("source_id", "nhmp-road-safety-syllabus")
        page = asset.get("page") if preferred else sign.get("source_page", 37 if source_id == "nhmp-road-safety-syllabus" else None)
        source_url = self.source_map.get(source_id, {}).get("original_url")
        return {"image_url": url, "image_width": width, "image_height": height,
                "image_source_id": source_id, "image_source_page": page,
                "image_source_url": source_url + (f"#page={page}" if page else "") if source_url else None}

    def _build_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for entry in self.guidance:
            items.append({"id": entry["id"], "title": entry["title"], "text": entry["summary"], "search_text": f"{entry['title']} {entry['summary']} {' '.join(entry.get('keywords', []))}", "modality": "text", "source_id": entry["source_id"], "metadata": {"kind": "guidance", "topic": entry["topic"], "page": entry.get("page"), "section": entry.get("section")}})
        for rule in self.rules:
            items.append({"id": rule["id"], "title": rule["title"], "text": rule["summary"], "search_text": f"{rule['title']} {rule['summary']} {rule['category']}", "modality": "text", "source_id": rule["source_id"], "metadata": {"page": rule["page"], "section": rule["section"], "category": rule["category"]}})
        for fine in self.fines:
            amounts = {key: fine.get(key) for key in ("motorcycle", "three_wheeler", "motorcar_under_2000cc", "motorcar_over_2000cc", "transport_vehicle")}
            amount_text = ", ".join(f"{key.replace('_', ' ')}: {value if value is not None else 'not listed'}" for key, value in amounts.items())
            items.append({"id": f"fine-{fine['code']}", "title": fine["violation"], "text": f"Official Punjab fines (PKR): {amount_text}", "search_text": f"fine penalty challan {fine['violation']} {' '.join(fine['keywords'])} {amount_text}", "modality": "table", "source_id": "punjab-fines-2026", "metadata": {"code": fine["code"], **amounts}})
        for sign in self.signs:
            page = sign.get("source_page", 37)
            items.append({"id": sign["id"], "title": sign["name"], "text": sign["meaning"], "search_text": f"traffic road sign symbol {sign['name']} {sign['meaning']} {sign['category']} {' '.join(sign.get('aliases', []))}", "modality": "image", "source_id": sign.get("source_id", "nhmp-road-safety-syllabus"), "metadata": {"kind": "sign", "aliases": sign.get("aliases", []), "example_only": sign.get("example_only", False), "page": page if sign.get("source_id") is None else None, "category": sign["category"], **self._sign_media(sign)}})
        # Image-backed reports participate in the general semantic index. The much larger
        # tabular corpus has a dedicated filtered retrieval path and is also seeded into RAG.
        for record in self.accidents:
            images = self.accident_image_map.get(record["id"], [])
            if not images:
                continue
            image_url = self._accident_image_url(images[0].get("image_path"))
            search_text = " ".join(str(value) for value in [record.get("title"), record.get("description"), record.get("city"), record.get("district"), record.get("province"), record.get("accident_type"), " ".join(record.get("vehicle_types") or []), "accident crash image photograph"] if value)
            items.append({"id": record["id"], "title": record["title"], "text": record["description"], "search_text": search_text, "modality": "image", "source_id": record["source_id"], "metadata": {"category": "Accident report with verified image", "image_url": image_url, "record_id": record["id"], "date": record.get("date"), "location": record.get("location"), "city": record.get("city"), "province": record.get("province"), "source_url": record.get("source_url"), "license": record.get("image_license"), "verified_record_match": True}})
        for statistic in self.accident_statistics:
            district = statistic.get("district", "Punjab")
            text = f"Rescue 1122 recorded {statistic.get('road_accidents')} road accidents in {district} for the published cumulative reporting period ending {statistic.get('reporting_period_end', 'on the source page')}."
            items.append({"id": f"rescue-stat-{district.lower().replace(' ', '-')}", "title": f"Rescue 1122 road-accident total — {district}", "text": text, "search_text": f"accident statistics aggregate {district} Punjab Pakistan {text}", "modality": "table", "source_id": "rescue1122-punjab-performance", "metadata": {"district": district, "category": "accident_statistics"}})
        for image in self.accident_images:
            if image.get("record_id"):
                continue
            items.append({"id": image["id"], "title": image["title"], "text": image["description"],
                          "search_text": f"generic accident crash image {image['title']} {image['description']}", "modality": "image", "source_id": image["source_id"],
                          "metadata": {**image, "category": "Generic accident dataset imagery; location unverified", "image_url": self._accident_image_url(image.get("image_path"))}})
        return items

    def _ensure_vectors(self) -> bool:
        if self._vectors is not None:
            return True
        try:
            texts = [item["search_text"] for item in self._items]
            self._vectors = self._embed(texts)
            self._embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            return True
        except Exception as exc:
            self.semantic_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Semantic embedding initialization failed")
            return False

    def _embed(self, texts: list[str]) -> np.ndarray:
        request = urllib.request.Request(f"{self.settings.semantic_worker_url}/embed", data=json.dumps({"texts": texts}).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=90) as response:
            vectors = np.asarray(json.loads(response.read())["embeddings"], dtype=np.float32)
        if vectors.shape != (len(texts), self.settings.embedding_dimension) or not np.isfinite(vectors).all():
            raise ValueError("Embedding service returned incompatible vectors")
        return vectors

    def classify_semantically(self, query: str) -> tuple[str, float, float]:
        from app.services.router import PROTOTYPES
        if self._route_vectors is None:
            self._route_vectors = self._embed(list(PROTOTYPES.values()))
        scores = self._route_vectors @ self._embed([query])[0]
        order = np.argsort(scores)[::-1]
        return list(PROTOTYPES)[order[0]], float(scores[order[0]]), float(scores[order[0]] - scores[order[1]])

    def list_fines(self, query: str = "") -> list[dict[str, Any]]:
        if not query:
            return self.fines
        wanted = tokens(query)
        return [item for item in self.fines if wanted & tokens(item["violation"] + " " + " ".join(item["keywords"]))]

    def list_rules(self, query: str = "", category: str = "") -> list[dict[str, Any]]:
        items = self.rules
        if category:
            items = [item for item in items if item["category"].lower() == category.lower()]
        if query:
            wanted = tokens(query)
            items = [item for item in items if wanted & tokens(item["title"] + " " + item["summary"])]
        return items

    def list_signs(self, query: str = "", category: str = "") -> list[dict[str, Any]]:
        query, _ = self.normalize_sign_query(query)
        items = self.signs
        if category:
            items = [item for item in items if item["category"].lower() == category.lower()]
        if query:
            items = [item for item in items if matches_sign(query, item)]
            if not items:
                hits = self.search_sign_pixels(query, 1) if visual_description(query) else self.search_sign_semantically(query)
                ids = {hit["id"] for hit in hits}
                items = [item for item in self.signs if item["id"] in ids and (not category or item["category"].lower() == category.lower())]
        return [{**item, **self._sign_media(item), "source_id": item.get("source_id", "nhmp-road-safety-syllabus"), "source_page": item.get("source_page", 37)} for item in items]

    def normalize_sign_query(self, query: str) -> tuple[str, dict[str, str]]:
        return correct_sign_query(query, self._sign_vocabulary)

    def _public_accident(self, record: dict[str, Any]) -> dict[str, Any]:
        images = [{**{key: value for key, value in image.items() if key != "image_path"}, "image_url": self._accident_image_url(image.get("image_path"))} for image in self.accident_image_map.get(record["id"], []) if image.get("verified_record_match") and image.get("record_id") == record["id"] and self._accident_image_url(image.get("image_path"))]
        public = {key: value for key, value in record.items() if key not in {"image_path", "image_paths", "report_path"}}
        return {**public, "image_url": images[0]["image_url"] if images else None, "images": images}

    def get_accident(self, record_id: str) -> dict[str, Any] | None:
        record = self.accident_by_id.get(record_id)
        return self._public_accident(record) if record else None

    def search_accidents(self, query: str = "", *, city: str = "", province: str = "", accident_type: str = "", vehicle_type: str = "", has_image: bool | None = None, limit: int = 20, offset: int = 0, semantic_ids: list[str] | None = None, date_from: str | None = None, date_to: str | None = None) -> tuple[list[dict[str, Any]], int]:
        query_tokens = tokens(query)
        meaningful = query_tokens - {"show", "accident", "crash", "road", "report", "image", "indexed", "latest", "near", "involving", "please"}
        semantic_rank = {}
        for index, record_id in enumerate(semantic_ids or []):
            semantic_rank.setdefault(record_id, 1 - index / max(len(semantic_ids or []), 1))
        vehicle_terms = {"motorcycle", "car", "truck", "bus", "van", "rickshaw", "bicycle", "tractor", "train"}
        requested_vehicles = meaningful & vehicle_terms
        location_terms = self.accident_location_tokens
        requested_locations = meaningful & location_terms
        explicit_place = re.search(r"\b(?:in|from|near)\s+([a-z][a-z .-]*)", query.lower())
        scene_description = bool(explicit_place and re.match(r"(?:a|an|the)\s+(?:river|stream|road|street|ditch|field|junction|intersection|fog|rain)\b", explicit_place[1]))
        if explicit_place and not requested_locations and not scene_description:
            # An unknown place must not silently match every record in the catalogue.
            return [], 0
        ranked = []
        for record in self.accidents:
            date = record.get("date")
            if date_from and (not date or date < date_from):
                continue
            if date_to and (not date or date > date_to):
                continue
            images = self.accident_image_map.get(record["id"], [])
            if has_image is True and not images:
                continue
            if has_image is False and images:
                continue
            if city and (record.get("city") or "").lower() != city.lower():
                continue
            if province and (record.get("province") or "").lower() != province.lower():
                continue
            if accident_type and accident_type.lower() not in (record.get("accident_type") or "").lower():
                continue
            vehicles = {value.lower() for value in record.get("vehicle_types") or []}
            if vehicle_type and vehicle_type.lower() not in vehicles:
                continue
            vehicle_match = bool(requested_vehicles & vehicles) if re.search(r"\bor\b", query.lower()) else requested_vehicles <= vehicles
            if requested_vehicles and not vehicle_match:
                continue
            haystack = " ".join(str(value) for value in [record.get("title"), record.get("description"), record.get("location"), record.get("city"), record.get("district"), record.get("province"), record.get("country"), record.get("accident_type"), " ".join(record.get("vehicle_types") or []), " ".join(record.get("causes") or [])] if value)
            hay_tokens = tokens(haystack)
            if requested_locations and not requested_locations <= hay_tokens:
                continue
            severity = str(record.get("severity", "")).lower()
            if "serious" in meaningful and severity not in {"fatal", "severe", "serious"}:
                continue
            if "fatal" in meaningful and severity != "fatal":
                continue
            score = len(meaningful & hay_tokens) / max(len(meaningful), 1)
            score += semantic_rank.get(record["id"], 0) * 2
            if images:
                score += 0.2 if "image" in query_tokens else 0.04
            ranked.append((score, record))
        latest = "latest" in query_tokens or "newest" in query_tokens
        ranked.sort(key=lambda pair: ((pair[1].get("date") or "") if latest else pair[0], pair[0], pair[1].get("date") or ""), reverse=True)
        return [self._public_accident(record) for _, record in ranked[offset:offset + limit]], len(ranked)

    def list_accident_images(self, query: str = "", limit: int = 12, semantic_ids: list[str] | None = None) -> list[dict[str, Any]]:
        records, _ = self.search_accidents(query, has_image=True, limit=limit, semantic_ids=semantic_ids)
        verified = [image for record in records for image in record["images"]]
        return [*verified, *self.generic_accident_images(query, limit)][:limit]

    def generic_accident_images(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        wanted = tokens(query)
        # Collection images have no city, vehicle or incident facts. They must
        # never satisfy a factual filter on that absent metadata.
        restricted = {"lahore", "punjab", "rawalpindi", "islamabad", "gilgit", "malakand", "motorcycle", "car", "truck", "bus", "bike", "bicycle", "serious", "fatal", "latest", "indexed", "report"}
        if wanted & restricted:
            return []
        if re.search(r"\b(?:in|from|near)\s+\w+", query.lower()):
            return []
        return [{**{key: value for key, value in image.items() if key != "image_path"}, "image_url": self._accident_image_url(image.get("image_path"))} for image in self.accident_images
                if not image.get("record_id") and self._accident_image_url(image.get("image_path"))][:limit]

    def accident_facets(self):
        return {"cities": sorted({record["city"] for record in self.accidents if record.get("city")}),
                "vehicle_types": sorted({vehicle for record in self.accidents for vehicle in record.get("vehicle_types", [])}),
                "accident_types": sorted({record["accident_type"] for record in self.accidents if record.get("accident_type")})}

    def search(self, query: str, top_k: int = 5, *, intent: str | None = None) -> list[dict[str, Any]]:
        from app.services.evidence import eligible, topic_terms
        from app.services.router import detect_intent
        intent = intent or detect_intent(query)
        if intent in {"TRAFFIC_SIGN", "VISUAL_SEARCH"}:
            query, _ = self.normalize_sign_query(query)
            if sign_subject(query) and not any(eligible(i, intent, query) for i in self._items):
                # Pixel descriptions and functional descriptions have different
                # confidence checks; neither bypasses specimen identity constraints.
                return self.search_sign_pixels(query, top_k) if visual_description(query) else self.search_sign_semantically(query)
        wanted = tokens(query)
        if intent == "ACCIDENT_STATISTICS":
            known_districts = set().union(*(tokens(s.get("district", "")) for s in self.accident_statistics))
            if not wanted & known_districts:
                return []
        # Prefer exact subject matches for fine questions (e.g. "helmet fine")
        # over semantically similar fine rows that happen to share the same
        # vehicle amounts.
        fine_subjects = [item for item in self._items
                         if item.get("source_id") == "punjab-fines-2026"
                         and (topic_terms(query) & topic_terms(item["title"]) or (wanted - {"fine", "penalty", "challan", "official", "punjab", "pkr", "motorcycle", "car", "vehicle", "lahore"}) & tokens(item["title"]))]
        if fine_subjects:
            subject_ids = {item["id"] for item in fine_subjects}
            candidates = [item for item in self._items
                          if item.get("source_id") != "punjab-fines-2026" or item["id"] in subject_ids]
        else:
            candidates = self._items
        semantic_scores = np.zeros(len(self._items), dtype=np.float32)
        semantic_available = False
        try:
            if self._ensure_vectors() and self._embedding_model is not None:
                query_vector = self._embed([query])[0]
                semantic_scores = self._vectors @ query_vector
                semantic_available = True
        except Exception as exc:
            self.semantic_error = type(exc).__name__
            logger.warning("query_embedding_failed_using_lexical", exc_info=True)
        ranked = []
        for index, item in enumerate(self._items):
            if item not in candidates:
                continue
            if not eligible(item, intent, query):
                continue
            if intent == "ACCIDENT_STATISTICS":
                requested = tokens(query) & set().union(*(tokens(s.get("district", "")) for s in self.accident_statistics))
                if requested and not requested <= tokens(item.get("metadata", {}).get("district", "")):
                    continue
            lexical = len(wanted & tokens(item["search_text"])) / max(len(wanted), 1)
            score = float(0.82 * semantic_scores[index] + 0.18 * lexical) if semantic_available else lexical
            topic_match = bool(topic_terms(query) & topic_terms(item["title"] + " " + item["text"]))
            if intent in {"TRAFFIC_SIGN", "VISUAL_SEARCH"} and item.get("metadata", {}).get("kind") == "sign":
                # Identity has already been checked by eligible(). Generic sign
                # words must neither outrank nor discard a matching named sign.
                topic_match = True
                if sign_subject(query):
                    score = max(score, 0.9)
            if topic_match:
                score = max(score, 0.52 + lexical * 0.12)
            if not topic_match and lexical < 0.15 and float(semantic_scores[index]) < 0.62:
                continue
            ranked.append({key: value for key, value in item.items() if key != "search_text"} | {"score": max(0.0, score)})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def _image_paths(self, item: dict) -> list[Path]:
        meta = item.get("metadata", {})
        paths = []
        for image in meta.get("images") or [meta]:
            url = image.get("image_url", "") or ""
            if image.get("record_id") and (not image.get("verified_record_match") or image["record_id"] != item["id"]):
                continue
            if url.startswith("/media/signs/"):
                filename = url.removeprefix("/media/signs/")
                if self._sign_image_url(filename) == url:
                    paths.append(self.settings.data_dir / "raw/traffic_signs" / filename)
            elif url.startswith("/media/accidents/"):
                filename = url.removeprefix("/media/accidents/")
                if self._accident_image_url(filename) == url:
                    paths.append(self.settings.data_dir / "raw/accidents/images" / filename)
        return paths

    def search_sign_semantically(self, query: str) -> list[dict]:
        """Resolve an unnamed function from existing source-backed sign passages.

        Require both absolute relevance and separation from all other signs,
        before checking constraints. Filtering first would manufacture a winner
        for a missing limit or a contradictory direction.
        """
        subject = sign_subject(query)
        if not subject or any(word not in self._sign_vocabulary and
                              any(one_edit_apart(word, protected) for protected in PROTECTED_WORDS)
                              for word in subject):
            return []  # Never guess a misspelled direction or negation.
        try:
            if not self._ensure_vectors():
                return []
            vector = self._embed([query])[0]
            scores = self._vectors @ vector
            order = sorted((index for index, item in enumerate(self._items)
                            if item.get("metadata", {}).get("kind") == "sign"
                            and self.source_map.get(item["source_id"], {}).get("official")),
                           key=lambda index: float(scores[index]), reverse=True)
            if len(order) < 2:
                return []
            score = float(scores[order[0]])
            margin = score - float(scores[order[1]])
            winner = self._items[order[0]]
            if (not np.isfinite([score, margin]).all() or score < SEMANTIC_MIN_SCORE or margin < SEMANTIC_MIN_MARGIN
                    or not visual_identity_compatible(query, winner, self._sign_vocabulary)
                    or not self._sign_colors_match(query, winner)):
                return []
            logger.info("semantic_sign_match %s", json.dumps({"id": winner["id"], "cosine": score, "margin": margin}))
            return [{key: value for key, value in winner.items() if key != "search_text"} | {
                "score": score, "metadata": {**winner["metadata"], "semantic_retrieval_query": query,
                    "semantic_identity_terms": sorted(visual_identity_terms(query, self._sign_vocabulary)),
                    "semantic_similarity": score, "semantic_margin": margin}}]
        except Exception:
            logger.warning("semantic_sign_resolution_unavailable", exc_info=True)
            return []

    def search_sign_pixels(self, query: str, top_k: int) -> list[dict]:
        # Unnamed visual descriptions still use real catalogue identities and meanings.
        signs = [i for i in self._items if i.get("metadata", {}).get("kind") == "sign"
                 and i["source_id"] != "project-illustration" and visual_identity_compatible(query, i, self._sign_vocabulary)
                 and self._sign_colors_match(query, i)]
        scores = self.image_index.scores(self._image_query(query, "TRAFFIC_SIGN"), {i["id"]: self._image_paths(i) for i in signs})
        allowed = {i["id"] for i in signs}
        scores = {key: value for key, value in scores.items() if key in allowed and np.isfinite(value)}
        ordered = sorted(scores, key=scores.get, reverse=True)
        if not ordered or scores[ordered[0]] < self.settings.image_similarity_threshold:
            return []
        winner_id = ordered[0]
        grounded = False
        try:
            if self._ensure_vectors():
                vector = self._embed([query])[0]
                text_scores = {i["id"]: float(self._vectors[self._items.index(i)] @ vector) for i in signs if i["id"] in scores}
                text_order = sorted(text_scores, key=text_scores.get, reverse=True)
                margin = text_scores[text_order[0]] - text_scores[text_order[1]] if len(text_order) > 1 else 1
                if text_scores[text_order[0]] >= 0.55 and margin >= 0.08 and scores[text_order[0]] >= self.settings.image_similarity_threshold:
                    winner_id, grounded = text_order[0], True
        except Exception:
            logger.warning("image_text_grounding_unavailable_using_pixels")
        if not grounded and len(ordered) > 1 and scores[ordered[0]] - scores[ordered[1]] < self.settings.image_similarity_margin:
            return []  # Closely tied different signs need a clearer description.
        winner = next(i for i in signs if i["id"] == winner_id)
        logger.info("visual_sign_match %s", json.dumps({"id": winner["id"], "cosine": scores[winner_id], "text_grounded": grounded}))
        return [{k: v for k, v in winner.items() if k != "search_text"} | {"score": 0.8,
                "metadata": {**winner["metadata"], "visual_retrieval_query": query,
                             "visual_identity_terms": sorted(visual_identity_terms(query, self._sign_vocabulary)),
                             "image_similarity": scores[winner_id]}}]

    def _sign_colors_match(self, query: str, item: dict) -> bool:
        requested = sign_subject(query) & VISUAL_COLORS
        if not requested:
            return True
        try:
            paths = self._image_paths(item)
            return bool(paths and requested <= sign_colors(str(paths[0]), paths[0].stat().st_mtime_ns, paths[0].stat().st_size))
        except (OSError, ValueError):
            return False

    def _image_query(self, query: str, intent: str) -> str:
        # Geography is checked in metadata first; CLIP encodes the depicted scene.
        query = query.split(". Follow-up:")[0]
        query = re.sub(r"^(?:please\s+)?(?:show|display|find|retrieve|give)(?:\s+me)?\s+", "", query, flags=re.I)
        if intent == "ACCIDENT_IMAGE_REQUEST":
            places = "|".join(re.escape(place) for place in sorted(self.accident_location_tokens, key=len, reverse=True))
            query = re.sub(rf"\b(?:{places})\b", "", query, flags=re.I)
            query = re.sub(r"\b(?:a |an |the )?(?:images?|photos?|photographs?|pictures?)(?:\s+of)?\b", "", query, flags=re.I)
            query = re.sub(r"\b(?:from|near|in)\s*[?.!]*$", "", query)
            query = re.sub(r"\b(?:road\s+)?accidents?\b", "road accident", query)
            query = re.sub(r"^\s*(?:a|an|the)\s+", "", query)
            query = f"a photograph of {query}"
        else:
            query = re.sub(r"^(?:a |an |the )?sign\b", "a traffic sign", query)
        return " ".join(query.strip("?.! ").split())

    def rank_image_evidence(self, query: str, items: list[dict], intent: str) -> list[dict]:
        """Fuse independent visual/text ranks; factual constraints precede scoring."""
        if intent not in {"TRAFFIC_SIGN", "VISUAL_SEARCH", "ACCIDENT_IMAGE_REQUEST"}:
            return items
        from app.services.evidence import eligible
        candidates = [i for i in items if eligible(i, intent, query) and self._image_paths(i)]
        visual_query = self._image_query(query, intent)
        scores = self.image_index.scores(visual_query, {i["id"]: self._image_paths(i) for i in candidates})
        allowed = {i["id"] for i in candidates}
        scores = {key: value for key, value in scores.items() if key in allowed and np.isfinite(value)}
        if not scores:
            return items
        # Exact named signs have independently verified identity even when CLIP
        # is weak on a small symbol. Other images require minimum visual support.
        threshold = self.settings.image_photo_similarity_threshold if intent == "ACCIDENT_IMAGE_REQUEST" else self.settings.image_similarity_threshold
        confident = {key: value for key, value in scores.items() if value >= threshold}
        if intent == "ACCIDENT_IMAGE_REQUEST":
            items = [i for i in items if i["id"] in confident]
        if not confident or not items:
            return items
        text_order = list(dict.fromkeys(i["id"] for i in sorted(items, key=lambda i: i["score"], reverse=True)))
        visual_order = sorted(confident, key=confident.get, reverse=True)
        text_rank = {key: rank + 1 for rank, key in enumerate(text_order)}
        visual_rank = {key: rank + 1 for rank, key in enumerate(visual_order)}
        fusion = {key: 1 / (60 + rank) + (2 / (60 + visual_rank[key]) if key in visual_rank else 0)
                  for key, rank in text_rank.items()}
        maximum = max(fusion.values())
        ranked = []
        for item in items:
            metadata = {**item.get("metadata", {}), "image_similarity": scores.get(item["id"]), "image_rank_fusion": True}
            attached = metadata.get("images") or []
            if intent == "ACCIDENT_IMAGE_REQUEST" and len(attached) > 1:
                views = {f"view-{n}": self._image_paths({**item, "metadata": {**metadata, "images": [image]}})
                         for n, image in enumerate(attached)}
                view_scores = self.image_index.scores(visual_query, views)
                if view_scores:
                    supported = [(view_scores[f"view-{n}"], image) for n, image in enumerate(attached)
                                 if view_scores.get(f"view-{n}", -1) >= threshold]
                    metadata["images"] = [image for _, image in sorted(supported, key=lambda pair: pair[0], reverse=True)]
                    metadata["image_url"] = metadata["images"][0]["image_url"] if metadata["images"] else None
            ranked.append({**item, "score": 0.52 + 0.38 * fusion[item["id"]] / maximum, "metadata": metadata})
        logger.info("image_rank_fusion %s", json.dumps({"intent": intent, "candidate_count": len(candidates), "supported_count": len(confident), "visual_scores": scores}))
        return sorted(ranked, key=lambda i: i["score"], reverse=True)

    def statistics(self) -> dict[str, Any]:
        raw_docs = [path for path in (self.settings.data_dir / "raw" / "documents").glob("*") if path.is_file() and not path.name.startswith(".")]
        image_roots = [self.settings.data_dir / "raw" / "traffic_signs", self.settings.data_dir / "raw" / "images", self.settings.data_dir / "raw" / "accidents" / "images"]
        image_files = [path for root in image_roots if root.exists() for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")]
        return {"documents": len(raw_docs), "images": len(image_files), "traffic_signs": len(self.signs), "fine_records": len(self.fines), "rules": len(self.rules), "accident_records": len(self.accidents), "accident_images": len(self.accident_images), "verified_accident_image_pairs": sum(bool(image.get("record_id") and image.get("verified_record_match")) for image in self.accident_images), "generic_accident_images": sum(not image.get("record_id") for image in self.accident_images), "accident_statistics": len(self.accident_statistics), "latest_indexed_accident": max((item.get("date") or "" for item in self.accidents), default=None), "sources": len(self.sources), "official_sources": sum(1 for source in self.sources if source["official"]), "regions": sorted({source["region"] for source in self.sources}), "semantic_retrieval": "ready" if self._vectors is not None else "loads on first query", "semantic_error": self.semantic_error}

    def knowledge_status(self) -> dict[str, Any]:
        manifest_path = self.settings.data_dir / "processed" / "metadata" / "ingestion_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        return {"indexed": bool(manifest and manifest.get("status") == "complete"), "manifest": manifest, "statistics": self.statistics(), "image_embeddings": self.image_index.status(), "rag_storage_files": len([p for p in self.settings.rag_storage.rglob("*") if p.is_file()])}
